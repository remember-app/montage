import io
import json
import logging
import math
import pathlib
import tempfile
import uuid

import numpy as np
import psycopg
from PIL import Image
from psycopg.types.json import Jsonb

from src.integrations import ai, ai_usage, ffmpeg, storage
from src.prompts.analysis import (
    AUDIO_PROMPT,
    FINAL_VIDEO_PROMPT,
    INSTRUCTION,
    MUSIC_PROMPT,
    PHOTO_PROMPT,
    SEGMENT_PROMPT,
    SOUND_PROMPT,
    VIDEO_PROMPT,
)

log = logging.getLogger("media.analysis")

IMAGE_SIDE = 768
METRIC_SIDE = 256
SHARPNESS_FULL = 0.01  # ponytail: one calibration knob; tune it on real footage

FRAME_STEP_SECONDS = 5
FRAME_MIN = 3
FRAME_MAX = 60

SEGMENT_SECONDS = 20
SEGMENT_STEP_SECONDS = 2
SEGMENT_FRAME_MAX = 12

SUMMARY_LIMIT = 500
DESCRIPTION_LIMIT = 4000
SEGMENT_LIMIT = 500
COMMENT_LIMIT = 500

LEVELS = ("low", "medium", "high")
LEVEL_RANK = {"low": 0, "medium": 1, "high": 2}

IMPORTANCE_KEY = "analysis.grading.importance"
QUALITY_KEY = "analysis.grading.quality"
MUSIC_KEY = "music.grading"

USABLE_QUALITY = 3.0  # ponytail: one calibration knob; the bottom of the usable band


def _text(value, limit: int = DESCRIPTION_LIMIT) -> str | None:
    if not isinstance(value, str):
        return None
    text = "".join(character for character in value if character >= " ").strip()
    return text[:limit] or None


def _tags(value) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    tags = {}
    for key, item in value.items():
        if not isinstance(key, str) or item is None:
            continue
        if isinstance(item, (str, int, float, bool)):
            tags[key] = str(item)
        else:
            tags[key] = json.dumps(item, ensure_ascii=False)
    return tags or None


def _score(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 5.0
    return round(min(max(number, 0.0), 10.0), 2)


def _metrics(blob: ai.Blob) -> dict[str, str]:
    with Image.open(io.BytesIO(blob.data)) as opened:
        image = opened.convert("L")
        image.thumbnail((METRIC_SIDE, METRIC_SIDE))
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    laplacian = (
        -4.0 * pixels
        + np.roll(pixels, 1, axis=0)
        + np.roll(pixels, -1, axis=0)
        + np.roll(pixels, 1, axis=1)
        + np.roll(pixels, -1, axis=1)
    )[1:-1, 1:-1]
    clipping = float(((pixels < 0.02) | (pixels > 0.98)).mean())
    return {
        "sharpness": f"{min(float(laplacian.var()) / SHARPNESS_FULL, 1.0):.2f}",
        "brightness": f"{float(pixels.mean()):.2f}",
        "exposure": f"{1.0 - clipping:.2f}",
    }


def _average_metrics(measured: list[dict[str, str]]) -> dict[str, str]:
    if not measured:
        return {}
    return {
        key: f"{sum(float(one.get(key, 0.0)) for one in measured) / len(measured):.2f}"
        for key in measured[0]
    }


def _with_metrics(tags: dict[str, str] | None, measured: dict[str, str]) -> dict[str, str] | None:
    if not measured:
        return tags
    merged = dict(tags or {})
    merged.update(measured)
    return merged


def _grading(conn: psycopg.Connection, key: str) -> str:
    row = conn.execute("select value from settings where key = %s", (key,)).fetchone()
    if row is None or not isinstance(row[0], str) or not row[0].strip():
        raise RuntimeError(f"setting {key} is missing")
    return row[0].strip()


def _importance(grading: str, comment: str | None) -> str:
    return grading.replace("{comment}", f'"{comment}"' if comment else "nothing")


def _event_instruction(conn: psycopg.Connection, media_id: uuid.UUID) -> str:
    row = conn.execute(
        """
        select p.title, p.description
        from project_items pi
        join projects p on p.id = pi.project_id
        where pi.original_id = %s
        order by (p.description <> '') desc, pi.created_at
        limit 1
        """,
        (media_id,),
    ).fetchone()
    if row is None:
        return INSTRUCTION
    title, description = row
    description = description.strip()
    if description:
        return f'{INSTRUCTION}\nWhat this event is: "{title}". {description}'
    return f'{INSTRUCTION}\nWhat this event is: "{title}".'


def _spans(value, duration: float | None) -> list[dict] | None:
    if not isinstance(value, list):
        return None
    spans = []
    for one in value[:SEGMENT_LIMIT]:
        if not isinstance(one, dict):
            continue
        text = _text(one.get("description") or one.get("text"))
        if text is None:
            continue
        raw_start, raw_end = one.get("secStart"), one.get("secEnd")
        if raw_start is None or raw_end is None:
            continue
        try:
            start = max(float(raw_start), 0.0)
            end = max(float(raw_end), start)
        except (TypeError, ValueError):
            continue
        if duration:
            start, end = min(start, duration), min(end, duration)
        spans.append(
            {
                "secStart": round(start, 2),
                "secEnd": round(end, 2),
                "description": text,
                "tags": _tags(one.get("tags")),
            }
        )
    spans.sort(key=lambda one: one["secStart"])
    return spans


def _frames(value, spans: list[tuple[float, float]]) -> list[dict]:
    described = value if isinstance(value, list) else []
    frames = []
    for index, (start, end) in enumerate(spans):
        described_one = described[index] if index < len(described) else None
        one = described_one if isinstance(described_one, dict) else {}
        text = _text(one.get("description") or one.get("text"))
        if text is None:
            continue
        frames.append(
            {"secStart": start, "secEnd": end, "description": text, "tags": _tags(one.get("tags"))}
        )
    return frames


def _has_audio(path: pathlib.Path) -> bool:
    return any(
        stream.get("codec_type") == "audio" for stream in ffmpeg.probe(path).get("streams", [])
    )


def _audio_blob(source: pathlib.Path, workdir: pathlib.Path, speech: bool) -> ai.Blob:
    sound = workdir / "sound.m4a"
    if speech:
        options = ("-vn", "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "64k")
    else:
        options = ("-vn", "-ac", "2", "-ar", "44100", "-c:a", "aac", "-b:a", "128k")
    ffmpeg.run("-i", str(source), *options, str(sound))
    return ai.Blob("audio/mp4", sound.read_bytes())


def _image_blob(source: pathlib.Path, workdir: pathlib.Path) -> ai.Blob:
    jpeg = workdir / "still.jpg"
    ffmpeg.run(
        "-i",
        str(source),
        "-vf",
        f"scale={IMAGE_SIDE}:{IMAGE_SIDE}:force_original_aspect_ratio=decrease",
        "-frames:v",
        "1",
        "-q:v",
        "3",
        str(jpeg),
    )
    return ai.Blob("image/jpeg", jpeg.read_bytes())


def _still(source: pathlib.Path, moment: float, workdir: pathlib.Path, name: str) -> ai.Blob:
    frame = workdir / f"{name}.jpg"
    ffmpeg.run(
        "-ss",
        f"{moment:.3f}",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-vf",
        f"scale={IMAGE_SIDE}:{IMAGE_SIDE}:force_original_aspect_ratio=decrease",
        "-q:v",
        "3",
        str(frame),
    )
    return ai.Blob("image/jpeg", frame.read_bytes())


def _spaced_moments(length: float, step: float, minimum: int, maximum: int) -> list[float]:
    count = minimum if length <= 0 else max(minimum, min(maximum, math.ceil(length / step)))
    return [round(length * (index + 0.5) / count, 2) for index in range(count)]


def _summary(answer: dict, description: str) -> str:
    return _text(answer.get("summary"), SUMMARY_LIMIT) or description


def _photo_analysis(
    conn: psycopg.Connection,
    job_id: int,
    media_id: uuid.UUID,
    source: pathlib.Path,
    workdir: pathlib.Path,
    importance: str,
    quality: str,
) -> dict:
    blob = _image_blob(source, workdir)
    answer = ai_usage.as_json(
        conn,
        operation="photo",
        job_id=job_id,
        instruction=_event_instruction(conn, media_id),
        prompt=f"{PHOTO_PROMPT}\n{importance}\n{quality}",
        blobs=(blob,),
    )
    description = _text(answer.get("description"))
    if description is None:
        raise RuntimeError("the model described nothing")
    return {
        "level": "high",
        "summary": _summary(answer, description),
        "description": description,
        "scoreImportance": _score(answer.get("scoreImportance")),
        "scoreQuality": _score(answer.get("scoreQuality")),
        "tags": _with_metrics(_tags(answer.get("tags")), _metrics(blob)),
    }


def _video_low(
    conn: psycopg.Connection,
    job_id: int,
    media_id: uuid.UUID,
    source: pathlib.Path,
    duration: float | None,
    workdir: pathlib.Path,
    importance: str,
    quality: str,
    instruction: str,
) -> dict:
    length = duration or ffmpeg.duration(source) or 0.0
    moments = _spaced_moments(length, FRAME_STEP_SECONDS, FRAME_MIN, FRAME_MAX)
    spans = [(moment, min(moment + FRAME_STEP_SECONDS, length or moment)) for moment in moments]
    blobs = []
    measured = []
    for index, moment in enumerate(moments):
        blob = _still(source, moment, workdir, f"frame-{index}")
        blobs.append(blob)
        measured.append(_metrics(blob))
    prompt = VIDEO_PROMPT.replace("{seconds}", ", ".join(f"{one:.2f}" for one in moments))
    answer = ai_usage.as_json(
        conn,
        operation="video",
        job_id=job_id,
        instruction=instruction,
        prompt=f"{prompt}\n{importance}\n{quality}",
        blobs=tuple(blobs),
    )
    description = _text(answer.get("description"))
    if description is None:
        raise RuntimeError("the model described nothing")
    frames = _frames(answer.get("frames"), spans)
    by_start = {span[0]: metric for span, metric in zip(spans, measured)}
    for frame in frames:
        frame["tags"] = _with_metrics(frame["tags"], by_start.get(frame["secStart"], {}))
    if not frames:
        frames = [
            {
                "secStart": 0.0,
                "secEnd": round(length, 2),
                "description": description,
                "tags": _average_metrics(measured),
            }
        ]
    return {
        "summary": _summary(answer, description),
        "description": description,
        "scoreImportance": _score(answer.get("scoreImportance")),
        "scoreQuality": _score(answer.get("scoreQuality")),
        "tags": _tags(answer.get("tags")),
        "frames": frames,
    }


def _segment_spans(length: float) -> list[tuple[float, float]]:
    spans = []
    start = 0.0
    while start < length:
        end = min(start + SEGMENT_SECONDS, length)
        spans.append((round(start, 2), round(end, 2)))
        start = end
    return spans


def _file_quality(scored: list[tuple[float, float]]) -> float:
    if len(scored) <= 1:
        return scored[0][1] if scored else 5.0
    total = sum(duration for duration, _ in scored)
    if total <= 0:
        return 5.0
    usable = [(duration, quality) for duration, quality in scored if quality >= USABLE_QUALITY]
    if not usable:
        return 0.0
    usable_seconds = sum(duration for duration, _ in usable)
    mean = sum(duration * quality for duration, quality in usable) / usable_seconds
    return round(usable_seconds / total * mean, 2)


def _video_medium(
    conn: psycopg.Connection,
    job_id: int,
    media_id: uuid.UUID,
    source: pathlib.Path,
    duration: float | None,
    workdir: pathlib.Path,
    importance: str,
    quality: str,
    instruction: str,
) -> dict:
    length = duration or ffmpeg.duration(source) or 0.0
    if length <= 0:
        return _video_low(
            conn, job_id, media_id, source, duration, workdir, importance, quality, instruction
        )

    frames = []
    scored = []
    context = ""
    for index, (start, end) in enumerate(_segment_spans(length)):
        moments = _spaced_moments(end - start, SEGMENT_STEP_SECONDS, 2, SEGMENT_FRAME_MAX)
        moments = [round(start + moment, 2) for moment in moments]
        blobs = []
        measured = []
        for order, moment in enumerate(moments):
            blob = _still(source, moment, workdir, f"segment-{index}-{order}")
            blobs.append(blob)
            measured.append(_metrics(blob))
        prompt = SEGMENT_PROMPT.replace(
            "{seconds}", ", ".join(f"{one:.2f}" for one in moments)
        ).replace("{context}", context or "nothing yet")
        answer = ai_usage.as_json(
            conn,
            operation="video",
            job_id=job_id,
            instruction=instruction,
            prompt=f"{prompt}\n{quality}",
            blobs=tuple(blobs),
        )
        description = _text(answer.get("description"))
        if description is None:
            raise RuntimeError(f"the model described nothing in segment {index}")
        segment_quality = _score(answer.get("scoreQuality"))
        tags = _tags(answer.get("tags")) or {}
        tags["scoreQuality"] = str(segment_quality)
        tags = _with_metrics(tags, _average_metrics(measured)) or tags
        frames.append({"secStart": start, "secEnd": end, "description": description, "tags": tags})
        scored.append((end - start, segment_quality))
        context = _text(answer.get("context"), SUMMARY_LIMIT) or description

    summary_input = json.dumps(
        [
            {"secStart": one["secStart"], "secEnd": one["secEnd"], "text": one["description"]}
            for one in frames
        ],
        ensure_ascii=False,
    )
    prompt = FINAL_VIDEO_PROMPT.replace("{segments}", summary_input)
    answer = ai_usage.as_json(
        conn,
        operation="text",
        job_id=job_id,
        instruction=instruction,
        prompt=f"{prompt}\n{importance}",
    )
    description = _text(answer.get("description")) or frames[-1]["description"]
    return {
        "summary": _summary(answer, description),
        "description": description,
        "scoreImportance": _score(answer.get("scoreImportance")),
        "scoreQuality": _file_quality(scored),
        "tags": _tags(answer.get("tags")),
        "frames": frames,
    }


def _video_high(*_args, **_kwargs) -> dict:
    raise RuntimeError("high video analysis is not implemented yet")


def _sound_analysis(
    conn: psycopg.Connection,
    job_id: int,
    media_id: uuid.UUID,
    source: pathlib.Path,
    duration: float | None,
    workdir: pathlib.Path,
    instruction: str,
) -> tuple[str | None, dict | None, list[dict] | None, list[dict] | None]:
    answer = ai_usage.as_json(
        conn,
        operation="video",
        job_id=job_id,
        instruction=instruction,
        prompt=SOUND_PROMPT,
        blobs=(_audio_blob(source, workdir, speech=True),),
    )
    return (
        _text(answer.get("description")),
        _tags(answer.get("tags")),
        _spans(answer.get("sounds"), duration),
        _spans(answer.get("words"), duration),
    )


def _video_analysis(
    conn: psycopg.Connection,
    job_id: int,
    media_id: uuid.UUID,
    source: pathlib.Path,
    duration: float | None,
    workdir: pathlib.Path,
    importance: str,
    quality: str,
    level: str,
) -> dict:
    instruction = _event_instruction(conn, media_id)
    if level == "high":
        result = _video_high(conn, media_id, source, duration, workdir, importance, quality)
    elif level == "medium":
        result = _video_medium(
            conn, job_id, media_id, source, duration, workdir, importance, quality, instruction
        )
    else:
        result = _video_low(
            conn, job_id, media_id, source, duration, workdir, importance, quality, instruction
        )

    tags = result["tags"]
    sounds = words = None
    if _has_audio(source):
        sound_description, sound_tags, sounds, words = _sound_analysis(
            conn, job_id, media_id, source, duration, workdir, instruction
        )
        tags = tags or {}
        if sound_description:
            tags["soundDescription"] = sound_description
        if sound_tags:
            tags.update({f"sound.{key}": value for key, value in sound_tags.items()})

    return {
        "level": level,
        "summary": result["summary"],
        "description": result["description"],
        "scoreImportance": result["scoreImportance"],
        "scoreQuality": result["scoreQuality"],
        "tags": tags,
        "frames": result["frames"],
        "sounds": sounds,
        "words": words,
    }


def _audio_analysis(
    conn: psycopg.Connection,
    job_id: int,
    media_id: uuid.UUID,
    source: pathlib.Path,
    duration: float | None,
    workdir: pathlib.Path,
    importance: str,
    quality: str,
) -> dict:
    answer = ai_usage.as_json(
        conn,
        operation="audio",
        job_id=job_id,
        instruction=_event_instruction(conn, media_id),
        prompt=f"{AUDIO_PROMPT}\n{importance}\n{quality}",
        blobs=(_audio_blob(source, workdir, speech=True),),
    )
    description = _text(answer.get("description"))
    if description is None:
        raise RuntimeError("the model described nothing")
    return {
        "level": "high",
        "summary": _summary(answer, description),
        "description": description,
        "scoreImportance": _score(answer.get("scoreImportance")),
        "scoreQuality": _score(answer.get("scoreQuality")),
        "tags": _tags(answer.get("tags")),
        "sounds": _spans(answer.get("sounds"), duration),
        "words": _spans(answer.get("words"), duration),
    }


def _music_analysis(
    conn: psycopg.Connection,
    job_id: int,
    media_id: uuid.UUID,
    source: pathlib.Path,
    duration: float | None,
    workdir: pathlib.Path,
    grading: str,
) -> dict:
    answer = ai_usage.as_json(
        conn,
        operation="audio",
        job_id=job_id,
        instruction=INSTRUCTION,
        prompt=f"{MUSIC_PROMPT}\n{grading}",
        blobs=(_audio_blob(source, workdir, speech=False),),
    )
    description = _text(answer.get("description"))
    if description is None:
        raise RuntimeError("the model described nothing")
    return {
        "level": "high",
        "summary": _summary(answer, description),
        "description": description,
        "score": _score(answer.get("score")),
        "tags": _tags(answer.get("tags")),
        "sounds": _spans(answer.get("sounds"), duration),
        "words": _spans(answer.get("words"), duration),
    }


def _download(key: str, workdir: pathlib.Path, name: str) -> pathlib.Path:
    path = workdir / name
    storage.download(key, path)
    ffmpeg.refuse_playlists(path)
    return path


def _item_exists(conn: psycopg.Connection, media_id: uuid.UUID) -> bool:
    [row] = conn.execute(
        "select count(*) from project_items where original_id = %s", (media_id,)
    ).fetchone()
    return row > 0


def _song_exists(conn: psycopg.Connection, media_id: uuid.UUID) -> bool:
    [row] = conn.execute(
        "select count(*) from songs where original_id = %s", (media_id,)
    ).fetchone()
    return row > 0


def _store_item(conn: psycopg.Connection, media_id: uuid.UUID, analysis: dict) -> int:
    rank = LEVEL_RANK.get(str(analysis.get("level")), 0)
    landed = conn.execute(
        """
        update project_items
        set analysis = %s
        where original_id = %s
          and (analysis is null
               or case analysis->>'level'
                    when 'low' then 0
                    when 'medium' then 1
                    when 'high' then 2
                    else 0
                  end < %s)
        """,
        (Jsonb(analysis), media_id, rank),
    ).rowcount
    return landed


def _store_song(conn: psycopg.Connection, media_id: uuid.UUID, analysis: dict) -> int:
    rank = LEVEL_RANK.get(str(analysis.get("level")), 0)
    landed = conn.execute(
        """
        update songs
        set analysis = %s
        where original_id = %s
          and (analysis is null
               or case analysis->>'level'
                    when 'low' then 0
                    when 'medium' then 1
                    when 'high' then 2
                    else 0
                  end < %s)
        """,
        (Jsonb(analysis), media_id, rank),
    ).rowcount
    return landed


def _video_level(level: str | None) -> str:
    return level if level in LEVELS else "medium"


def run(conn: psycopg.Connection, ref: uuid.UUID, level: str | None, job_id: int) -> None:
    item = _item_exists(conn, ref)
    song = _song_exists(conn, ref)
    if not item and not song:
        log.info("media %s needs no analysis", ref)
        return

    row = conn.execute(
        """
        select m.kind, m.storage_key, nullif(m.meta->>'duration', '')::float8,
               (select pm.storage_key from project_items pi
                  join media_items pm on pm.id = pi.compressed_id
                 where pi.original_id = m.id limit 1) as compressed_key,
               (select pi.comment from project_items pi
                 where pi.original_id = m.id and pi.comment is not null limit 1) as comment
        from media_items m where m.id = %s
        """,
        (ref,),
    ).fetchone()
    if row is None:
        log.info("media %s is gone, nothing to analyze", ref)
        return
    kind, storage_key, duration, compressed_key, raw_comment = row
    comment = _text(raw_comment, COMMENT_LIMIT)

    with tempfile.TemporaryDirectory() as directory:
        workdir = pathlib.Path(directory)
        if item:
            importance = _importance(_grading(conn, IMPORTANCE_KEY), comment)
            quality = _grading(conn, QUALITY_KEY)
            source_key = compressed_key or storage_key
            source = _download(source_key, workdir, "item-source")
            if kind == "photo":
                analysis = _photo_analysis(conn, job_id, ref, source, workdir, importance, quality)
            elif kind == "video":
                analysis = _video_analysis(
                    conn, job_id, ref, source, duration, workdir, importance, quality,
                    _video_level(level),
                )
            else:
                analysis = _audio_analysis(
                    conn, job_id, ref, source, duration, workdir, importance, quality
                )
            landed = _store_item(conn, ref, analysis)
            log.info("media %s analyzed for %s item(s)", ref, landed)

        if song:
            grading = _grading(conn, MUSIC_KEY)
            source = _download(storage_key, workdir, "song-source")
            analysis = _music_analysis(conn, job_id, ref, source, duration, workdir, grading)
            landed = _store_song(conn, ref, analysis)
            log.info("media %s analyzed for %s song(s)", ref, landed)
