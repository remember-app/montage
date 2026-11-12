import logging
import math
import pathlib
import tempfile
import uuid

import psycopg
from psycopg.types.json import Jsonb

from src import config
from src.integrations import ai, ai_usage, ffmpeg, storage

log = logging.getLogger("media.analysis")

IMAGE_SIDE = 768
FRAME_STEP_SECONDS = 5
FRAME_MIN = 3
FRAME_MAX = 60
DESCRIPTION_LIMIT = 2000
SEGMENT_LIMIT = 500
COMMENT_LIMIT = 500

GRADING_KEY = "analysis.grading"


INSTRUCTION = (
    "You analyze frames shot by people at the same event. Answer in English and reply with a "
    "JSON object only."
)
PHOTO_PROMPT = (
    "Describe in one or two sentences what is happening in the photo. "
    'Reply: {"description": "...", "score": 0.0}'
)
VIDEO_PROMPT = (
    "These are several frames from one video, taken at seconds {seconds}. Describe in one or two "
    "sentences what is happening in the video. Describe each frame separately, in the same order, "
    'with its second. Reply: {"description": "...", "score": 0.0, "frames": ['
    '{"second": 0.0, "text": "..."}]}'
)
TRANSCRIBE_PROMPT = (
    "Transcribe the speech in the recording and split it into time segments in seconds. Return "
    "only what is said, without comments. If no speech is audible, return an empty list. "
    'Reply: {"transcription": [{"secStart": 0.0, "secEnd": 0.0, "text": "..."}]}'
)
AUDIO_PROMPT = (
    "Describe in one sentence what is heard in the recording. Transcribe the speech and split it "
    'into time segments in seconds. Reply: {"description": "...", "score": 0.0, "transcription": ['
    '{"secStart": 0.0, "secEnd": 0.0, "text": "..."}]}'
)
MUSIC_PROMPT = (
    "This is a music track. Describe it in one sentence and score how useful it is for a movie "
    "from 0.00 to 10.00 with two decimal places: 0 is unusable, 10 is a perfect fit. Split it into "
    "musical sections and briefly describe each: mood, instruments, what changes. Transcribe the "
    "lyrics if any, with time segments in seconds. "
    'Reply: {"description": "...", "score": 0.0, '
    '"transcription": [{"secStart": 0.0, "secEnd": 0.0, "text": "..."}], '
    '"mood": [{"secStart": 0.0, "secEnd": 0.0, "text": "..."}]}'
)


def _said(value, limit: int = DESCRIPTION_LIMIT) -> str | None:
    if not isinstance(value, str):
        return None
    text = "".join(character for character in value if character >= " ").strip()
    return text[:limit] or None


def _score(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 5.0
    return round(min(max(number, 0.0), 10.0), 2)


def _grading(conn: psycopg.Connection) -> str:
    row = conn.execute("select value from settings where key = %s", (GRADING_KEY,)).fetchone()
    if row is None or not isinstance(row[0], list):
        raise RuntimeError(f"setting {GRADING_KEY} is missing")
    lines = []
    for one in row[0]:
        if not isinstance(one, dict):
            continue
        points, question = one.get("points"), one.get("question")
        if isinstance(points, (int, float)) and isinstance(question, str) and question.strip():
            unit = "point" if points == 1 else "points"
            lines.append(f"- ({points:g} {unit}) {question.strip()}")
    if not lines:
        raise RuntimeError(f"setting {GRADING_KEY} has no usable questions")
    return "\n".join(lines)


def _graded(grading: str, comment: str | None) -> str:
    block = (
        "Score the frame from 0.00 to 10.00 with two decimal places, answering these questions:\n"
        + grading
    )
    if comment:
        block += f'\nThe uploader wrote about this frame: "{comment}"'
    return block


def _segments(value, duration: float | None) -> list[dict]:
    if not isinstance(value, list):
        return []
    segments = []
    for one in value[:SEGMENT_LIMIT]:
        if not isinstance(one, dict):
            continue
        text = _said(one.get("text"))
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
        segments.append({"secStart": round(start, 2), "secEnd": round(end, 2), "text": text})
    segments.sort(key=lambda one: one["secStart"])
    return segments


def _frames(value, seconds: list[float]) -> list[dict]:
    described = {}
    if isinstance(value, list):
        for one in value:
            if not isinstance(one, dict):
                continue
            text = _said(one.get("text"))
            raw_second = one.get("second")
            if raw_second is None:
                continue
            try:
                second = round(float(raw_second), 2)
            except (TypeError, ValueError):
                continue
            if text is not None:
                described[second] = text
    return [
        {"second": second, "text": described.get(second, "")}
        for second in seconds
        if described.get(second)
    ]


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


def _frame_blobs(
    source: pathlib.Path, duration: float | None, workdir: pathlib.Path
) -> tuple[list[ai.Blob], list[float]]:
    length = duration or ffmpeg.duration(source) or 0.0
    count = (
        FRAME_MIN
        if length <= 0
        else max(FRAME_MIN, min(FRAME_MAX, math.ceil(length / FRAME_STEP_SECONDS)))
    )
    moments = [round(length * (index + 0.5) / count, 2) for index in range(count)]
    blobs, seconds = [], []
    for index, moment in enumerate(moments):
        frame = workdir / f"frame-{index}.jpg"
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
        blobs.append(ai.Blob("image/jpeg", frame.read_bytes()))
        seconds.append(moment)
    return blobs, seconds


def _photo_analysis(
    conn: psycopg.Connection,
    media_id: uuid.UUID,
    source: pathlib.Path,
    workdir: pathlib.Path,
    grading: str,
    comment: str | None,
) -> dict:
    if config.ai_mock():
        return {"description": "Mock: a frame from the event.", "score": 5.0}
    blob = _image_blob(source, workdir)
    answer = ai_usage.as_json(
        conn,
        operation="photo",
        media_id=media_id,
        instruction=INSTRUCTION,
        prompt=f"{PHOTO_PROMPT}\n{_graded(grading, comment)}",
        blobs=(blob,),
    )
    description = _said(answer.get("description"))
    if description is None:
        raise RuntimeError("the model described nothing")
    return {"description": description, "score": _score(answer.get("score"))}


def _video_analysis(
    conn: psycopg.Connection,
    media_id: uuid.UUID,
    source: pathlib.Path,
    duration: float | None,
    workdir: pathlib.Path,
    grading: str,
    comment: str | None,
) -> dict:
    if config.ai_mock():
        return {
            "description": "Mock: a clip from the event.",
            "score": 5.0,
            "transcription": [],
            "frames": [],
        }
    blobs, seconds = _frame_blobs(source, duration, workdir)
    prompt = VIDEO_PROMPT.replace("{seconds}", ", ".join(f"{one:.2f}" for one in seconds))
    answer = ai_usage.as_json(
        conn,
        operation="video",
        media_id=media_id,
        instruction=INSTRUCTION,
        prompt=f"{prompt}\n{_graded(grading, comment)}",
        blobs=tuple(blobs),
    )
    description = _said(answer.get("description"))
    if description is None:
        raise RuntimeError("the model described nothing")
    transcription = []
    if _has_audio(source):
        heard = ai_usage.as_json(
            conn,
            operation="video",
            media_id=media_id,
            instruction=INSTRUCTION,
            prompt=TRANSCRIBE_PROMPT,
            blobs=(_audio_blob(source, workdir, speech=True),),
        )
        transcription = _segments(heard.get("transcription"), duration)
    return {
        "description": description,
        "score": _score(answer.get("score")),
        "transcription": transcription,
        "frames": _frames(answer.get("frames"), seconds),
    }


def _audio_analysis(
    conn: psycopg.Connection,
    media_id: uuid.UUID,
    source: pathlib.Path,
    duration: float | None,
    workdir: pathlib.Path,
    grading: str,
    comment: str | None,
) -> dict:
    if config.ai_mock():
        return {
            "description": "Mock: a sound from the event.",
            "score": 5.0,
            "transcription": [],
            "mood": None,
        }
    answer = ai_usage.as_json(
        conn,
        operation="audio",
        media_id=media_id,
        instruction=INSTRUCTION,
        prompt=f"{AUDIO_PROMPT}\n{_graded(grading, comment)}",
        blobs=(_audio_blob(source, workdir, speech=True),),
    )
    description = _said(answer.get("description"))
    if description is None:
        raise RuntimeError("the model described nothing")
    return {
        "description": description,
        "score": _score(answer.get("score")),
        "transcription": _segments(answer.get("transcription"), duration),
        "mood": None,
    }


def _music_analysis(
    conn: psycopg.Connection,
    media_id: uuid.UUID,
    source: pathlib.Path,
    duration: float | None,
    workdir: pathlib.Path,
) -> dict:
    if config.ai_mock():
        return {
            "description": "Mock: a music track.",
            "score": 5.0,
            "transcription": [],
            "mood": [],
        }
    answer = ai_usage.as_json(
        conn,
        operation="audio",
        media_id=media_id,
        instruction=INSTRUCTION,
        prompt=MUSIC_PROMPT,
        blobs=(_audio_blob(source, workdir, speech=False),),
    )
    description = _said(answer.get("description"))
    if description is None:
        raise RuntimeError("the model described nothing")
    return {
        "description": description,
        "score": _score(answer.get("score")),
        "transcription": _segments(answer.get("transcription"), duration),
        "mood": _segments(answer.get("mood"), duration),
    }


def _download(key: str, workdir: pathlib.Path, name: str) -> pathlib.Path:
    path = workdir / name
    storage.download(key, path)
    ffmpeg.refuse_playlists(path)
    return path


def _item_pending(conn: psycopg.Connection, media_id: uuid.UUID) -> bool:
    [row] = conn.execute(
        "select count(*) from project_items where media_id = %s and analysis is null", (media_id,)
    ).fetchone()
    return row > 0


def _song_pending(conn: psycopg.Connection, media_id: uuid.UUID) -> bool:
    [row] = conn.execute(
        "select count(*) from songs where media_id = %s and analysis is null", (media_id,)
    ).fetchone()
    return row > 0


def run(conn: psycopg.Connection, ref: uuid.UUID) -> None:
    item = _item_pending(conn, ref)
    song = _song_pending(conn, ref)
    if not item and not song:
        log.info("media %s needs no analysis", ref)
        return

    row = conn.execute(
        """
        select m.kind, m.storage_key, m.duration,
               (select pm.storage_key from project_items pi
                  join media_items pm on pm.id = pi.preview_id
                 where pi.media_id = m.id limit 1) as preview_key,
               (select pi.description from project_items pi
                 where pi.media_id = m.id and pi.description is not null limit 1) as comment
        from media_items m where m.id = %s
        """,
        (ref,),
    ).fetchone()
    if row is None:
        log.info("media %s is gone, nothing to analyze", ref)
        return
    kind, storage_key, duration, preview_key, raw_comment = row
    comment = _said(raw_comment, COMMENT_LIMIT)

    with tempfile.TemporaryDirectory() as directory:
        workdir = pathlib.Path(directory)
        if item:
            grading = _grading(conn)
            source_key = preview_key or storage_key
            source = _download(source_key, workdir, "item-source")
            if kind == "photo":
                analysis = _photo_analysis(conn, ref, source, workdir, grading, comment)
            elif kind == "video":
                analysis = _video_analysis(conn, ref, source, duration, workdir, grading, comment)
            else:
                analysis = _audio_analysis(conn, ref, source, duration, workdir, grading, comment)
            landed = conn.execute(
                "update project_items set analysis = %s where media_id = %s and analysis is null",
                (Jsonb(analysis), ref),
            ).rowcount
            log.info("media %s analyzed for %s item(s)", ref, landed)

        if song:
            source = _download(storage_key, workdir, "song-source")
            analysis = _music_analysis(conn, ref, source, duration, workdir)
            landed = conn.execute(
                "update songs set analysis = %s where media_id = %s and analysis is null",
                (Jsonb(analysis), ref),
            ).rowcount
            log.info("media %s analyzed for %s song(s)", ref, landed)
