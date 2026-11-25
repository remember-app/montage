import dataclasses
import json
import logging
import math
import pathlib
import tempfile
import uuid

import psycopg
from psycopg.types.json import Jsonb

from src import config
from src.integrations import ffmpeg, storage
from src.movie import notify
from src.movie.types import EditPlan, ShotCommand

log = logging.getLogger("movie.editor")

FPS = 30
CRF = "20"
AUDIO_RATE = "44100"
AUDIO_BITRATE = "192k"
TARGET_ALLOWANCE = 0.1
TARGET_ALLOWANCE_MIN = 2.0
ZOOM_SMOOTH = 4
CONCAT_GROUP = 4

EDGE_FADE = 0.05
MUSIC_TAIL = 1.5
DUCK_THRESHOLD = 0.05  # ponytail: one calibration knob; tune it on real event audio
DUCK_RATIO = 8
DUCK_ATTACK = 20
DUCK_RELEASE = 400
LOUDNESS = "loudnorm=I=-14:TP=-1.5:LRA=11"


@dataclasses.dataclass
class Item:
    item_id: str
    media_id: str
    kind: str
    key: str
    duration: float | None
    cover_key: str | None
    preview_key: str | None


@dataclasses.dataclass
class Song:
    media_id: str
    key: str
    duration: float | None


@dataclasses.dataclass
class Montage:
    id: uuid.UUID
    project_id: uuid.UUID
    created_by: uuid.UUID
    title: str
    settings: dict


def _f(value) -> str:
    return f"{float(value):.3f}"


def _gain(cue) -> float:
    value = cue.get("gain")
    return 1.0 if value is None else float(value)


def _declick(seconds: float) -> str:
    fade = min(EDGE_FADE, seconds / 2)
    return f"afade=t=in:st=0:d={_f(fade)},afade=t=out:st={_f(max(seconds - fade, 0))}:d={_f(fade)}"


def overlap(shot: ShotCommand) -> float:
    return float(shot.get("transitionSeconds") or 0)


def timeline_length(shots: list[ShotCommand]) -> float:
    total = 0.0
    for index, shot in enumerate(shots):
        total += float(shot["seconds"])
        if index < len(shots) - 1:
            total -= overlap(shot)
    return total


def _load(conn: psycopg.Connection, montage_id: uuid.UUID) -> Montage:
    row = conn.execute(
        """
        select m.id, m.project_id, m.created_by, m.manifest, p.title
        from montages m join projects p on p.id = m.project_id
        where m.id = %s
        """,
        (montage_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"montage {montage_id} is gone")
    identifier, project_id, created_by, manifest, title = row
    settings = (manifest or {}).get("preparationConfig") or {}
    name = settings.get("name") or title or "Movie"
    return Montage(identifier, project_id, created_by, name, settings)


def _items(conn: psycopg.Connection, item_ids: list[str]) -> dict[str, Item]:
    if not item_ids:
        return {}
    rows = conn.execute(
        """
        select pi.id, pi.original_id, pi.kind, m.storage_key,
               nullif(m.meta->>'duration', '')::float8,
               (select pm.storage_key from media_items pm
                 where pm.id = (pi.previews_ids->>'small')::uuid),
               (select pm.storage_key from media_items pm where pm.id = pi.compressed_id)
        from project_items pi join media_items m on m.id = pi.original_id
        where pi.id = any(%s)
        """,
        ([uuid.UUID(item_id) for item_id in item_ids],),
    ).fetchall()
    return {
        str(item_id): Item(
            str(item_id), str(media_id), kind, key, duration, cover_key, preview_key
        )
        for item_id, media_id, kind, key, duration, cover_key, preview_key in rows
    }


def _songs(conn: psycopg.Connection, media_ids: list[str]) -> dict[str, Song]:
    if not media_ids:
        return {}
    rows = conn.execute(
        """
        select s.original_id, m.storage_key, nullif(m.meta->>'duration', '')::float8
        from songs s join media_items m on m.id = s.original_id
        where s.original_id = any(%s)
        """,
        ([uuid.UUID(media_id) for media_id in media_ids],),
    ).fetchall()
    return {str(media_id): Song(str(media_id), key, duration) for media_id, key, duration in rows}


def _cover(conn: psycopg.Connection) -> str | None:
    raw = config.audio_cover_media_id()
    if not raw:
        return None
    row = conn.execute(
        "select storage_key from media_items where id = %s", (uuid.UUID(raw),)
    ).fetchone()
    return row[0] if row else None


def _validate(
    plan: EditPlan, montage: Montage, items: dict[str, Item], songs: dict[str, Song]
) -> None:
    issues: list[str] = []
    shots = [command for command in plan if command.get("type") == "shot"]
    music = [command for command in plan if command.get("type") == "music"]
    if not shots:
        issues.append("the plan has no shots")
    included = set(montage.settings.get("includeItemIds") or [])
    for command in plan:
        if command.get("type") not in ("shot", "music"):
            issues.append(f"unknown command {command.get('type')!r}")
    for shot in shots:
        item = items.get(str(shot.get("itemId")))
        if item is None:
            issues.append(f"item {shot.get('itemId')} is not in this project")
            continue
        if str(shot.get("mediaId")) != item.media_id:
            issues.append(f"item {item.item_id} has another media id")
        if shot.get("kind") != item.kind:
            issues.append(f"item {item.item_id} has another kind")
        if included and item.item_id not in included:
            issues.append(f"item {item.item_id} is not among the included items")
        seconds = float(shot.get("seconds") or 0)
        start = float(shot.get("sourceStartSeconds") or 0)
        if seconds <= 0.2:
            issues.append(f"item {item.item_id} is too short")
        if start < 0:
            issues.append(f"item {item.item_id} starts before the file")
        if item.duration and start + seconds > item.duration + 0.05:
            issues.append(f"item {item.item_id} runs past the file")
        live = float(shot.get("liveLevel") or 0)
        if not 0 <= live <= 1:
            issues.append(f"item {item.item_id} has live level outside 0..1")
        framing = shot.get("framing")
        if framing is not None:
            values = [framing.get(side) for side in ("x", "y", "w", "h")]
            if any(value is None for value in values):
                issues.append(f"item {item.item_id} has an incomplete framing")
            else:
                x, y, w, h = (float(value) for value in values)
                if min(x, y, w, h) < 0 or w <= 0 or h <= 0 or x + w > 1.001 or y + h > 1.001:
                    issues.append(f"item {item.item_id} has framing outside the frame")
        if shot.get("motion") not in ("none", "zoom_in", "zoom_out"):
            issues.append(f"item {item.item_id} has an unknown motion")
        role = shot.get("role")
        if not isinstance(role, str) or not role.strip():
            issues.append(f"item {item.item_id} has no role")
    for index, shot in enumerate(shots[:-1]):
        following = float(shots[index + 1]["seconds"])
        transition = overlap(shot)
        if transition < 0:
            issues.append(f"item {shot.get('itemId')} has a negative transition")
        elif 0 < transition < 1 / FPS:
            issues.append(f"item {shot.get('itemId')} has a transition shorter than one frame")
        elif transition >= min(float(shot["seconds"]), following):
            issues.append(f"item {shot.get('itemId')} overlaps the next shot entirely")
    required = set(montage.settings.get("requiredItemIds") or [])
    used = {str(shot.get("itemId")) for shot in shots}
    for item_id in required - used:
        issues.append(f"required item {item_id} is missing")
    total = timeline_length(shots)
    target = float(montage.settings.get("targetSeconds") or 0)
    if target > 0:
        allowance = max(TARGET_ALLOWANCE_MIN, target * TARGET_ALLOWANCE)
        if abs(total - target) > allowance:
            issues.append(
                f"the plan is {total:.1f} seconds long but the movie was ordered for "
                f"{target:.0f} seconds"
            )
    pool = set(montage.settings.get("musicIds") or [])
    for cue in music:
        song = songs.get(str(cue.get("mediaId")))
        if song is None:
            issues.append(f"song {cue.get('mediaId')} is not in this project")
            continue
        if pool and song.media_id not in pool:
            issues.append(f"song {song.media_id} is not among the chosen tracks")
        seconds = float(cue.get("seconds") or 0)
        start = float(cue.get("sourceStartSeconds") or 0)
        at = float(cue.get("atSeconds") or 0)
        if seconds <= 0 or start < 0 or at < 0:
            issues.append(f"song {song.media_id} has a negative or empty cue")
        if song.duration and start + seconds > song.duration + 0.05:
            issues.append(f"song {song.media_id} runs past the file")
        if at + seconds > total + 0.05:
            issues.append(f"song {song.media_id} plays past the movie")
        gain = _gain(cue)
        if not 0 <= gain <= 2:
            issues.append(f"song {song.media_id} has gain outside 0..2")
        for side in ("fadeInSeconds", "fadeOutSeconds"):
            fade = float(cue.get(side) or 0)
            if fade < 0 or fade > seconds:
                issues.append(f"song {song.media_id} has {side} outside its cue")
    if issues:
        raise RuntimeError("; ".join(issues))


def _fit_size(
    shot: ShotCommand, source: pathlib.Path, width: int, height: int
) -> tuple[int, int]:
    source_width, source_height = ffmpeg.dimensions(source) or (width, height)
    framing = shot.get("framing")
    if framing:
        source_width *= float(framing["w"])
        source_height *= float(framing["h"])
    scale = min(width / source_width, height / source_height)
    return max(1, round(source_width * scale)), max(1, round(source_height * scale))


def _video_filters(
    shot: ShotCommand, width: int, height: int, fit_width: int, fit_height: int
) -> list[str]:
    filters = []
    framing = shot.get("framing")
    if framing:
        filters.append(
            "crop=iw*{w}:ih*{h}:iw*{x}:ih*{y}".format(
                x=_f(framing["x"]), y=_f(framing["y"]), w=_f(framing["w"]), h=_f(framing["h"])
            )
        )
    motion = shot.get("motion") or "none"
    if motion != "none" and shot.get("kind") in ("photo", "audio"):
        filters.append(f"scale={fit_width * ZOOM_SMOOTH}:{fit_height * ZOOM_SMOOTH}")
        zoom = "min(1.0+0.0008*on,1.15)" if motion == "zoom_in" else "max(1.15-0.0008*on,1.0)"
        filters.append(
            f"zoompan=z='{zoom}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d=1:s={fit_width}x{fit_height}:fps={FPS}"
        )
    else:
        filters.append(f"scale={fit_width}:{fit_height}")
    filters.append("setsar=1")
    filters.append(f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black")
    filters.append(f"fps={FPS}")
    return filters


def _clip(
    shot: ShotCommand,
    item: Item,
    cover: str | None,
    out: pathlib.Path,
    workdir: pathlib.Path,
    width: int,
    height: int,
) -> None:
    seconds = float(shot["seconds"])
    start = float(shot["sourceStartSeconds"])
    live = float(shot.get("liveLevel") or 0)
    source = workdir / f"source-{uuid.uuid4().hex}"
    key = (item.preview_key or item.key) if shot.get("kind") == "photo" else item.key
    storage.download(key, source)
    ffmpeg.refuse_playlists(source)

    if shot.get("kind") == "photo":
        picture = source
        inputs = ["-loop", "1", "-framerate", str(FPS), "-t", _f(seconds), "-i", str(source)]
        inputs += ["-f", "lavfi", "-t", _f(seconds), "-i", "anullsrc=r=44100:cl=stereo"]
        audio_map, audio_filter = "1:a:0", None
    elif shot.get("kind") == "video":
        picture = source
        inputs = ["-ss", _f(start), "-t", _f(seconds), "-i", str(source)]
        if live > 0 and ffmpeg.has_audio(source):
            audio_map, audio_filter = "0:a:0", f"volume={_f(live)},apad,{_declick(seconds)}"
        else:
            inputs += ["-f", "lavfi", "-t", _f(seconds), "-i", "anullsrc=r=44100:cl=stereo"]
            audio_map, audio_filter = "1:a:0", None
    else:
        picture = item.cover_key or cover
        if not picture:
            raise RuntimeError(f"audio item {item.item_id} has no cover")
        image = workdir / f"cover-{uuid.uuid4().hex}"
        storage.download(picture, image)
        picture = image
        inputs = ["-loop", "1", "-framerate", str(FPS), "-t", _f(seconds), "-i", str(image)]
        inputs += ["-ss", _f(start), "-t", _f(seconds), "-i", str(source)]
        audio_map, audio_filter = "1:a:0", f"volume={_f(live)},apad,{_declick(seconds)}"

    fit_width, fit_height = _fit_size(shot, picture, width, height)
    args = [
        *inputs,
        "-vf",
        ",".join(_video_filters(shot, width, height, fit_width, fit_height)),
        "-map",
        "0:v:0",
        "-map",
        audio_map,
    ]
    if audio_filter:
        args += ["-af", audio_filter]
    args += [
        "-t",
        _f(seconds),
        "-r",
        str(FPS),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        CRF,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        AUDIO_BITRATE,
        "-ar",
        AUDIO_RATE,
        "-ac",
        "2",
        str(out),
    ]
    ffmpeg.run(*args)


def _video_duration(path: pathlib.Path) -> float | None:
    for stream in ffmpeg.probe(path).get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        value = stream.get("duration")
        return float(value) if value else None
    return None


def _join(clips: list[pathlib.Path], transitions: list[float], out: pathlib.Path) -> None:
    if len(clips) == 1:
        ffmpeg.run("-i", str(clips[0]), "-c", "copy", "-movflags", "+faststart", str(out))
        return
    args: list[str] = []
    for clip in clips:
        args += ["-i", str(clip)]
    filters = [f"[{index}:v]settb=1/{FPS}[n{index}]" for index in range(len(clips))]
    last_video, last_audio = "n0", "0:a"
    durations = [_video_duration(clip) or ffmpeg.duration(clip) or 0.0 for clip in clips]
    total = durations[0]
    for index in range(1, len(clips)):
        duration = transitions[index - 1]
        video, audio = f"v{index}", f"a{index}"
        if duration <= 0:
            filters.append(f"[{last_video}][n{index}]concat=n=2:v=1:a=0[cv{index}]")
            filters.append(f"[cv{index}]settb=1/{FPS}[{video}]")
            filters.append(f"[{last_audio}][{index}:a]concat=n=2:v=0:a=1[{audio}]")
            total += durations[index]
        else:
            offset = total - duration
            filters.append(
                f"[{last_video}][n{index}]xfade=transition=fade"
                f":duration={_f(duration)}:offset={_f(offset)}[cv{index}]"
            )
            filters.append(f"[cv{index}]settb=1/{FPS}[{video}]")
            filters.append(f"[{last_audio}][{index}:a]acrossfade=d={_f(duration)}[{audio}]")
            total = total + durations[index] - duration
        last_video, last_audio = video, audio
    args += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        f"[{last_video}]",
        "-map",
        f"[{last_audio}]",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        CRF,
        "-pix_fmt",
        "yuv420p",
        "-r",
        str(FPS),
        "-c:a",
        "aac",
        "-b:a",
        AUDIO_BITRATE,
        "-ar",
        AUDIO_RATE,
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        str(out),
    ]
    ffmpeg.run(*args)


def _join_tree(
    clips: list[pathlib.Path], transitions: list[float], out: pathlib.Path, level: int = 0
) -> None:
    if len(clips) <= CONCAT_GROUP:
        _join(clips, transitions, out)
        return
    groups = [clips[index : index + CONCAT_GROUP] for index in range(0, len(clips), CONCAT_GROUP)]
    outputs: list[pathlib.Path] = []
    boundaries: list[float] = []
    for index, group in enumerate(groups):
        start = index * CONCAT_GROUP
        end = start + len(group)
        group_out = out.parent / f"group-{level}-{index}-{uuid.uuid4().hex[:8]}.mp4"
        _join(group, transitions[start : end - 1], group_out)
        outputs.append(group_out)
        if index > 0:
            boundaries.append(transitions[start - 1])
    _join_tree(outputs, boundaries, out, level + 1)


def _concat(clips: list[pathlib.Path], shots: list[ShotCommand], out: pathlib.Path) -> None:
    transitions = [overlap(shots[index]) for index in range(len(shots) - 1)]
    _join_tree(clips, transitions, out)


def _music(
    plan: EditPlan, songs: dict[str, Song], total: float, out: pathlib.Path, workdir: pathlib.Path
) -> pathlib.Path | None:
    cues = [command for command in plan if command.get("type") == "music"]
    if not cues:
        return None
    inputs: list[str] = []
    chains: list[str] = []
    labels: list[str] = []
    for index, cue in enumerate(cues):
        song = songs[str(cue["mediaId"])]
        path = workdir / f"song-{uuid.uuid4().hex}"
        storage.download(song.key, path)
        ffmpeg.refuse_playlists(path)
        seconds = float(cue["seconds"])
        at = float(cue.get("atSeconds") or 0)
        inputs += ["-ss", _f(cue["sourceStartSeconds"]), "-t", _f(seconds), "-i", str(path)]
        filters = [f"volume={_f(_gain(cue))}"]
        fade_in = float(cue.get("fadeInSeconds") or 0) or EDGE_FADE
        fade_out = float(cue.get("fadeOutSeconds") or 0) or EDGE_FADE
        if at + seconds >= total - 0.05:
            fade_out = max(fade_out, min(MUSIC_TAIL, seconds))
        fade_in = min(fade_in, seconds)
        fade_out = min(fade_out, seconds)
        if fade_in > 0:
            filters.append(f"afade=t=in:st=0:d={_f(fade_in)}")
        if fade_out > 0:
            filters.append(f"afade=t=out:st={_f(max(seconds - fade_out, 0))}:d={_f(fade_out)}")
        filters.append(f"adelay={int(at * 1000)}:all=1")
        label = f"m{index}"
        chains.append(f"[{index}:a]" + ",".join(filters) + f"[{label}]")
        labels.append(f"[{label}]")
    if len(labels) == 1:
        mixed = f"{labels[0]}alimiter=limit=0.95,apad[a]"
    else:
        mixed = (
            "".join(labels)
            + f"amix=inputs={len(labels)}:normalize=0:duration=longest,"
            + "alimiter=limit=0.95,apad[a]"
        )
    ffmpeg.run(
        *inputs,
        "-filter_complex",
        ";".join(chains + [mixed]),
        "-map",
        "[a]",
        "-t",
        _f(total),
        "-c:a",
        "aac",
        "-b:a",
        AUDIO_BITRATE,
        "-ar",
        AUDIO_RATE,
        "-ac",
        "2",
        str(out),
    )
    return out


def _loudness(path: pathlib.Path) -> str:
    result = ffmpeg.run(
        "-hide_banner", "-i", str(path), "-af", f"{LOUDNESS}:print_format=json", "-f", "null", "-"
    )
    text = result.stderr
    try:
        measured = json.loads(text[text.rfind("{") : text.rfind("}") + 1])
        values = [
            float(measured[key])
            for key in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
        ]
    except (ValueError, KeyError):
        log.warning("cannot measure the loudness of %s, the mix stays as it is", path)
        return LOUDNESS
    if not all(math.isfinite(value) for value in values):
        log.info("%s is too quiet to measure, the mix stays as it is", path)
        return LOUDNESS
    return (
        f"{LOUDNESS}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
        f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}:linear=true"
    )


def _final(base: pathlib.Path, music: pathlib.Path | None, out: pathlib.Path) -> None:
    if music is None:
        ffmpeg.run(
            "-i",
            str(base),
            "-af",
            _loudness(base),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            AUDIO_BITRATE,
            "-ar",
            AUDIO_RATE,
            "-ac",
            "2",
            "-movflags",
            "+faststart",
            str(out),
        )
        return
    mixed = out.parent / "mix.m4a"
    ffmpeg.run(
        "-i",
        str(base),
        "-i",
        str(music),
        "-filter_complex",
        (
            "[0:a]asplit=2[live][voice];"
            f"[1:a][voice]sidechaincompress=threshold={DUCK_THRESHOLD}:ratio={DUCK_RATIO}"
            f":attack={DUCK_ATTACK}:release={DUCK_RELEASE}[ducked];"
            "[live][ducked]amix=inputs=2:normalize=0:duration=first,alimiter=limit=0.95[a]"
        ),
        "-map",
        "[a]",
        "-c:a",
        "aac",
        "-b:a",
        AUDIO_BITRATE,
        "-ar",
        AUDIO_RATE,
        "-ac",
        "2",
        str(mixed),
    )
    ffmpeg.run(
        "-i",
        str(base),
        "-i",
        str(mixed),
        "-filter_complex",
        f"[1:a]{_loudness(mixed)}[a]",
        "-map",
        "0:v",
        "-map",
        "[a]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        AUDIO_BITRATE,
        "-ar",
        AUDIO_RATE,
        "-ac",
        "2",
        "-movflags",
        "+faststart",
        str(out),
    )


def _poster(final: pathlib.Path, total: float, out: pathlib.Path) -> None:
    at = min(1.0, max(total / 3, 0.0))
    ffmpeg.run(
        "-ss",
        _f(at),
        "-i",
        str(final),
        "-frames:v",
        "1",
        "-vf",
        "scale='min(640,iw)':-2",
        "-q:v",
        "3",
        str(out),
    )


def _preview(final: pathlib.Path, out: pathlib.Path) -> None:
    ffmpeg.run(
        "-i",
        str(final),
        "-vf",
        "scale=-2:'min(720,ih)'",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "26",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        str(out),
    )


def _meta(width: int | None, height: int | None, duration: float | None) -> dict:
    return {
        "dimensions": {"width": width, "height": height} if width and height else None,
        "duration": duration,
        "at": None,
        "place": None,
        "camera": None,
    }


def _store(
    conn: psycopg.Connection,
    montage: Montage,
    final: pathlib.Path,
    poster: pathlib.Path,
    preview: pathlib.Path,
    total: float,
    width: int,
    height: int,
) -> None:
    movie_id = uuid.uuid5(uuid.NAMESPACE_URL, f"remember:montage:{montage.id}:movie")
    original_id = uuid.uuid5(uuid.NAMESPACE_URL, f"remember:montage:{montage.id}:video")
    compressed_id = uuid.uuid5(uuid.NAMESPACE_URL, f"remember:montage:{montage.id}:compressed")
    preview_id = uuid.uuid5(uuid.NAMESPACE_URL, f"remember:montage:{montage.id}:poster")
    poster_width, poster_height = ffmpeg.dimensions(poster) or (None, None)
    preview_width, preview_height = ffmpeg.dimensions(preview) or (None, None)
    preview_length = ffmpeg.duration(preview) or total
    files = [
        (original_id, "video", "mp4", final, width, height, total),
        (compressed_id, "video", "mp4", preview, preview_width, preview_height, preview_length),
        (preview_id, "photo", "jpeg", poster, poster_width, poster_height, None),
    ]
    keys = [storage.media_key(str(media_id), fmt) for media_id, _, fmt, _, _, _, _ in files]
    try:
        for media_id, _, fmt, path, _, _, _ in files:
            key = storage.media_key(str(media_id), fmt)
            storage.upload(path, key, f"{'image' if fmt == 'jpeg' else 'video'}/{fmt}")
        with conn.transaction():
            for media_id, kind, fmt, path, w, h, duration in files:
                conn.execute(
                    """
                    insert into media_items (id, kind, storage_key, format, bytes, meta)
                    values (%s, %s, %s, %s, %s, %s)
                    on conflict (id) do update set
                        kind = excluded.kind,
                        storage_key = excluded.storage_key,
                        format = excluded.format,
                        bytes = excluded.bytes,
                        meta = excluded.meta
                    """,
                    (
                        media_id,
                        kind,
                        storage.media_key(str(media_id), fmt),
                        fmt,
                        path.stat().st_size,
                        Jsonb(_meta(w, h, duration)),
                    ),
                )
            conn.execute("delete from movies where montage_id = %s", (montage.id,))
            conn.execute(
                """
                insert into movies (
                    id, name, project_id, created_by, montage_id,
                    original_id, compressed_id, preview_id
                )
                values (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    movie_id,
                    montage.title,
                    montage.project_id,
                    montage.created_by,
                    montage.id,
                    original_id,
                    compressed_id,
                    preview_id,
                ),
            )
            landed = conn.execute(
                "update montages set status = 'done', error = null where id = %s", (montage.id,)
            ).rowcount
            if not landed:
                raise RuntimeError(f"montage {montage.id} went away during assembly")
    except Exception:
        for key in keys:
            storage.delete(key)
        raise


def validate(conn: psycopg.Connection, montage_id: uuid.UUID, plan: EditPlan) -> None:
    montage = _load(conn, montage_id)
    shots = [command for command in plan if command.get("type") == "shot"]
    items = _items(conn, [str(shot["itemId"]) for shot in shots])
    songs = _songs(conn, [str(cue["mediaId"]) for cue in plan if cue.get("type") == "music"])
    _validate(plan, montage, items, songs)


def render(conn: psycopg.Connection, montage_id: uuid.UUID, plan: EditPlan) -> None:
    montage = _load(conn, montage_id)
    shots = [command for command in plan if command.get("type") == "shot"]
    items = _items(conn, [str(shot["itemId"]) for shot in shots])
    songs = _songs(conn, [str(cue["mediaId"]) for cue in plan if cue.get("type") == "music"])
    _validate(plan, montage, items, songs)

    dimensions = montage.settings.get("dimensions") or {}
    width, height = int(dimensions.get("width") or 1080), int(dimensions.get("height") or 1920)
    cover = _cover(conn)

    with tempfile.TemporaryDirectory() as directory:
        workdir = pathlib.Path(directory)
        clips = []
        for index, shot in enumerate(shots):
            clip = workdir / f"clip-{index}.mp4"
            _clip(shot, items[str(shot["itemId"])], cover, clip, workdir, width, height)
            clips.append(clip)
        base = workdir / "base.mp4"
        _concat(clips, shots, base)
        total = ffmpeg.duration(base) or timeline_length(shots)
        music = _music(plan, songs, total, workdir / "music.m4a", workdir)
        final = workdir / "final.mp4"
        _final(base, music, final)
        total = ffmpeg.duration(final) or total
        poster = workdir / "poster.jpg"
        _poster(final, total, poster)
        preview = workdir / "preview.mp4"
        _preview(final, preview)
        _store(conn, montage, final, poster, preview, total, width, height)

    log.info("montage %s is ready: %.1f seconds", montage_id, total)
    notify.ready(montage_id)


if __name__ == "__main__":
    import shutil
    import wave

    import numpy as np

    with tempfile.TemporaryDirectory() as directory:
        work = pathlib.Path(directory)
        tone = work / "song.wav"
        ffmpeg.run("-f", "lavfi", "-i", "sine=f=300:d=4", "-ar", AUDIO_RATE, "-ac", "2", str(tone))

        def download(_key, path):
            shutil.copy(tone, path)

        original = storage.download
        storage.download = download
        try:
            plan: EditPlan = [
                {
                    "type": "music",
                    "songId": "s",
                    "mediaId": "s",
                    "sourceStartSeconds": 0.0,
                    "seconds": 4.0,
                    "atSeconds": 0.0,
                    "gain": 1.0,
                    "fadeInSeconds": 0.0,
                    "fadeOutSeconds": 0.0,
                }
            ]
            music = _music(plan, {"s": Song("s", "key", 4.0)}, 4.0, work / "music.m4a", work)
            assert music is not None
            base = work / "base.mp4"
            ffmpeg.run(
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x64:d=4:r=30",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=44100:cl=stereo",
                "-t",
                "4",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(base),
            )
            final = work / "final.mp4"
            _final(base, music, final)
            assert abs((ffmpeg.duration(final) or 0) - 4.0) < 0.2

            def rms(from_second: float) -> float:
                wav = work / f"rms-{from_second}.wav"
                ffmpeg.run(
                    "-ss",
                    f"{from_second}",
                    "-t",
                    "0.2",
                    "-i",
                    str(final),
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    str(wav),
                )
                with wave.open(str(wav)) as file:
                    data = np.frombuffer(file.readframes(file.getnframes()), dtype=np.int16)
                return float(np.sqrt(((data / 32768) ** 2).mean()))

            assert rms(3.8) < rms(2.0) * 0.5, (rms(3.8), rms(2.0))
        finally:
            storage.download = original
    print("ok")
