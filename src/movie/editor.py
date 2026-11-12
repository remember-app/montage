import dataclasses
import logging
import pathlib
import tempfile
import uuid

import psycopg

from src import config
from src.integrations import ffmpeg, storage
from src.movie import notify
from src.movie.types import EditPlan, ShotCommand

log = logging.getLogger("movie.editor")

FPS = 30
CRF = "20"
AUDIO_RATE = "44100"
AUDIO_BITRATE = "192k"
MIN_TRANSITION = 0.01


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


def _transition(shot: ShotCommand) -> float:
    return max(float(shot.get("transitionSeconds") or 0), MIN_TRANSITION)


def _total(shots: list[ShotCommand]) -> float:
    total = 0.0
    for index, shot in enumerate(shots):
        total += float(shot["seconds"])
        if index < len(shots) - 1:
            total -= _transition(shot)
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
        select pi.id, pi.media_id, pi.kind, m.storage_key, m.duration,
               (select pm.storage_key from media_items pm where pm.id = pi.thumbnail_id),
               (select pm.storage_key from media_items pm where pm.id = pi.preview_id)
        from project_items pi join media_items m on m.id = pi.media_id
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
        select s.media_id, m.storage_key, m.duration
        from songs s join media_items m on m.id = s.media_id
        where s.media_id = any(%s)
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
    for index, shot in enumerate(shots[:-1]):
        following = float(shots[index + 1]["seconds"])
        if _transition(shot) >= min(float(shot["seconds"]), following):
            issues.append(f"item {shot.get('itemId')} overlaps the next shot entirely")
    required = set(montage.settings.get("requiredItemIds") or [])
    used = {str(shot.get("itemId")) for shot in shots}
    for item_id in required - used:
        issues.append(f"required item {item_id} is missing")
    total = _total(shots)
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
        gain = float(cue.get("gain") or 0)
        if not 0 <= gain <= 2:
            issues.append(f"song {song.media_id} has gain outside 0..2")
        for side in ("fadeInSeconds", "fadeOutSeconds"):
            fade = float(cue.get(side) or 0)
            if fade < 0 or fade > seconds:
                issues.append(f"song {song.media_id} has {side} outside its cue")
    if issues:
        raise RuntimeError("; ".join(issues))


def _video_filters(shot: ShotCommand, width: int, height: int) -> list[str]:
    filters = []
    framing = shot.get("framing")
    if framing:
        filters.append(
            "crop=iw*{w}:ih*{h}:iw*{x}:ih*{y}".format(
                x=_f(framing["x"]), y=_f(framing["y"]), w=_f(framing["w"]), h=_f(framing["h"])
            )
        )
    filters.append(f"scale={width}:{height}:force_original_aspect_ratio=increase")
    filters.append(f"crop={width}:{height}")
    filters.append("setsar=1")
    motion = shot.get("motion") or "none"
    if motion != "none" and shot.get("kind") in ("photo", "audio"):
        zoom = "min(1.0+0.0008*on,1.15)" if motion == "zoom_in" else "max(1.15-0.0008*on,1.0)"
        filters.append(
            f"zoompan=z='{zoom}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d=1:s={width}x{height}:fps={FPS}"
        )
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
        inputs = ["-loop", "1", "-framerate", str(FPS), "-t", _f(seconds), "-i", str(source)]
        inputs += ["-f", "lavfi", "-t", _f(seconds), "-i", "anullsrc=r=44100:cl=stereo"]
        audio_map, audio_filter = "1:a:0", None
    elif shot.get("kind") == "video":
        inputs = ["-ss", _f(start), "-t", _f(seconds), "-i", str(source)]
        if live > 0 and ffmpeg.has_audio(source):
            audio_map, audio_filter = "0:a:0", f"volume={_f(live)},apad"
        else:
            inputs += ["-f", "lavfi", "-t", _f(seconds), "-i", "anullsrc=r=44100:cl=stereo"]
            audio_map, audio_filter = "1:a:0", None
    else:
        picture = item.cover_key or cover
        if not picture:
            raise RuntimeError(f"audio item {item.item_id} has no cover")
        image = workdir / f"cover-{uuid.uuid4().hex}"
        storage.download(picture, image)
        inputs = ["-loop", "1", "-framerate", str(FPS), "-t", _f(seconds), "-i", str(image)]
        inputs += ["-ss", _f(start), "-t", _f(seconds), "-i", str(source)]
        audio_map, audio_filter = "1:a:0", f"volume={_f(live)},apad"

    args = [
        *inputs,
        "-vf",
        ",".join(_video_filters(shot, width, height)),
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


def _concat(clips: list[pathlib.Path], shots: list[ShotCommand], out: pathlib.Path) -> None:
    if len(clips) == 1:
        ffmpeg.run("-i", str(clips[0]), "-c", "copy", "-movflags", "+faststart", str(out))
        return
    args: list[str] = []
    for clip in clips:
        args += ["-i", str(clip)]
    filters = []
    last_video, last_audio = "0:v", "0:a"
    total = float(shots[0]["seconds"])
    for index in range(1, len(clips)):
        duration = _transition(shots[index - 1])
        offset = total - duration
        video, audio = f"v{index}", f"a{index}"
        filters.append(
            f"[{last_video}][{index}:v]xfade=transition=fade"
            f":duration={_f(duration)}:offset={_f(offset)}[{video}]"
        )
        filters.append(f"[{last_audio}][{index}:a]acrossfade=d={_f(duration)}[{audio}]")
        last_video, last_audio = video, audio
        total = total + float(shots[index]["seconds"]) - duration
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
        inputs += ["-ss", _f(cue["sourceStartSeconds"]), "-t", _f(seconds), "-i", str(path)]
        filters = [f"volume={_f(cue.get('gain', 1))}"]
        fade_in = float(cue.get("fadeInSeconds") or 0)
        fade_out = float(cue.get("fadeOutSeconds") or 0)
        if fade_in > 0:
            filters.append(f"afade=t=in:st=0:d={_f(fade_in)}")
        if fade_out > 0:
            filters.append(f"afade=t=out:st={_f(max(seconds - fade_out, 0))}:d={_f(fade_out)}")
        filters.append(f"adelay={int(float(cue.get('atSeconds') or 0) * 1000)}:all=1")
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


def _final(base: pathlib.Path, music: pathlib.Path | None, out: pathlib.Path) -> None:
    if music is None:
        ffmpeg.run("-i", str(base), "-c", "copy", "-movflags", "+faststart", str(out))
        return
    ffmpeg.run(
        "-i",
        str(base),
        "-i",
        str(music),
        "-filter_complex",
        "[0:a][1:a]amix=inputs=2:normalize=0:duration=first,alimiter=limit=0.95[a]",
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
    video_id = uuid.uuid5(uuid.NAMESPACE_URL, f"remember:montage:{montage.id}:video")
    thumbnail_id = uuid.uuid5(uuid.NAMESPACE_URL, f"remember:montage:{montage.id}:thumbnail")
    preview_id = uuid.uuid5(uuid.NAMESPACE_URL, f"remember:montage:{montage.id}:preview")
    video_key = storage.media_key(str(video_id), "mp4")
    thumbnail_key = storage.media_key(str(thumbnail_id), "jpeg")
    preview_key = storage.media_key(str(preview_id), "mp4")
    keys = [video_key, thumbnail_key, preview_key]
    try:
        storage.upload(final, video_key, "video/mp4")
        storage.upload(poster, thumbnail_key, "image/jpeg")
        storage.upload(preview, preview_key, "video/mp4")
        poster_width, poster_height = ffmpeg.dimensions(poster) or (None, None)
        preview_width, preview_height = ffmpeg.dimensions(preview) or (None, None)
        preview_length = ffmpeg.duration(preview) or total
        with conn.transaction():
            conn.execute(
                """
                insert into media_items
                    (id, kind, storage_key, uploaded_by, format, bytes, width, height, duration)
                values (%s, 'video', %s, %s, 'mp4', %s, %s, %s, %s)
                on conflict (id) do update set
                    storage_key = excluded.storage_key,
                    uploaded_by = excluded.uploaded_by,
                    bytes = excluded.bytes,
                    width = excluded.width,
                    height = excluded.height,
                    duration = excluded.duration
                """,
                (
                    video_id,
                    video_key,
                    montage.created_by,
                    final.stat().st_size,
                    width,
                    height,
                    total,
                ),
            )
            conn.execute(
                """
                insert into media_items
                    (id, kind, storage_key, uploaded_by, format, bytes, width, height, duration)
                values (%s, 'photo', %s, %s, 'jpeg', %s, %s, %s, null)
                on conflict (id) do update set
                    storage_key = excluded.storage_key,
                    uploaded_by = excluded.uploaded_by,
                    bytes = excluded.bytes,
                    width = excluded.width,
                    height = excluded.height
                """,
                (
                    thumbnail_id,
                    thumbnail_key,
                    montage.created_by,
                    poster.stat().st_size,
                    poster_width,
                    poster_height,
                ),
            )
            conn.execute(
                """
                insert into media_items
                    (id, kind, storage_key, uploaded_by, format, bytes, width, height, duration)
                values (%s, 'video', %s, %s, 'mp4', %s, %s, %s, %s)
                on conflict (id) do update set
                    storage_key = excluded.storage_key,
                    uploaded_by = excluded.uploaded_by,
                    bytes = excluded.bytes,
                    width = excluded.width,
                    height = excluded.height,
                    duration = excluded.duration
                """,
                (
                    preview_id,
                    preview_key,
                    montage.created_by,
                    preview.stat().st_size,
                    preview_width,
                    preview_height,
                    preview_length,
                ),
            )
            conn.execute("delete from movies where montage_id = %s", (montage.id,))
            conn.execute(
                """
                insert into movies
                    (name, project_id, created_by, montage_id, video_id, thumbnail_id, preview_id)
                values (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    montage.title,
                    montage.project_id,
                    montage.created_by,
                    montage.id,
                    video_id,
                    thumbnail_id,
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
        total = ffmpeg.duration(base) or _total(shots)
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
