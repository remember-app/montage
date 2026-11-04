import dataclasses
import logging
import pathlib
import tempfile
import uuid
from datetime import datetime

import psycopg

from src import config
from src.integrations import ffmpeg, storage
from src.media import probe

log = logging.getLogger("media.derive")

THUMBNAIL_SIDE = 640
PREVIEW_LONG_SIDE = 2048
PREVIEW_HEIGHT = 720


@dataclasses.dataclass
class Derivative:
    id: uuid.UUID
    kind: str
    format: str
    path: pathlib.Path
    width: int | None
    height: int | None
    duration: float | None


def _derivative_id(media_id: uuid.UUID, part: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"remember:{media_id}:{part}")


def _time(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _video_thumbnail(original: pathlib.Path, out: pathlib.Path, duration: float | None) -> None:
    at = min(1.0, (duration or 2.0) / 2)
    ffmpeg.run(
        "-ss",
        f"{at:.3f}",
        "-i",
        str(original),
        "-frames:v",
        "1",
        "-vf",
        f"scale='min({THUMBNAIL_SIDE},iw)':-2",
        str(out),
    )


def _video_preview(original: pathlib.Path, out: pathlib.Path) -> None:
    ffmpeg.run(
        "-i",
        str(original),
        "-vf",
        f"scale=-2:'min({PREVIEW_HEIGHT},ih)'",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(out),
    )


def _cover_id(conn: psycopg.Connection) -> uuid.UUID | None:
    raw = config.audio_cover_media_id()
    if not raw:
        return None
    try:
        value = uuid.UUID(raw)
    except ValueError:
        log.warning("AUDIO_COVER_MEDIA_ID is not a uuid: %r", raw)
        return None
    found = conn.execute("select 1 from media_items where id = %s", (value,)).fetchone()
    if found is None:
        log.warning("AUDIO_COVER_MEDIA_ID %s is not a media item", value)
        return None
    return value


def _files(
    original: pathlib.Path,
    work: pathlib.Path,
    media_id: uuid.UUID,
    kind: str,
    duration: float | None,
) -> list[Derivative]:
    if kind == "photo":
        thumbnail, preview = work / "thumbnail.jpg", work / "preview.jpg"
        small = probe.render_photo(original, thumbnail, THUMBNAIL_SIDE)
        large = probe.render_photo(original, preview, PREVIEW_LONG_SIDE)
        return [
            Derivative(
                _derivative_id(media_id, "thumbnail"),
                "photo",
                "jpeg",
                thumbnail,
                small["width"],
                small["height"],
                None,
            ),
            Derivative(
                _derivative_id(media_id, "preview"),
                "photo",
                "jpeg",
                preview,
                large["width"],
                large["height"],
                None,
            ),
        ]
    if kind == "video":
        thumbnail, preview = work / "thumbnail.jpg", work / "preview.mp4"
        _video_thumbnail(original, thumbnail, duration)
        _video_preview(original, preview)
        small = ffmpeg.dimensions(thumbnail) or (None, None)
        large = ffmpeg.dimensions(preview) or (None, None)
        return [
            Derivative(
                _derivative_id(media_id, "thumbnail"),
                "photo",
                "jpeg",
                thumbnail,
                small[0],
                small[1],
                None,
            ),
            Derivative(
                _derivative_id(media_id, "preview"),
                "video",
                "mp4",
                preview,
                large[0],
                large[1],
                ffmpeg.duration(preview),
            ),
        ]
    return []


def run(conn: psycopg.Connection, ref: uuid.UUID) -> None:
    upload = conn.execute("select media_id from uploads where id = %s", (ref,)).fetchone()
    if upload is None or upload[0] is None:
        log.info("upload %s has no media, nothing to derive", ref)
        return
    media_id = upload[0]
    row = conn.execute(
        "select kind, storage_key, uploaded_by from media_items where id = %s", (media_id,)
    ).fetchone()
    if row is None:
        log.info("media item %s is gone, nothing to derive", media_id)
        return
    kind, storage_key, uploaded_by = row

    with tempfile.TemporaryDirectory() as directory:
        work = pathlib.Path(directory)
        original = work / "original"
        storage.download(storage_key, original)
        ffmpeg.refuse_playlists(original)
        info = probe.read(original, kind)
        derivatives = _files(original, work, media_id, kind, info.get("duration"))
        cover = _cover_id(conn) if kind == "audio" else None

        for derivative in derivatives:
            key = storage.media_key(str(derivative.id), derivative.format)
            content_type = "video/mp4" if derivative.format == "mp4" else "image/jpeg"
            storage.upload(derivative.path, key, content_type)

        with conn.transaction():
            landed = conn.execute(
                """
                update media_items
                set width = coalesce(%s, width),
                    height = coalesce(%s, height),
                    duration = coalesce(%s, duration),
                    captured_at = coalesce(captured_at, %s),
                    camera_model = coalesce(camera_model, %s),
                    focal_length_mm = coalesce(focal_length_mm, %s),
                    aperture = coalesce(aperture, %s),
                    iso = coalesce(iso, %s),
                    exposure_ev = coalesce(exposure_ev, %s),
                    shutter = coalesce(shutter, %s),
                    latitude = coalesce(latitude, %s),
                    longitude = coalesce(longitude, %s)
                where id = %s
                returning id
                """,
                (
                    info.get("width"),
                    info.get("height"),
                    info.get("duration"),
                    _time(info.get("captured_at")),
                    info.get("camera_model"),
                    info.get("focal_length_mm"),
                    info.get("aperture"),
                    info.get("iso"),
                    info.get("exposure_ev"),
                    info.get("shutter"),
                    info.get("latitude"),
                    info.get("longitude"),
                    media_id,
                ),
            ).fetchone()
            if landed is None:
                for derivative in derivatives:
                    storage.delete(storage.media_key(str(derivative.id), derivative.format))
                log.info("media item %s went away mid-derive", media_id)
                return

            for derivative in derivatives:
                conn.execute(
                    """
                    insert into media_items (
                        id, parent_media_id, kind, storage_key, uploaded_by, format, bytes,
                        width, height, duration
                    )
                    values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    on conflict (id) do update
                    set parent_media_id = excluded.parent_media_id,
                        kind = excluded.kind,
                        storage_key = excluded.storage_key,
                        uploaded_by = excluded.uploaded_by,
                        format = excluded.format,
                        bytes = excluded.bytes,
                        width = excluded.width,
                        height = excluded.height,
                        duration = excluded.duration
                    """,
                    (
                        derivative.id,
                        media_id,
                        derivative.kind,
                        storage.media_key(str(derivative.id), derivative.format),
                        uploaded_by,
                        derivative.format,
                        derivative.path.stat().st_size,
                        derivative.width,
                        derivative.height,
                        derivative.duration,
                    ),
                )

            thumbnail = cover if kind == "audio" else _derivative_id(media_id, "thumbnail")
            preview = None if kind == "audio" else _derivative_id(media_id, "preview")
            conn.execute(
                """
                update project_items
                set thumbnail_id = %s, preview_id = %s
                where media_id = %s
                """,
                (thumbnail, preview, media_id),
            )

    log.info("media item %s (%s) derived: %s files", media_id, kind, len(derivatives))
