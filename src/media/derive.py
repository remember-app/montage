import dataclasses
import logging
import pathlib
import tempfile
import uuid
from datetime import datetime

import psycopg
from psycopg.types.json import Jsonb

from src import config
from src.integrations import ffmpeg, storage
from src.media import fingerprint, perceptual, probe

log = logging.getLogger("media.derive")

PREVIEW_SIDES = {"small": 160, "middle": 480, "large": 960}
COMPRESSED_SIDE = 2048
COMPRESSED_HEIGHT = 720
SDR_TONEMAP = (
    "zscale=t=linear:npl=100,format=gbrpf32le,"
    "zscale=p=bt709,tonemap=hable:desat=0,"
    "zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
)


@dataclasses.dataclass
class Derivative:
    id: uuid.UUID
    role: str
    kind: str
    format: str
    path: pathlib.Path
    width: int | None
    height: int | None
    duration: float | None


def _derivative_id(media_id: uuid.UUID, role: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"remember:{media_id}:{role}")


def _time(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _meta(width: int | None, height: int | None, duration: float | None) -> dict:
    return {
        "dimensions": {"width": width, "height": height} if width and height else None,
        "duration": duration,
        "at": None,
        "place": None,
        "camera": None,
    }


def _content_type(file_format: str) -> str:
    return {
        "jpeg": "image/jpeg",
        "mp4": "video/mp4",
        "m4a": "audio/mp4",
    }.get(file_format, "application/octet-stream")


def _video_frame(original: pathlib.Path, out: pathlib.Path) -> None:
    ffmpeg.run("-i", str(original), "-frames:v", "1", "-vf", SDR_TONEMAP, str(out))


def _video_is_hdr(path: pathlib.Path) -> bool:
    for stream in ffmpeg.probe(path).get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        transfer = str(stream.get("color_transfer") or "")
        pixel = str(stream.get("pix_fmt") or "")
        return transfer in {"smpte2084", "arib-std-b67"} or pixel.endswith(
            ("10le", "12le", "14le", "16le")
        )
    return False


def _video_compressed(original: pathlib.Path, out: pathlib.Path) -> None:
    filters = f"scale=-2:'min({COMPRESSED_HEIGHT},ih)'"
    filters += f",{SDR_TONEMAP}" if _video_is_hdr(original) else ",format=yuv420p"
    ffmpeg.run(
        "-i",
        str(original),
        "-vf",
        filters,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(out),
    )


def _audio_compressed(original: pathlib.Path, out: pathlib.Path) -> None:
    ffmpeg.run("-i", str(original), "-vn", "-c:a", "aac", "-b:a", "96k", str(out))


def _cover_id(conn: psycopg.Connection) -> uuid.UUID | None:
    raw = config.audio_cover_media_id()
    if not raw:
        return None
    try:
        value = uuid.UUID(raw)
    except ValueError:
        log.warning("AUDIO_COVER_MEDIA_ID is not a uuid: %r", raw)
        return None
    if conn.execute("select 1 from media_items where id = %s", (value,)).fetchone() is None:
        log.warning("AUDIO_COVER_MEDIA_ID %s is not a media item", value)
        return None
    return value


def _files(
    original: pathlib.Path,
    work: pathlib.Path,
    media_id: uuid.UUID,
    kind: str,
) -> list[Derivative]:
    result: list[Derivative] = []
    if kind == "photo":
        compressed = work / "compressed.jpg"
        size = probe.render_photo(original, compressed, COMPRESSED_SIDE)
        result.append(
            Derivative(
                _derivative_id(media_id, "compressed"),
                "compressed",
                "photo",
                "jpeg",
                compressed,
                size["width"],
                size["height"],
                None,
            )
        )
        for role, side in PREVIEW_SIDES.items():
            path = work / f"preview-{role}.jpg"
            size = probe.render_photo(original, path, side)
            result.append(
                Derivative(
                    _derivative_id(media_id, role),
                    role,
                    "photo",
                    "jpeg",
                    path,
                    size["width"],
                    size["height"],
                    None,
                )
            )
        return result
    if kind == "video":
        compressed = work / "compressed.mp4"
        _video_compressed(original, compressed)
        cw, ch = ffmpeg.dimensions(compressed) or (None, None)
        result.append(
            Derivative(
                _derivative_id(media_id, "compressed"),
                "compressed",
                "video",
                "mp4",
                compressed,
                cw,
                ch,
                ffmpeg.duration(compressed),
            )
        )
        frame = work / "frame.jpg"
        _video_frame(compressed, frame)
        for role, side in PREVIEW_SIDES.items():
            path = work / f"preview-{role}.jpg"
            size = probe.render_photo(frame, path, side)
            result.append(
                Derivative(
                    _derivative_id(media_id, f"{role}:sdr"),
                    role,
                    "photo",
                    "jpeg",
                    path,
                    size["width"],
                    size["height"],
                    None,
                )
            )
        return result
    if kind == "audio":
        compressed = work / "compressed.m4a"
        _audio_compressed(original, compressed)
        result.append(
            Derivative(
                _derivative_id(media_id, "compressed"),
                "compressed",
                "audio",
                "m4a",
                compressed,
                None,
                None,
                ffmpeg.duration(compressed),
            )
        )
    return result


def _camera(info: dict) -> dict | None:
    keys = ("camera_model", "focal_length_mm", "aperture", "iso", "exposure_ev", "shutter")
    if not any(info.get(key) is not None for key in keys):
        return None
    return {
        "model": info.get("camera_model"),
        "focalLengthMm": info.get("focal_length_mm"),
        "exposure": {
            "aperture": info.get("aperture"),
            "iso": info.get("iso"),
            "ev": info.get("exposure_ev"),
            "shutter": info.get("shutter"),
        },
    }


def _original_meta(info: dict, kind: str, current: dict, fallback_at: str | None) -> dict:
    meta = dict(current)
    captured = _time(fallback_at) or _time(info.get("captured_at"))
    updates = {
        "dimensions": (
            {"width": info.get("width"), "height": info.get("height")}
            if info.get("width") is not None and info.get("height") is not None
            else None
        ),
        "duration": info.get("duration") if kind != "photo" else None,
        "at": captured.isoformat() if captured is not None else None,
        "camera": _camera(info),
    }
    for key, value in updates.items():
        if meta.get(key) is None and value is not None:
            meta[key] = value
    return meta


def run(conn: psycopg.Connection, ref: uuid.UUID, level: str | None = None) -> None:
    upload = conn.execute(
        "select media_id, file->>'capturedAt' from uploads where id = %s", (ref,)
    ).fetchone()
    if upload is None or upload[0] is None:
        log.info("upload %s has no media, nothing to derive", ref)
        return
    media_id, device_at = upload
    row = conn.execute(
        "select kind, storage_key, meta, fingerprint from media_items where id = %s", (media_id,)
    ).fetchone()
    if row is None:
        log.info("media item %s is gone, nothing to derive", media_id)
        return
    kind, storage_key, current_meta, stored_fingerprint = row
    owner = conn.execute(
        """
        select 'item' as kind, id from project_items where original_id = %s
        union all
        select 'song' as kind, id from songs where original_id = %s
        limit 1
        """,
        (media_id, media_id),
    ).fetchone()
    if owner is None:
        log.info("media item %s has no owner, nothing to derive", media_id)
        return
    owner_kind, owner_id = owner
    old_previews: list[str] = []
    if kind == "video" and owner_kind == "item":
        stored = conn.execute(
            "select previews_ids from project_items where id = %s", (owner_id,)
        ).fetchone()
        if stored and stored[0]:
            old_previews = list(stored[0].values())

    with tempfile.TemporaryDirectory() as directory:
        work = pathlib.Path(directory)
        original = work / "original"
        storage.download(storage_key, original)
        ffmpeg.refuse_playlists(original)
        info = probe.read(original, kind)
        derivatives = _files(original, work, media_id, kind)
        middle = work / "preview-middle.jpg"
        perceptual_hash = perceptual.hash_of(middle) if middle.exists() else None
        content_hash = stored_fingerprint or fingerprint.sha256_of(original)
        meta = _original_meta(info, kind, current_meta or {}, device_at)

        for derivative in derivatives:
            key = storage.media_key(str(derivative.id), derivative.format)
            storage.upload(derivative.path, key, _content_type(derivative.format))

        with conn.transaction():
            conn.execute(
                "update media_items set meta = %s, "
                "perceptual_hash = coalesce(%s, perceptual_hash), "
                "fingerprint = %s where id = %s",
                (Jsonb(meta), perceptual_hash, content_hash, media_id),
            )
            for derivative in derivatives:
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
                        derivative.id,
                        derivative.kind,
                        storage.media_key(str(derivative.id), derivative.format),
                        derivative.format,
                        derivative.path.stat().st_size,
                        Jsonb(
                            _meta(derivative.width, derivative.height, derivative.duration)
                        ),
                    ),
                )
            compressed = next((item for item in derivatives if item.role == "compressed"), None)
            previews = {
                item.role: str(item.id) for item in derivatives if item.role in PREVIEW_SIDES
            }
            if kind == "audio":
                cover = _cover_id(conn)
                if cover is not None:
                    previews = {role: str(cover) for role in PREVIEW_SIDES}
            if owner_kind == "item":
                conn.execute(
                    """
                    update project_items
                    set compressed_id = %s, previews_ids = %s
                    where id = %s
                    """,
                    (compressed.id if compressed else None, Jsonb(previews), owner_id),
                )
            else:
                conn.execute(
                    "update songs set compressed_id = %s where id = %s",
                    (compressed.id if compressed else None, owner_id),
                )
            if kind == "video":
                stale = [uuid.UUID(one) for one in set(old_previews) - set(previews.values())]
                if stale:
                    removed = conn.execute(
                        "delete from media_items where id = any(%s) returning storage_key",
                        (stale,),
                    ).fetchall()
                    for (key,) in removed:
                        storage.delete(key)

    log.info("media item %s (%s) derived: %s files", media_id, kind, len(derivatives))
