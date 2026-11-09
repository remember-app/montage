import dataclasses
import logging
import pathlib
import tempfile
import uuid

import psycopg
from psycopg.types.json import Jsonb

from src.integrations import ffmpeg, storage
from src.movie.types import MontageReport

log = logging.getLogger("movie.preparation")

THRESHOLD = 0.95


@dataclasses.dataclass
class ShotRow:
    item_id: str
    media_id: str
    kind: str
    description: str | None
    analysis: dict | None
    duration: float | None
    width: int | None
    height: int | None
    captured_at: str | None
    preview_key: str | None
    key: str
    labels: list[str]


@dataclasses.dataclass
class SongRow:
    song_id: str
    media_id: str
    name: str
    author: str
    duration: float | None
    analysis: dict | None


def _settings(conn: psycopg.Connection, montage_id: uuid.UUID) -> dict:
    row = conn.execute("select manifest from montages where id = %s", (montage_id,)).fetchone()
    if row is None:
        raise RuntimeError(f"montage {montage_id} is gone")
    settings = (row[0] or {}).get("preparationConfig") or {}
    if not settings:
        raise RuntimeError(f"montage {montage_id} has no preparation settings")
    return settings


def _has_audio(shot: ShotRow) -> bool:
    if shot.kind == "audio":
        return True
    if shot.kind == "photo":
        return False
    key = shot.preview_key or shot.key
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "clip"
        storage.download(key, path)
        ffmpeg.refuse_playlists(path)
        return ffmpeg.has_audio(path)


def _labels(conn: psycopg.Connection, item_ids: list[uuid.UUID]) -> dict[str, list[str]]:
    labels: dict[str, list[str]] = {}
    if not item_ids:
        return labels
    for item_id, name in conn.execute(
        """
        select pil.project_item_id, l.name
        from project_item_labels pil join labels l on l.id = pil.label_id
        where pil.project_item_id = any(%s)
        order by l.name
        """,
        (item_ids,),
    ):
        labels.setdefault(str(item_id), []).append(name)
    return labels


def _shots(conn: psycopg.Connection, settings: dict) -> list[ShotRow]:
    ids = [uuid.UUID(one) for one in settings.get("includeItemIds") or []]
    if not ids:
        raise RuntimeError("the order has no included items")
    labels = _labels(conn, ids)
    rows = conn.execute(
        """
        select pi.id, pi.media_id, pi.kind, pi.description, pi.analysis,
               m.duration, m.width, m.height, m.captured_at,
               (select pm.storage_key from media_items pm where pm.id = pi.preview_id),
               m.storage_key
        from project_items pi join media_items m on m.id = pi.media_id
        where pi.id = any(%s)
        order by coalesce(m.captured_at, pi.created_at), m.id
        """,
        (ids,),
    ).fetchall()
    found = {str(row[0]) for row in rows}
    for item_id in ids:
        if str(item_id) not in found:
            raise RuntimeError(f"item {item_id} is not in this project")
    return [
        ShotRow(
            str(item_id),
            str(media_id),
            kind,
            description,
            analysis,
            duration,
            width,
            height,
            captured.isoformat() if captured else None,
            preview_key,
            key,
            labels.get(str(item_id), []),
        )
        for (
            item_id,
            media_id,
            kind,
            description,
            analysis,
            duration,
            width,
            height,
            captured,
            preview_key,
            key,
        ) in rows
    ]


def _music(conn: psycopg.Connection, settings: dict) -> list[SongRow]:
    ids = [uuid.UUID(one) for one in settings.get("musicIds") or []]
    if not ids:
        return []
    rows = conn.execute(
        """
        select s.id, s.media_id, s.name, s.author, s.analysis, m.duration
        from songs s join media_items m on m.id = s.media_id
        where s.media_id = any(%s)
        """,
        (ids,),
    ).fetchall()
    by_media = {str(row[1]): row for row in rows}
    tracks = []
    for media_id in ids:
        row = by_media.get(str(media_id))
        if row is None:
            raise RuntimeError(f"song {media_id} is not available to this project")
        tracks.append(SongRow(str(row[0]), str(row[1]), row[2], row[3], row[5], row[4]))
    return tracks


def _coverage(shots: list[ShotRow], music: list[SongRow]) -> None:
    total = len(shots) + len(music)
    if total == 0:
        raise RuntimeError("the order has no files")
    described = sum(1 for shot in shots if shot.analysis) + sum(
        1 for song in music if song.analysis
    )
    if described / total <= THRESHOLD:
        missing = [shot.media_id for shot in shots if not shot.analysis]
        missing += [song.media_id for song in music if not song.analysis]
        raise RuntimeError(
            f"analysis covers {described}/{total} files; more than 95% is required; "
            f"missing: {', '.join(missing)}"
        )


def _report(settings: dict, shots: list[ShotRow], music: list[SongRow]) -> MontageReport:
    dimensions = settings.get("dimensions") or {}
    return {
        "movie": {
            "targetSeconds": int(settings.get("targetSeconds") or 0),
            "dimensions": {
                "width": int(dimensions.get("width") or 0),
                "height": int(dimensions.get("height") or 0),
            },
            "story": settings.get("prompt"),
            "requiredItemIds": [str(one) for one in settings.get("requiredItemIds") or []],
        },
        "shots": [
            {
                "itemId": shot.item_id,
                "mediaId": shot.media_id,
                "kind": shot.kind,
                "capturedAt": shot.captured_at,
                "width": shot.width,
                "height": shot.height,
                "durationSeconds": shot.duration,
                "hasAudio": _has_audio(shot),
                "shownWithCover": shot.kind == "audio",
                "description": shot.description,
                "labels": shot.labels,
                "analysis": shot.analysis,
            }
            for shot in shots
        ],
        "music": [
            {
                "songId": song.song_id,
                "mediaId": song.media_id,
                "name": song.name,
                "author": song.author,
                "durationSeconds": song.duration,
                "analysis": song.analysis,
            }
            for song in music
        ],
    }


def build(conn: psycopg.Connection, montage_id: uuid.UUID) -> MontageReport:
    settings = _settings(conn, montage_id)
    shots = _shots(conn, settings)
    music = _music(conn, settings)
    _coverage(shots, music)
    report: MontageReport = _report(settings, shots, music)
    landed = conn.execute(
        """
        update montages
        set manifest = jsonb_set(
            manifest, '{directingInput}',
            jsonb_build_object('mergedMontageData', %s::jsonb)
        )
        where id = %s
        """,
        (Jsonb(report), montage_id),
    ).rowcount
    if not landed:
        raise RuntimeError(f"montage {montage_id} went away while preparing")
    log.info("montage %s prepared: %s shots, %s tracks", montage_id, len(shots), len(music))
    return report
