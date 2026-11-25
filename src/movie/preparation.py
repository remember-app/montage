"""Загрузка заказа монтажа: настройки, кадры, музыка и событие."""

import dataclasses
import logging
import uuid

import psycopg

from src.movie.types import MontageLevel

log = logging.getLogger("movie.preparation")

THRESHOLD = 0.95


def level(settings: dict) -> MontageLevel:
    value = settings.get("level")
    return value if value in ("low", "medium", "high") else "low"


@dataclasses.dataclass
class ShotRow:
    item_id: str
    media_id: str
    kind: str
    comment: str | None
    analysis: dict | None
    duration: float | None
    width: int | None
    height: int | None
    captured_at: str | None
    thumbnail_key: str | None
    perceptual_hash: str
    key: str
    labels: list[str]
    place: tuple[float, float] | None = None

    @property
    def place_text(self) -> str | None:
        if self.place is None:
            return None
        return f"{self.place[0]:.4f},{self.place[1]:.4f}"

    @property
    def orientation(self) -> str | None:
        if not self.width or not self.height:
            return None
        if self.width > self.height:
            return "landscape"
        return "portrait" if self.height > self.width else "square"


@dataclasses.dataclass
class SongRow:
    song_id: str
    media_id: str
    name: str
    author: str | None
    duration: float | None
    analysis: dict | None
    key: str
    format: str


@dataclasses.dataclass
class Order:
    settings: dict
    event: dict
    shots: list[ShotRow]
    music: list[SongRow]


def _settings(conn: psycopg.Connection, montage_id: uuid.UUID) -> dict:
    row = conn.execute("select manifest from montages where id = %s", (montage_id,)).fetchone()
    if row is None:
        raise RuntimeError(f"montage {montage_id} is gone")
    settings = (row[0] or {}).get("preparationConfig") or {}
    if not settings:
        raise RuntimeError(f"montage {montage_id} has no preparation settings")
    return settings


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
        select pi.id, pi.original_id, pi.kind, pi.comment, pi.analysis,
               nullif(m.meta->>'duration', '')::float8,
               (m.meta->'dimensions'->>'width')::int,
               (m.meta->'dimensions'->>'height')::int,
               (m.meta->>'at')::timestamptz,
               coalesce(
                   (select pm.storage_key from media_items pm
                     where pm.id = (pi.previews_ids->>'middle')::uuid),
                   (select pm.storage_key from media_items pm
                     where pm.id = (pi.previews_ids->>'small')::uuid),
                   (select pm.storage_key from media_items pm where pm.id = pi.compressed_id)
               ),
               m.perceptual_hash,
               m.storage_key,
               (m.meta->'place'->>'latitude')::float8,
               (m.meta->'place'->>'longitude')::float8
        from project_items pi join media_items m on m.id = pi.original_id
        where pi.id = any(%s)
        order by coalesce((m.meta->>'at')::timestamptz, pi.created_at), m.id
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
            comment,
            analysis,
            duration,
            width,
            height,
            captured.isoformat() if captured else None,
            thumbnail_key,
            perceptual_hash or "",
            key,
            labels.get(str(item_id), []),
            (latitude, longitude) if latitude is not None and longitude is not None else None,
        )
        for (
            item_id,
            media_id,
            kind,
            comment,
            analysis,
            duration,
            width,
            height,
            captured,
            thumbnail_key,
            perceptual_hash,
            key,
            latitude,
            longitude,
        ) in rows
    ]


def _music(conn: psycopg.Connection, settings: dict) -> list[SongRow]:
    ids = [uuid.UUID(one) for one in settings.get("musicIds") or []]
    if not ids:
        return []
    rows = conn.execute(
        """
        select s.id, s.original_id, s.name, s.author, s.analysis,
               nullif(m.meta->>'duration', '')::float8,
               coalesce(
                   (select pc.storage_key from media_items pc where pc.id = s.compressed_id),
                   m.storage_key
               ),
               coalesce(
                   (select pc.format from media_items pc where pc.id = s.compressed_id),
                   m.format
               )
        from songs s join media_items m on m.id = s.original_id
        where s.original_id = any(%s)
        """,
        (ids,),
    ).fetchall()
    by_media = {str(row[1]): row for row in rows}
    tracks = []
    for media_id in ids:
        row = by_media.get(str(media_id))
        if row is None:
            raise RuntimeError(f"song {media_id} is not available to this project")
        tracks.append(
            SongRow(str(row[0]), str(row[1]), row[2], row[3], row[5], row[4], row[6], row[7])
        )
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


def _event(conn: psycopg.Connection, montage_id: uuid.UUID) -> dict:
    row = conn.execute(
        """
        select p.title, p.description, p.event_start, p.event_end,
               (select count(*) from project_members pm where pm.project_id = p.id)
        from montages m
        join projects p on p.id = m.project_id
        where m.id = %s
        """,
        (montage_id,),
    ).fetchone()
    if row is None:
        raise RuntimeError(f"montage {montage_id} has no event")
    title, description, start, end, people = row
    return {
        "title": title,
        "description": description,
        "window": {"start": start.isoformat(), "end": end.isoformat() if end else None},
        "people": int(people),
    }


def load(conn: psycopg.Connection, montage_id: uuid.UUID) -> Order:
    settings = _settings(conn, montage_id)
    shots = _shots(conn, settings)
    music = _music(conn, settings)
    _coverage(shots, music)
    return Order(settings, _event(conn, montage_id), shots, music)
