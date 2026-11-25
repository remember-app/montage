"""Компактный контекст для агента: индекс кадров, музыка и контактный лист."""

import dataclasses
import datetime
import io
import logging
import math
import pathlib
import tempfile
import uuid

import psycopg
from PIL import Image, ImageDraw, ImageFont
from psycopg.types.json import Jsonb

from src.harness.policies import LevelPolicy
from src.integrations import storage
from src.integrations.ai import Blob
from src.media import perceptual
from src.movie import preparation

log = logging.getLogger("harness.context")

DUPLICATE_BITS = 10
DUPLICATE_SECONDS = 30.0
SUMMARY_LIMIT = 240
SECTION_LIMIT = 20
TILE = 240


@dataclasses.dataclass
class Context:
    order: preparation.Order
    data: dict
    contact: Blob | None
    labels: list[dict]


def build(conn: psycopg.Connection, montage_id: uuid.UUID, policy: LevelPolicy) -> Context:
    order = preparation.load(conn, montage_id)
    if len(order.shots) > policy.max_shots:
        raise RuntimeError(
            f"the order has {len(order.shots)} shots; level {policy.name} takes up to "
            f"{policy.max_shots}"
        )
    data = _data(order, policy)
    landed = conn.execute(
        """
        update montages
        set manifest = jsonb_set(
                manifest, '{directingInput}',
                jsonb_build_object('mergedMontageData', %s::jsonb)
            )
        where id = %s
        """,
        (Jsonb(data), montage_id),
    ).rowcount
    if not landed:
        raise RuntimeError(f"montage {montage_id} went away while preparing context")
    contact, labels = _contact_sheet(order, policy)
    log.info(
        "montage %s context: %s shots, %s tracks, contact sheet %s",
        montage_id,
        len(order.shots),
        len(order.music),
        "yes" if contact else "no",
    )
    return Context(order, data, contact, labels)


def _data(order: preparation.Order, policy: LevelPolicy) -> dict:
    required = {str(one) for one in order.settings.get("requiredItemIds") or []}
    groups = _duplicate_groups(order.shots)
    return {
        "movie": {
            "event": order.event,
            "span": _span(order.shots),
            "targetSeconds": int(order.settings.get("targetSeconds") or 0),
            "dimensions": order.settings.get("dimensions") or {},
            "direction": order.settings.get("prompt"),
            "requiredItemIds": sorted(required),
            "level": order.settings.get("level") or policy.name,
        },
        "shots": [_shot(shot, required, groups) for shot in order.shots],
        "music": [_music(song) for song in order.music],
    }


def _shot(shot: preparation.ShotRow, required: set[str], groups: dict[str, str]) -> dict:
    analysis = shot.analysis or {}
    entry = {
        "itemId": shot.item_id,
        "mediaId": shot.media_id,
        "kind": shot.kind,
        "capturedAt": shot.captured_at,
        "place": shot.place_text,
        "orientation": shot.orientation,
        "durationSeconds": shot.duration,
        "scoreImportance": analysis.get("scoreImportance"),
        "scoreQuality": analysis.get("scoreQuality"),
        "summary": _text(analysis.get("summary"), SUMMARY_LIMIT),
        "labels": shot.labels,
        "duplicateGroup": groups.get(shot.item_id),
        "required": shot.item_id in required,
        "hasAudio": _has_audio(shot),
    }
    return {key: value for key, value in entry.items() if value is not None or key in _FLAGS}


_FLAGS = ("required", "hasAudio")


def _music(song: preparation.SongRow) -> dict:
    analysis = song.analysis or {}
    sections = []
    for one in (analysis.get("sounds") or [])[:SECTION_LIMIT]:
        if not isinstance(one, dict):
            continue
        text = _text(one.get("description"), 160)
        start, end = one.get("secStart"), one.get("secEnd")
        if text is None or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        sections.append(
            {
                "secStart": round(float(start), 2),
                "secEnd": round(float(end), 2),
                "description": text,
            }
        )
    entry = {
        "songId": song.song_id,
        "mediaId": song.media_id,
        "name": song.name,
        "author": song.author,
        "durationSeconds": song.duration,
        "summary": _text(analysis.get("summary"), SUMMARY_LIMIT),
        "sections": sections,
    }
    return {key: value for key, value in entry.items() if value is not None}


def _span(shots: list[preparation.ShotRow]) -> dict | None:
    times = sorted(shot.captured_at for shot in shots if shot.captured_at)
    if not times:
        return None
    return {"first": times[0], "last": times[-1], "days": len({time[:10] for time in times})}


def _has_audio(shot: preparation.ShotRow) -> bool:
    if shot.kind == "audio":
        return True
    if shot.kind == "photo":
        return False
    analysis = shot.analysis or {}
    tags = analysis.get("tags") or {}
    return bool(analysis.get("sounds") or analysis.get("words") or tags.get("soundDescription"))


def _text(value, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = "".join(character for character in value if character >= " ").strip()
    return text[:limit] or None


def _duplicate_groups(shots: list[preparation.ShotRow]) -> dict[str, str]:
    parent = list(range(len(shots)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        left, right = find(first), find(second)
        if left != right:
            parent[right] = left

    for first in range(len(shots)):
        for second in range(first + 1, len(shots)):
            if _similar(shots[first], shots[second]):
                union(first, second)

    groups: dict[int, list[int]] = {}
    for index in range(len(shots)):
        groups.setdefault(find(index), []).append(index)

    result: dict[str, str] = {}
    number = 0
    for members in groups.values():
        if len(members) < 2:
            continue
        number += 1
        label = f"g{number}"
        for index in members:
            result[shots[index].item_id] = label
    return result


def _similar(first: preparation.ShotRow, second: preparation.ShotRow) -> bool:
    if not first.perceptual_hash or not second.perceptual_hash:
        return False
    if perceptual.distance(first.perceptual_hash, second.perceptual_hash) > DUPLICATE_BITS:
        return False
    if first.captured_at and second.captured_at:
        left, right = _time(first.captured_at), _time(second.captured_at)
        if left and right:
            try:
                return abs((left - right).total_seconds()) <= DUPLICATE_SECONDS
            except TypeError:
                return True
    return True


def _time(value: str) -> datetime.datetime | None:
    try:
        return datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _contact_sheet(order: preparation.Order, policy: LevelPolicy) -> tuple[Blob | None, list[dict]]:
    required = {str(one) for one in order.settings.get("requiredItemIds") or []}
    chosen = sorted(
        order.shots,
        key=lambda shot: (
            shot.item_id not in required,
            -_score(shot, "scoreImportance"),
            -_score(shot, "scoreQuality"),
        ),
    )[: policy.contact_sheet_shots]
    chosen = [shot for shot in chosen if shot.thumbnail_key]
    if not chosen:
        return None, []

    tiles: list[Image.Image] = []
    labels: list[dict] = []
    with tempfile.TemporaryDirectory() as directory:
        for shot in chosen:
            key = shot.thumbnail_key
            if not key:
                continue
            label = len(labels) + 1
            path = pathlib.Path(directory) / f"tile-{label}.jpg"
            try:
                storage.download(key, path)
                image = Image.open(path).convert("RGB")
            except Exception:
                log.warning("contact sheet: cannot read the thumb of %s", shot.item_id)
                continue
            tiles.append(_tile(image, label))
            labels.append({"label": label, "itemId": shot.item_id})

    if not tiles:
        return None, []
    columns = min(4, len(tiles))
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (columns * TILE, rows * TILE), (0, 0, 0))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * TILE, (index // columns) * TILE))
    buffer = io.BytesIO()
    sheet.save(buffer, "JPEG", quality=85)
    return Blob("image/jpeg", buffer.getvalue()), labels


def _tile(image: Image.Image, label: int) -> Image.Image:
    image.thumbnail((TILE, TILE))
    tile = Image.new("RGB", (TILE, TILE), (0, 0, 0))
    tile.paste(image, ((TILE - image.width) // 2, (TILE - image.height) // 2))
    draw = ImageDraw.Draw(tile)
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:
        font = ImageFont.load_default()
    text = str(label)
    box = draw.textbbox((0, 0), text, font=font)
    draw.rectangle((4, 4, 4 + box[2] + 8, 4 + box[3] + 8), fill=(0, 0, 0))
    draw.text((8, 6), text, fill=(255, 255, 255), font=font)
    return tile


def _score(shot: preparation.ShotRow, key: str) -> float:
    value = (shot.analysis or {}).get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


if __name__ == "__main__":
    timestamp = "2026-11-16T12:00:00+00:00"

    def shot(item: str, digest: str, audio: bool = False) -> preparation.ShotRow:
        analysis = {
            "summary": "a moment",
            "scoreImportance": 7,
            "scoreQuality": 6,
            "sounds": [{"description": "laughter"}] if audio else [],
            "tags": {"soundDescription": "laughter"} if audio else {},
        }
        return preparation.ShotRow(
            item,
            f"media-{item}",
            "video" if audio else "photo",
            None,
            analysis,
            3.0,
            1080,
            1920,
            timestamp,
            "thumb",
            digest,
            "original",
            [],
        )

    shots = [
        shot("a", "0000000000000000", True),
        shot("b", "0000000000000001", True),
        shot("c", "ffffffffffffffff"),
    ]
    groups = _duplicate_groups(shots)
    assert groups == {"a": "g1", "b": "g1"}, groups
    data = _data(
        preparation.Order(
            {"targetSeconds": 60, "dimensions": {"width": 1080, "height": 1920}},
            {"title": "Evening", "description": ""},
            shots,
            [],
        ),
        type("Policy", (), {"name": "low", "contact_sheet_shots": 20})(),
    )
    assert data["shots"][0]["duplicateGroup"] == "g1"
    assert data["shots"][0]["hasAudio"] is True
    assert data["shots"][2]["hasAudio"] is False
    assert "description" not in data["shots"][0]
    assert "frames" not in data["shots"][0]
    assert data["shots"][0]["orientation"] == "portrait"
    assert data["movie"]["span"] == {
        "first": timestamp,
        "last": timestamp,
        "days": 1,
    }
    assert shots[0].place_text is None
    assert shots[0].orientation == "portrait"

    tile = _tile(Image.new("RGB", (100, 100), (255, 0, 0)), 1)
    assert tile.size == (TILE, TILE)
    print("ok")
