import json

from src.harness.tools.context import ToolContext
from src.integrations.ai import ToolCall, ToolResult

DESCRIPTION_LIMIT = 240
MOMENT_LIMIT = 100
MOMENTS = 4
SOUNDS = 2
TAGS = 12
TAG_LIMIT = 80


def run(context: ToolContext, call: ToolCall) -> ToolResult:
    ids = call.arguments.get("itemIds")
    if not isinstance(ids, list) or not ids or any(not isinstance(one, str) for one in ids):
        return _error(call, "itemIds must be a non-empty array of strings")
    if len(ids) > context.policy.max_get_shots_batch:
        return _error(call, f"at most {context.policy.max_get_shots_batch} ids per call")
    shots, missing = [], []
    for item_id in ids:
        shot = context.shots.get(item_id)
        if shot is None:
            missing.append(item_id)
            continue
        shots.append(_detail(shot))
    return ToolResult(
        call.id,
        json.dumps({"ok": True, "shots": shots, "missing": missing}, ensure_ascii=False),
        name=call.name,
    )


def _error(call: ToolCall, message: str) -> ToolResult:
    return ToolResult(
        call.id,
        json.dumps({"ok": False, "error": message}, ensure_ascii=False),
        is_error=True,
        name=call.name,
    )


def _detail(shot) -> dict:
    analysis = shot.analysis or {}
    entry = {
        "itemId": shot.item_id,
        "mediaId": shot.media_id,
        "kind": shot.kind,
        "capturedAt": shot.captured_at,
        "place": shot.place_text,
        "orientation": shot.orientation,
        "summary": _text(analysis.get("summary"), 240),
        "description": _text(analysis.get("description"), DESCRIPTION_LIMIT),
        "scoreImportance": analysis.get("scoreImportance"),
        "scoreQuality": analysis.get("scoreQuality"),
        "tags": _tags(analysis.get("tags")),
        "frames": _moments(analysis.get("frames"), MOMENTS),
        "sounds": _moments(analysis.get("sounds"), SOUNDS),
        "words": _moments(analysis.get("words"), MOMENTS),
        "comment": shot.comment,
        "labels": shot.labels,
    }
    return {key: value for key, value in entry.items() if value is not None}


def _tags(value) -> dict | None:
    if not isinstance(value, dict):
        return None
    tags = {
        str(key): str(item)[:TAG_LIMIT]
        for key, item in list(value.items())[:TAGS]
        if item is not None
    }
    return tags or None


def _moments(value, limit: int) -> list[dict]:
    result = []
    for one in (value or [])[:limit]:
        moment = _moment(one)
        if moment is not None:
            result.append(moment)
    return result


def _moment(one) -> dict | None:
    if not isinstance(one, dict):
        return None
    text = _text(one.get("description"), MOMENT_LIMIT)
    start, end = one.get("secStart"), one.get("secEnd")
    if text is None or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None
    return {
        "secStart": round(float(start), 2),
        "secEnd": round(float(end), 2),
        "description": text,
        "tags": one.get("tags"),
    }


def _text(value, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = "".join(character for character in value if character >= " ").strip()
    return text[:limit] or None


if __name__ == "__main__":
    import dataclasses

    from src.harness.policies import LOW
    from src.harness.tools.context import ToolContext
    from src.movie.preparation import Order, ShotRow

    def shot(item: str) -> ShotRow:
        return ShotRow(
            item,
            f"media-{item}",
            "photo",
            "the note",
            {
                "summary": "a moment",
                "description": "d" * 2000,
                "scoreImportance": 7,
                "scoreQuality": 6,
                "tags": {"light": "warm"},
                "frames": [{"secStart": 0, "secEnd": 2, "description": "frame"}],
            },
            2.0,
            1080,
            1920,
            None,
            "thumb",
            "0000000000000000",
            "original",
            ["family"],
        )

    order = Order({}, {}, [shot("a"), shot("b")], [])
    context = ToolContext(None, None, 0, LOW, order, {"a": shot("a"), "b": shot("b")}, {})
    result = run(context, ToolCall("c1", "get_shots", {"itemIds": ["a", "missing"]}))
    payload = json.loads(result.text)
    assert payload["ok"] and len(payload["shots"]) == 1 and payload["missing"] == ["missing"]
    assert len(payload["shots"][0]["description"]) == DESCRIPTION_LIMIT
    capped = dataclasses.replace(LOW, max_get_shots_batch=1)
    context = dataclasses.replace(context, policy=capped)
    refused = run(context, ToolCall("c2", "get_shots", {"itemIds": ["a", "b"]}))
    assert refused.is_error
    print("ok")
