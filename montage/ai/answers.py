import json
import math
import re

from montage.video import library

PARTS_LIMIT = 99
FOCUS_SIDES = ("x", "y", "w", "h")


def as_json(answer):
    try:
        parsed = loaded(answer)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, list):
        objects = [one for one in parsed if isinstance(one, dict)]
        if objects:
            return objects[0] if len(objects) == 1 else _folded(objects)
    elif isinstance(parsed, dict):
        return parsed
    raise RuntimeError(f"the model did not answer with a JSON object: {str(answer)[:500]}")


def _folded(objects):
    folded = {}
    for one in objects:
        for name, value in one.items():
            if name not in folded:
                folded[name] = value
    description = next(
        (one.get("description") for one in objects if said(one.get("description"))), None
    )
    if description is not None:
        folded["description"] = description
    cast = []
    for one in objects:
        for person in listed(one.get("cast")):
            if isinstance(person, str) and person not in cast:
                cast.append(person)
    if cast:
        folded["cast"] = cast
    return folded


_UNHOLDABLE = re.compile(r"[\x00\ud800-\udfff]")

DEPTH_LIMIT = 64

_NESTING = re.compile(r'"(?:[^"\\]|\\.)*+"?|[\[\]{}]', re.DOTALL)


def loaded(text):
    if isinstance(text, bytes):
        text = text.decode(json.detect_encoding(text), "surrogatepass")
    return _holdable(json.loads(_shallow(text)))


def _shallow(text):
    depth, kept, start = 0, [], 0
    for mark in _NESTING.finditer(text):
        if mark.group() in ("[", "{"):
            depth += 1
            if depth == DEPTH_LIMIT + 1:
                kept += (text[start : mark.start()], "null")
        elif mark.group() in ("]", "}"):
            depth -= 1
            if depth == DEPTH_LIMIT:
                start = mark.end()
    if depth <= DEPTH_LIMIT:
        kept.append(text[start:])
    return "".join(kept)


def _holdable(value):
    if isinstance(value, str):
        return _UNHOLDABLE.sub("", value)
    if isinstance(value, list):
        return [_holdable(one) for one in value]
    if isinstance(value, dict):
        return {_holdable(name): _holdable(one) for name, one in value.items()}
    if value is None or isinstance(value, bool):
        return value
    try:
        return value if math.isfinite(value) else None
    except OverflowError:
        return None


COUNT_LIMIT = 2**31 - 1


def counted(number):
    if isinstance(number, bool) or not isinstance(number, int):
        return None
    return number if 0 <= number <= COUNT_LIMIT else None


def planned(answered):
    if answered is None:
        return None
    left_out = {}
    for one in listed(answered.get("left_out")):
        media_id = _media_id(one)
        if media_id:
            left_out[media_id] = said(one.get("reason"))
    steps = (_step(one) for one in listed(answered.get("plan")))
    return {
        "overview": said(answered.get("overview")),
        "plan": [step for step in steps if step],
        "left_out": left_out,
        "track": track(answered.get("track")),
    }


def _step(step):
    media_id = _media_id(step)
    if media_id is None:
        return None
    return {
        "media_id": media_id,
        "start": number(step.get("start")),
        "seconds": number(step.get("seconds")),
        "live": _live(step.get("audio"), step.get("level")),
        "thought": said(step.get("thought")) or said(step.get("accent")),
        "part": _part(step.get("part")),
        "part_title": said(step.get("part_title")),
        "focus": _focus(step.get("focus")),
    }


def _media_id(step):
    if not isinstance(step, dict):
        return None
    media_id = step.get("media_id")
    if not isinstance(media_id, (str, int)) or isinstance(media_id, bool):
        return None
    return str(media_id).strip() or None


def said(words):
    if not isinstance(words, str):
        return None
    return words.strip() or None


def number(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _live(audio, level):
    if audio == "live":
        return 1.0
    if audio == "mix":
        level = number(level)
        return 0.5 if level is None else min(max(level, 0.0), 1.0)
    return 0.0


def track(value):
    return value if isinstance(value, str) and value in library.TRACKS else library.DEFAULT_TRACK


def _part(number):
    try:
        number = int(number)
    except (TypeError, ValueError):
        return None
    return number if 1 <= number <= PARTS_LIMIT else None


def _focus(frame):
    if not isinstance(frame, dict):
        return None
    try:
        x, y, width, height = (float(frame[side]) for side in FOCUS_SIDES)
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 <= x and 0 <= y and 0 < width and 0 < height and x + width <= 1 and y + height <= 1):
        return None
    return dict(zip(FOCUS_SIDES, (x, y, width, height)))


def listed(value):
    return value if isinstance(value, list) else []


FRAME_FIELDS = ("place", "light", "mood", "shot")


def described(answered):
    if not isinstance(answered, dict):
        return {}
    fields = {name: said(answered.get(name)) for name in FRAME_FIELDS}
    for name in ("camera", "text"):
        fields[name] = said(answered.get(name))
    moments = [
        {"seconds": number(one.get("seconds")), "text": said(one.get("text"))}
        for one in listed(answered.get("moments"))
        if isinstance(one, dict)
    ]
    moments = [
        one for one in moments if one["seconds"] is not None and one["seconds"] >= 0 and one["text"]
    ]
    fields = {name: words for name, words in fields.items() if words}
    if moments:
        fields["moments"] = moments
    return fields
