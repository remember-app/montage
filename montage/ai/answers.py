"""Что вернула модель — тем, что из этого можно прочитать и положить в базу.

Ответ облачной модели — чужой текст: ни один слой выше не смеет верить ни его форме, ни его
содержимому. Здесь он становится либо значением, которое `jsonb` и `text` точно возьмут, либо
ничем — и «ничем» тут всегда осознанный ответ, а не молчание об ошибке.
"""

import json
import math
import re

from montage.video import library

# Частей в ролике единицы, а `story_part` в журнале — `integer` с `CHECK (> 0)`: номер больше
# этого — не номер части, и база отвергла бы с ним всю реплику.
PARTS_LIMIT = 99
FOCUS_SIDES = ("x", "y", "w", "h")


def as_json(answer):
    """The model's words as the object they were asked to be. A model that answered in prose has
    failed the call, and the text of it goes into the journal as the error — that is the only
    place anyone will look for why.

    An object and not merely valid JSON: every caller here reads fields off what comes back, and a
    bare `null` or a list parses perfectly well and then fails three handlers down as an
    `AttributeError` about `NoneType`, with the answer that caused it nowhere to be seen.

    Read by `loaded`, the same as the provider's envelope the text came in.
    """
    try:
        parsed = loaded(answer)
    except (ValueError, TypeError):
        # A JSONDecodeError is a ValueError; a TypeError is an answer that is no text at all.
        parsed = None
    if not isinstance(parsed, dict):
        raise RuntimeError(f"the model did not answer with a JSON object: {str(answer)[:500]}")
    return parsed


# NUL, and half of a UTF-16 pair: `\u0000` and a lone `\ud800` both parse into a `str`, and neither
# `jsonb` nor a `text` column takes either. A pair that belongs together is one character by then.
_UNHOLDABLE = re.compile(r"[\x00\ud800-\udfff]")

# An envelope or a director's plan nests four or five levels deep. Parsing gives out an order of
# magnitude further in: `_holdable` at Python's recursion limit (1000), `json.loads` at its own
# (about 10000 on 3.12, the C stack on 3.14) — and a model can nest past both in one answer.
DEPTH_LIMIT = 64

# One string, whole — the brackets in it are words — or one bracket. The closing quote is optional
# so that a string left open runs to the end once, not again from every quote inside it. The repeat
# is possessive: a greedy one keeps a backtracking frame per character, a gigabyte for 10 MB.
_NESTING = re.compile(r'"(?:[^"\\]|\\.)*+"?|[\[\]{}]', re.DOTALL)


def loaded(text):
    """JSON off the wire — the provider's envelope, and the model's own JSON inside it — with
    nothing left in it that Postgres refuses.

    Every answer reaches the journal, the tables and the caller through here. `json.loads` hands
    back what `jsonb` and `text` do not take: `NaN`, `Infinity` and a number past a double
    (`1e999`), a NUL and a lone surrogate out of `\\u` escapes. Let through, any one of them breaks
    a write after the answer is paid for, and the retry pays for it again; refusing the answer over
    it costs the same. So the value goes and the answer stays: a number nothing can hold becomes
    `null`, a character nothing can hold is dropped. An integer too long for a double goes the same
    way: every reader here takes a number as a float, and `float()` raises on it rather than
    answering `inf`.

    Nesting past `DEPTH_LIMIT` goes the same way, and before anything parses it: an array or an
    object opened deeper is `null`, unread — so neither `json.loads` nor `_holdable` ever recurses
    further than that.
    """
    if isinstance(text, bytes):
        text = text.decode(json.detect_encoding(text), "surrogatepass")  # as `json.loads` would
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


# The journal keeps a count in an `integer`; one past it would fail the UPDATE after the call is
# paid for.
COUNT_LIMIT = 2**31 - 1


def counted(number):
    """A token count off the provider's envelope — a whole number the journal can hold, or none. A
    string, a fraction or a negative there is no count, and not worth failing an answer over."""
    if isinstance(number, bool) or not isinstance(number, int):
        return None
    return number if 0 <= number <= COUNT_LIMIT else None


# --- чтение плана режиссёра -------------------------------------------------
# Ответ модели — недоверенный ввод, а слова режиссёра увидят все участники воспоминания и рамку
# «куда смотрит» приложение нарисует поверх кадра. Поэтому план разбирается здесь, у самой двери, и
# выше уходит только то, что выдержало чтение: шаг без куска — не шаг, длина и начало отрезка не
# конечным числом, часть не номером и рамка не четырьмя долями внутри кадра — их просто нет, а
# звук, названный не так, — музыка, трек не с полки — умолчание. Чей кусок назван, здесь не
# проверить: что показали модели, знает `director`, он и сверяет.


def planned(answered):
    """План словами режиссёра — тем, что из него можно прочитать.

    План без новых полей (записанный ответ, снятый до того, как режиссёра попросили говорить)
    читается этим же разбором: мыслей, частей, рамок и начал отрезков в нём просто нет, а порядок и
    длины — есть. `accent` — старое имя мысли, и он ею и становится.

    Один и тот же кусок может стоять в плане не раз — разными отрезками одного видео, — и шаги здесь
    не схлопываются: что из них один и тот же отрезок, решает `director`, уже отрезав его.
    """
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
    """Один шаг плана: чей кусок, с какой секунды файла и сколько секунд, как он звучит и что
    режиссёр о нём сказал."""
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
    """Кусок, про который шаг, — строкой, как его назвали в вопросе. Не объект или без куска —
    и шага нет: длина и мысль без кадра, к которому они относятся, ничьи."""
    if not isinstance(step, dict):
        return None
    media_id = step.get("media_id")
    if not isinstance(media_id, (str, int)) or isinstance(media_id, bool):
        return None
    return str(media_id).strip() or None


def said(words):
    """Слова модели — строкой, или ничего вовсе: мысль, которую покажут людям, она либо сказала,
    либо нет, и число или объект на её месте словами не станут. Длину держит и служебные символы
    вычищает журнал (`journal.fit`) — всё, что модель говорит, идёт в базу через него."""
    if not isinstance(words, str):
        return None
    return words.strip() or None


def number(value):
    """Число — длина куска, его начало в файле, громкость — конечным, иначе никакое. Строку `"NaN"`
    `float` читает в «не число», а оно проходит сквозь любые `min` и `max`, которыми его потом
    держат."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _live(audio, level):
    """Сколько на куске слышно его собственного звука, долей от 0 до 1 (`selection.Piece.live`).

    `"music"` — нисколько, `"live"` — целиком, `"mix"` — `level`, втянутый в долю, а без
    читаемого `level` — поровну: смесь режиссёр попросил, а громкость не назвал. Всё остальное —
    нет поля, неизвестное слово, не строка — музыка, как звучал любой кусок до выбора звука.
    """
    if audio == "live":
        return 1.0
    if audio == "mix":
        level = number(level)
        return 0.5 if level is None else min(max(level, 0.0), 1.0)
    return 0.0


def track(value):
    """Трек, названный режиссёром из полки, — id полки, иначе «тёплая» по умолчанию.

    Полка закрыта (`library.TRACKS`): файл названного трека достаёт воркер, и трека вне полки у
    него нет. Трека в ответе нет, назван он не строкой или такого трека на полке нет — умолчание,
    а не падение: без музыки ролик не собирается, и выдумка модели стоит одной подстановки.
    """
    return value if isinstance(value, str) and value in library.TRACKS else library.DEFAULT_TRACK


def _part(number):
    """Номер части истории — целым и небольшим, иначе никаким (`PARTS_LIMIT`)."""
    try:
        number = int(number)
    except (TypeError, ValueError):
        return None
    return number if 1 <= number <= PARTS_LIMIT else None


def _focus(frame):
    """Рамка «куда смотрит» — четыре доли кадра, или ничего вовсе.

    Долями, а не пикселями: приложение рисует её поверх миниатюры любого размера. Либо все четыре,
    либо ни одной — три доли база не примет и отвергнет с ними всю реплику, — и только внутри
    кадра: рамка, вылезающая за край, съехала бы по чужому кадру. Не сходится что угодно из этого
    — рамки нет, а мысль остаётся: выдумка модели не стоит того, чтобы потерять её слова.
    """
    if not isinstance(frame, dict):
        return None
    try:
        x, y, width, height = (float(frame[side]) for side in FOCUS_SIDES)
    except (KeyError, TypeError, ValueError):
        return None
    # `NaN` проходит сквозь любое сравнение ложью, поэтому проверка написана «должно быть так», а
    # не «не должно быть эдак»: рамка с ним не сходится и выбрасывается вместе с остальными.
    if not (0 <= x and 0 <= y and 0 < width and 0 < height and x + width <= 1 and y + height <= 1):
        return None
    return dict(zip(FOCUS_SIDES, (x, y, width, height)))


def listed(value):
    """То, что пришло списком, — списком; всё остальное пусто. Строку иначе разобрало бы по
    буквам, а каждая буква стала бы шагом плана."""
    return value if isinstance(value, list) else []


# --- описание кадра ----------------------------------------------------------
# Описание идёт в журнал полями (контракт api, «details» у анализа), и у каждого поля свой потолок:
# место, свет, настроение, план и движение камеры — короткой строкой, а подробное описание текстом —
# длиннее всех. Потолки держит `journal.fit`; у самой двери остаётся форма, потому что без неё в
# `details` едет то, чему там не место.

# Поля, которые модель называет одинаково у фотографии и у клипа, и которые журнал рисует одинаково.
FRAME_FIELDS = ("place", "light", "mood", "shot")


def described(answered):
    """Описание кадра по полям — тем, что из ответа модели выдержало чтение.

    Поле, которого модель не назвала или назвала не строкой, просто отсутствует: приложение ключ,
    которого нет, не рисует (контракт api), а выдуманное нами «не разглядеть» стоило бы участникам
    воспоминания чужого слова на их карточке. `moments` — моменты клипа, у каждого своя секунда и
    свой текст; момент без конечной секунды, без текста или с секундой до нуля — не момент.

    Чего в кадре не слышно, здесь не спрашивают: модель видит вырезанные кадры и звука не слышит
    вообще, и просить её описать то, что слышно, значило бы выдумать это и показать людям. Речь
    приходит из `media_speech` — оттуда же, откуда её читает режиссёр.
    """
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
