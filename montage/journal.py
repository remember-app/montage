import functools
import json
import logging
import unicodedata

from psycopg.types.json import Jsonb

ESTIMATE = {
    "prep": (1.5, 0.0),
    "analysis": (0.8, 0.0),
    "directing": (0.5, 0.0),
    "music": (0.2, 0.1),
    "assembly": (0.0, 1.4),
}

STAGES = tuple(ESTIMATE)

ASSEMBLY_STEPS = ("clips", "sound", "cover", "file")

ASSEMBLY_WEIGHTS = dict(
    zip(
        ASSEMBLY_STEPS,
        (75, *[8] * (len(ASSEMBLY_STEPS) - 2), 100 - 75 - 8 * (len(ASSEMBLY_STEPS) - 2)),
    )
)

COLUMNS = (
    "media_id",
    "thought",
    "decision",
    "reason",
    "series_size",
    "story_part",
    "story_part_title",
    "source_from_seconds",
    "source_to_seconds",
    "labels",
    "focus_x",
    "focus_y",
    "focus_w",
    "focus_h",
    "details",
)

THOUGHT_CHARACTERS = 500
REASON_CHARACTERS = 40
TITLE_CHARACTERS = 80
LABELS_COUNT, LABELS_CHARACTERS = 12, 240

DETAIL_CHARACTERS = {
    "place": 120,
    "light": 120,
    "mood": 120,
    "shot": 120,
    "camera": 120,
    "speech": 240,
    "text": 1500,
}
MOMENTS_COUNT, MOMENT_CHARACTERS = 8, 120

SHARES = ("music", "live")

log = logging.getLogger("montage.journal")


def quiet(write):

    @functools.wraps(write)
    def written(*args, **kwargs):
        try:
            return write(*args, **kwargs)
        except Exception as exc:
            log.warning("journal: %s did not land: %s", write.__name__, exc)

    return written


def seconds_for(stage, files, target_seconds):
    per_file, per_second = ESTIMATE[stage]
    return round(per_file * files + per_second * target_seconds)


@quiet
def begin(conn, ref, files, target_seconds):
    conn.execute("DELETE FROM montage_journal_entries WHERE montage_id = %s", (ref,))
    with conn.cursor() as cursor:
        cursor.executemany(
            """INSERT INTO montage_stages (montage_id, stage, eta_seconds) VALUES (%s, %s, %s)
               ON CONFLICT (montage_id, stage) DO UPDATE
                 SET state = 'pending', started_at = null, finished_at = null,
                     eta_seconds = excluded.eta_seconds, done_count = 0, total_count = null,
                     outcome = '{}', updated_at = now()""",
            [(ref, stage, seconds_for(stage, files, target_seconds)) for stage in STAGES],
        )


@quiet
def start(conn, ref, stage, total=None, **outcome):
    conn.execute(
        """UPDATE montage_stages
              SET state = 'running', started_at = now(), finished_at = null,
                  total_count = coalesce(%s, total_count),
                  outcome = outcome || %s::jsonb, updated_at = now()
            WHERE montage_id = %s AND stage = %s""",
        (total, _outcome(outcome), ref, stage),
    )


@quiet
def progress(conn, ref, stage, done, total=None, **outcome):
    conn.execute(
        """UPDATE montage_stages
              SET done_count = %(done)s,
                  total_count = coalesce(%(total)s, total_count),
                  outcome = outcome || %(outcome)s::jsonb,
                  eta_seconds = CASE
                      WHEN %(done)s > 0 AND started_at IS NOT NULL
                           AND coalesce(%(total)s, total_count) >= %(done)s
                      THEN greatest(round(extract(epoch FROM now() - started_at) / %(done)s
                                          * (coalesce(%(total)s, total_count) - %(done)s)), 0)
                      ELSE eta_seconds END,
                  updated_at = now()
            WHERE montage_id = %(ref)s AND stage = %(stage)s""",
        {"done": done, "total": total, "outcome": _outcome(outcome), "ref": ref, "stage": stage},
    )


@quiet
def finish(conn, ref, stage, **outcome):
    conn.execute(
        """UPDATE montage_stages
              SET state = 'done', finished_at = now(), eta_seconds = 0,
                  done_count = coalesce(total_count, done_count),
                  outcome = outcome || %s::jsonb, updated_at = now()
            WHERE montage_id = %s AND stage = %s""",
        (_outcome(outcome), ref, stage),
    )


def _outcome(values):
    return json.dumps({name: value for name, value in values.items() if value is not None})


@quiet
def appended(conn, ref, stage, **outcome):
    if not outcome:
        return
    conn.execute(
        """UPDATE montage_stages SET outcome = outcome || %s::jsonb, updated_at = now()
            WHERE montage_id = %s AND stage = %s""",
        (_outcome(outcome), ref, stage),
    )


def assembling(running, done=0, of=0):
    if running is None:
        before, share = ASSEMBLY_STEPS, 0.0
    else:
        before = ASSEMBLY_STEPS[: ASSEMBLY_STEPS.index(running)]
        share = min(max(done / of, 0.0), 1.0) if of else 0.0
    return {
        "steps": [
            {"step": code, "state": "done" if code in before else _state(code, running)}
            for code in ASSEMBLY_STEPS
        ],
        "percent": min(
            round(
                sum(ASSEMBLY_WEIGHTS[code] for code in before) + ASSEMBLY_WEIGHTS[running] * share
            )
            if running
            else 100,
            100,
        ),
    }


def _state(code, running):
    return "running" if code == running else "pending"


@quiet
def failed(conn, ref, stopped):
    conn.execute(
        """UPDATE montage_stages
              SET state = 'failed', finished_at = now(), eta_seconds = null,
                  outcome = outcome || %s::jsonb, updated_at = now()
            WHERE montage_id = %s AND state = 'running'""",
        (_outcome({"stopped": stopped}), ref),
    )


@quiet
def entry(conn, ref, stage, **fields):
    fields = fit(fields)
    unknown = set(fields) - set(COLUMNS)
    if unknown:
        raise ValueError(f"the journal has no column(s) {sorted(unknown)}")
    columns = [name for name in COLUMNS if fields.get(name) is not None]
    named = "".join(f", {one}" for one in columns)
    conn.execute(
        f"""INSERT INTO montage_journal_entries (montage_id, stage{named})
            VALUES (%s, %s{", %s" * len(columns)})""",
        (ref, stage, *(_jsonb(name, fields[name]) for name in columns)),
    )


def _jsonb(name, value):
    return Jsonb(value) if name == "details" else value


def fit(fields):
    fitted = dict(fields)
    for name, characters in (
        ("thought", THOUGHT_CHARACTERS),
        ("reason", REASON_CHARACTERS),
        ("story_part_title", TITLE_CHARACTERS),
    ):
        if fitted.get(name) is not None:
            fitted[name] = _clean(fitted[name])[:characters] or None
    if fitted.get("labels") is not None:
        labels = [_clean(one) for one in fitted["labels"]]
        labels = [one for one in labels if one][:LABELS_COUNT]
        while labels and len(",".join(labels)) > LABELS_CHARACTERS:
            labels.pop()
        fitted["labels"] = labels or None
    if fitted.get("details") is not None:
        fitted["details"] = _fit_details(fitted["details"]) or None
    return fitted


def _fit_details(details):
    fitted = {}
    for name, characters in DETAIL_CHARACTERS.items():
        words = _clean(details.get(name) or "")[:characters]
        if words:
            fitted[name] = words
    moments = [
        {"seconds": moment["seconds"], "text": _clean(moment.get("text") or "")[:MOMENT_CHARACTERS]}
        for moment in (details.get("moments") or [])[:MOMENTS_COUNT]
        if isinstance(moment, dict)
        and isinstance(moment.get("seconds"), (int, float))
        and moment["seconds"] >= 0
    ]
    moments = [one for one in moments if one["text"]]
    if moments:
        fitted["moments"] = moments
    for name in ("people", "earlier"):
        if details.get(name) is not None:
            fitted[name] = details[name]
    for name in SHARES:
        share = details.get(name)
        if isinstance(share, (int, float)) and not isinstance(share, bool):
            fitted[name] = round(min(max(float(share), 0.0), 1.0), 2)
    if details.get("comments"):
        fitted["comments"] = [
            words for words in (_clean(one) for one in details["comments"]) if words
        ] or None
    return fitted


def _clean(words):
    letters = []
    for letter in str(words):
        category = unicodedata.category(letter)
        if category == "Cc":
            letters.append(" ")
        elif category != "Cf":
            letters.append(letter)
    return " ".join("".join(letters).split())
