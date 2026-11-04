from montage import journal
from montage.media import understanding
from montage.video import layout, selection

PEOPLE_LABEL, ANIMAL_LABEL = "человек", "животное"

UNDESCRIBED = "Не удалось разобрать кадр"


@journal.quiet
def what_was_seen(conn, ref, judged, earlier, undescribed=()):
    seen = conn.execute(
        """SELECT media_id, description, cast_labels, details FROM media_understandings
           WHERE media_id = ANY(%s)""",
        ([piece.media_id for piece in judged],),
    ).fetchall()
    described = sum(one[0] not in earlier for one in seen)
    journal.start(
        conn,
        ref,
        "analysis",
        total=len(judged) - len(earlier),
        described=described,
        earlier=len(earlier),
    )
    spoken = _spoken(conn, [piece.media_id for piece in judged])
    comments = understanding.frame_comments(conn, [piece.media_id for piece in judged])
    cast, faces = set(), 0
    for media_id, description, labels, details in seen:
        labels = labels or ()
        cast.update(labels)
        faces += sum(label.startswith(PEOPLE_LABEL) for label in labels)
        journal.entry(
            conn,
            ref,
            "analysis",
            media_id=media_id,
            thought=description,
            labels=labels,
            details={
                **(details or {}),
                "people": sum(label.startswith(PEOPLE_LABEL) for label in labels),
                **({"earlier": True} if media_id in earlier else {}),
                **({"speech": spoken[media_id]} if media_id in spoken else {}),
                **({"comments": comments[media_id]} if comments.get(media_id) else {}),
            },
        )
    for media_id in undescribed:
        journal.entry(conn, ref, "analysis", media_id=media_id, thought=UNDESCRIBED)
    journal.finish(
        conn,
        ref,
        "analysis",
        described=described,
        earlier=len(earlier),
        faces=faces,
        people=sum(label.startswith(PEOPLE_LABEL) for label in cast),
        animals=sum(label.startswith(ANIMAL_LABEL) for label in cast),
    )


def _spoken(conn, media_ids):
    return {
        media_id: transcript
        for media_id, transcript in conn.execute(
            """SELECT media_id, transcript FROM media_speech
                WHERE media_id = ANY(%s) AND transcript <> ''""",
            (list(media_ids),),
        )
    }


def choices(conn, ref, chosen, turned_down, judged):
    told = {piece.media_id for piece in judged if piece.rejected}
    for piece in chosen:
        journal.entry(
            conn, ref, "directing", media_id=piece.media_id, decision="take", **_taken(piece)
        )
    for piece in turned_down:
        if piece.media_id in told or piece.rejected == selection.EXCLUDED:
            continue
        journal.entry(
            conn, ref, "directing", media_id=piece.media_id, decision="skip", reason=piece.rejected
        )


def plan(conn, ref, chosen, thoughts):
    laid = iter(chosen)
    for thought in thoughts:
        fields = dict(thought)
        if fields.get("decision") == "take":
            fields.update(_taken(next(laid)))
        journal.entry(conn, ref, "directing", **fields)


def _parts(pieces, thoughts):
    stood, spans = layout.starts(pieces), {}
    for thought, piece, at, after in zip(
        (one for one in thoughts if one.get("decision") == "take"),
        pieces,
        stood,
        [*stood[1:], stood[-1] + pieces[-1].seconds],
    ):
        if thought.get("story_part"):
            spans.setdefault(
                thought["story_part"],
                {
                    "title": journal.fit(thought).get("story_part_title"),
                    "thought": thought.get("thought"),
                    "from_seconds": at,
                    "pieces": [],
                },
            )["to_seconds"] = after
            spans[thought["story_part"]]["pieces"].append(piece)
    return spans.items()


def parts(pieces, thoughts):
    spans = list(_parts(pieces, thoughts))
    if not spans:
        return {}
    return {
        "parts": [
            {
                "title": span["title"],
                "from_seconds": round(span["from_seconds"], 1),
                "to_seconds": round(span["to_seconds"], 1),
            }
            for _part, span in spans
        ]
    }


def sound(conn, ref, pieces, thoughts, track):
    for part, span in _parts(pieces, thoughts):
        journal.entry(
            conn,
            ref,
            "music",
            story_part=part,
            story_part_title=span["title"],
            thought=span["thought"],
            details=_shares(span["pieces"], track),
        )


def _shares(pieces, track):
    if not track:
        return {"music": 0.0, "live": 1.0}
    return {"music": 1.0, "live": round(sum(one.live for one in pieces) / len(pieces), 2)}


def _taken(piece):
    fields = {"series_size": piece.series_size if piece.series_size > 1 else None}
    if piece.kind == "video":
        fields.update(
            source_from_seconds=round(piece.start, 2),
            source_to_seconds=round(piece.start + piece.seconds, 2),
        )
    return fields


def gluing(conn, ref, thoughts, done):
    titles = [
        journal.fit(one).get("story_part_title")
        for one in thoughts
        if one.get("decision") == "take"
    ]
    title = titles[done - 1] if 0 < done <= len(titles) else None
    journal.entry(
        conn, ref, "assembly", thought=f"Склейка {done}: {title}" if title else f"Склейка {done}"
    )
