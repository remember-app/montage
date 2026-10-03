"""Где в ролике стоит каждый взятый кадр — в `montage_frames`, для тех, кто спросит об этом снаружи.

Таблица принадлежит api (его миграция `100.do.frame-info.sql`), и пишет её только этот модуль:
api читает её, чтобы ответить «в каких роликах стоит этот кадр и на какой секунде» на листе кадра
(doc-02) и чтобы расставить кадры в журнале монтажа.
"""

import logging

from montage.video import assembly

log = logging.getLogger("montage.video.layout")


def starts(pieces):
    """Where each piece stands in the video, in seconds from its start — the same count `write`
    lays the chain out by, so anything that names a place in the finished video (a part of the
    story in the journal's stage outcome) says where that piece actually stands."""
    at, stood = 0.0, []
    for piece in pieces:
        stood.append(at)
        at += piece.seconds - assembly.CROSSFADE_SECONDS
    return stood


def write(conn, ref, pieces):
    """Where each piece stands in this montage, into `montage_frames` — in place of whatever an
    earlier assembly of the same order left there.

    The table is the api's (its migration `100.do.frame-info.sql`) and this is the only thing that
    writes it: the api reads it to answer «в каких роликах стоит этот кадр и на какой секунде» on
    the frame sheet (doc-02), for `ready` montages only, and to place frames in the montage journal,
    for a montage in any state — which is why an attempt starts by writing it empty. One row
    per *appearance* rather than per frame, which is why the key carries the second: a file may
    legitimately stand in a video twice — a clip cut in two, a photograph that comes back at the
    end. The whole of m-24 and its neighbours read this one table; a montage's layout written down
    a second time somewhere else would be two answers to the same question.

    The seconds are counted the way `assembly` lays the chain out: each join eats half a second of
    the total, so a piece starts where the one before it ends, less the crossfade the two share.

    Nothing here may cost the montage. A video that takes minutes to assemble is not thrown away
    because an insert did not land, so a failure is a line in the log and the assembly goes on —
    the frame sheet would then show this montage under none of its frames, which is a montage
    missing from a list and not a montage missing.
    """
    rows = [(ref, piece.media_id, at, piece.seconds) for piece, at in zip(pieces, starts(pieces))]
    try:
        # Both statements or neither, on a connection that otherwise commits each on its own: this
        # layout is read while the video is still being assembled, and between the two an emptied
        # table reads as «этого кадра нет ни в одном ролике».
        with conn.transaction():
            conn.execute("DELETE FROM montage_frames WHERE montage_id = %s", (ref,))
            with conn.cursor() as cursor:
                cursor.executemany(
                    """INSERT INTO montage_frames
                         (montage_id, media_id, start_seconds, duration_seconds)
                       VALUES (%s, %s, %s, %s)""",
                    rows,
                )
    except Exception as exc:
        log.warning("montage %s: its layout was not written, assembling it anyway: %s", ref, exc)
