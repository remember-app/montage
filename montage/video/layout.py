import logging

from montage.video import assembly

log = logging.getLogger("montage.video.layout")


def starts(pieces):
    at, stood = 0.0, []
    for piece in pieces:
        stood.append(at)
        at += piece.seconds - assembly.CROSSFADE_SECONDS
    return stood


def write(conn, ref, pieces):
    rows = [(ref, piece.media_id, at, piece.seconds) for piece, at in zip(pieces, starts(pieces))]
    try:
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
