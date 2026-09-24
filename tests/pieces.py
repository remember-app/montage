"""Куски ролика, собранные руками: ни файла, ни ffmpeg — только то, что читает отбор."""

import dataclasses
import datetime

from montage.video import selection


def at(second, sharpness, *, dated=True, media_id=""):
    """A piece shot `second` seconds past midnight, as sharp as it is told to be."""
    when = datetime.datetime(2026, 9, 5) + datetime.timedelta(seconds=second)
    return selection.Piece(
        media_id=media_id, sharpness=sharpness, when=when, captured_at=when if dated else None
    )


def clip(second, sharpness, *, duration=60.0, media_id=""):
    """A piece as `judge` cuts one out of a video file that runs `duration` seconds: the default
    four seconds around a best moment in the middle of it."""
    return dataclasses.replace(
        at(second, sharpness, media_id=media_id),
        kind="video",
        duration=duration,
        best_second=duration / 2,
        seconds=min(selection.CLIP_SECONDS, duration),
    )
