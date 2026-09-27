"""Куски ролика, собранные руками: ни файла, ни ffmpeg — только то, что читает отбор."""

import dataclasses
import datetime
import random

from PIL import Image, ImageEnhance

from montage.video import frames, selection


def at(second, sharpness, *, dated=True, media_id="", frame=b""):
    """A piece shot `second` seconds past midnight, as sharp as it is told to be."""
    when = datetime.datetime(2026, 9, 5) + datetime.timedelta(seconds=second)
    return selection.Piece(
        media_id=media_id,
        sharpness=sharpness,
        when=when,
        captured_at=when if dated else None,
        frame=frame,
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


def picture(scene, *, moved=(0, 0), light=1.0, stepped_in=None):
    """A sampled frame (`frames.sampled`) of a made-up scene, as a phone would have taken it.

    The scene is a field of soft light and shadow, the same for the same `scene` and unlike any
    other — what a photograph is at 96×54, rather than the noise or the checkerboard a sharpness
    test wants. `moved` is the camera pointed that many pixels aside (ten at most), `light` the
    exposure it chose, and `stepped_in` another scene over the bottom right eighth of the frame —
    somebody walking into it.
    """
    width, height = frames.SAMPLE_WIDTH, frames.SAMPLE_HEIGHT

    def field(number):
        rough = Image.frombytes("L", (8, 5), random.Random(number).randbytes(8 * 5))
        return rough.resize((width + 20, height + 20), Image.Resampling.BICUBIC)

    x, y = 10 + moved[0], 10 + moved[1]
    shot = field(scene).crop((x, y, x + width, y + height))
    if stepped_in is not None:
        corner = (width // 4, height // 2)
        shot.paste(field(stepped_in).crop((0, 0, *corner)), (width - corner[0], height - corner[1]))
    return ImageEnhance.Brightness(shot).enhance(light).tobytes()
