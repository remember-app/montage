"""Что видно в файле: насколько кадр резкий, не чёрный ли он и где в клипе лучший момент.

Один файл, ничего вокруг: сюда не доходит ни заказ, ни слово организатора, ни длина ролика. Эти же
числа получает отбор, и они же — единственное, что меняется, когда «насколько хорош кадр» начнут
считать моделью, а не резкостью.
"""

import collections
import logging

import numpy

from montage import ffmpeg

# Every frame is judged on a small grey copy of itself: blur and exposure both survive the scaling,
# and 96×54 is 5k pixels rather than 2M — cheap enough to judge every sampled second of every file
# in the memory.
#
# What is handed in matters to these numbers, though, and not only to how long they take: swscale
# cannot fit its filter past about a 64× reduction, so a 12032px original squeezed straight to 96
# is *sampled* rather than averaged. That reads high on sharpness (aliasing counts as detail) and
# wrong on exposure — a frame whose true mean luma is 25 came out at 13. A photograph therefore
# arrives here as the 2048px copy the preview made of it (`storage.large_copy`), which is a plain
# average of every pixel, and the thresholds below are set against that (TASK-184).
SAMPLE_WIDTH, SAMPLE_HEIGHT = 96, 54
SAMPLE_FPS = 1.0  # «семпл кадров ≈1/сек»
MIN_SHARPNESS = 5.0  # variance of the Laplacian under which a frame is just blur
LUMA_RANGE = (24, 232)  # mean brightness outside this is a black or a blown-out frame
CLIPPED_LIMIT = 0.5  # and so is one whose histogram is mostly at either end

# Почему в кадре нечего показывать — коды, с которыми отбор потом живёт дальше (`selection.py`,
# там же и остальные его причины).
UNREADABLE = "unreadable"  # ffmpeg would not open it, or found no frame in it at all
BLUR = "blur"
EXPOSURE = "exposure"

log = logging.getLogger("montage.video.frames")


def sampled(path, per_second=None):
    """The file as raw 96×54 grey frames — one a second for a video, its only one for a photo.

    One ffmpeg call and one bytes object: everything below reads those pixels directly, so nothing
    is decoded twice and no image library is needed to ask how sharp a frame is.
    """
    sampling = f"fps={per_second}," if per_second else ""
    framing = () if per_second else ("-frames:v", "1")  # a photo has the one frame to look at
    pixels = ffmpeg.raw(
        "-i",
        str(path),
        *framing,
        "-vf",
        f"{sampling}scale={SAMPLE_WIDTH}:{SAMPLE_HEIGHT},format=gray",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    )
    frame = SAMPLE_WIDTH * SAMPLE_HEIGHT
    return [pixels[at : at + frame] for at in range(0, len(pixels) - frame + 1, frame)]


def _sharpness(frame):
    """Variance of the Laplacian: how much the frame changes from pixel to pixel. A photograph in
    focus is full of edges and scores high; a smeared one has almost none and scores near zero.

    The shifted slices are the same five neighbours a per-pixel loop would visit, over the same
    interior, and `.var()` (ddof=0) is `statistics.pvariance` under another name — this is the old
    Python loop's number, ~100x faster. That win is the whole reason the worker depends on numpy.
    """
    # Signed before the subtraction: as `uint8` the neighbours would wrap around instead of going
    # negative, and a dark pixel next to a bright one is exactly what this is looking for.
    pixels = numpy.frombuffer(frame, dtype=numpy.uint8).reshape(SAMPLE_HEIGHT, SAMPLE_WIDTH)
    pixels = pixels.astype(numpy.int16)
    laplacian = (
        4 * pixels[1:-1, 1:-1]
        - pixels[1:-1, :-2]
        - pixels[1:-1, 2:]
        - pixels[:-2, 1:-1]
        - pixels[2:, 1:-1]  # the one-pixel border has no four neighbours
    )
    return float(laplacian.var())


def _exposed(frame):
    """False for a frame that is mostly black or blown out, off its own histogram."""
    histogram = collections.Counter(frame)
    pixels = len(frame)
    mean = sum(value * count for value, count in histogram.items()) / pixels
    clipped = sum(count for value, count in histogram.items() if value < 16 or value > 239) / pixels
    return LUMA_RANGE[0] <= mean <= LUMA_RANGE[1] and clipped <= CLIPPED_LIMIT


def best_moment(frames):
    """`(second, sharpness)` of the frame the piece is worth cutting around, and `None` beside it —
    or, when no frame in the file is worth showing, the sharpest one there is and the reason.

    A rejected file gets a moment back too, because the rejection is only a proposal: the organizer
    can overrule it («включить»), and a dark clip that goes into the video anyway should still go in
    cut around the most detailed thing in it rather than around whatever its first frame happens to
    be. `None` for the moment means there was no frame at all to have an opinion about.

    ponytail: sharpness is standing in for the ranking architecture.md asks for — NIMA over ONNX,
    which is «how good is this picture» rather than «how in focus is it». Deliberately deferred:
    the model's weights, provenance and licence are a question for the human, not a default to pick
    here. When it is answered, the score returned from this function is the only thing that changes.
    """
    best = usable = None  # the sharpest frame of all, and the sharpest one worth showing
    exposed = False
    for index, frame in enumerate(frames):
        second, sharpness = index / SAMPLE_FPS, _sharpness(frame)
        if best is None or sharpness > best[1]:
            best = (second, sharpness)
        if not _exposed(frame):
            continue
        exposed = True
        if sharpness >= MIN_SHARPNESS and (usable is None or sharpness > usable[1]):
            usable = (second, sharpness)
    if usable:
        return usable, None
    if best is None:
        return None, UNREADABLE
    # Both bars were tried on every frame, so the reason is whichever one nothing got past: a file
    # with frames that are exposed and still score nothing is blurred, one with no exposed frame at
    # all is too dark or too blown out to say anything about focus.
    return best, BLUR if exposed else EXPOSURE
