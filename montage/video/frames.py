import collections
import logging

import numpy

from montage import ffmpeg

SAMPLE_WIDTH, SAMPLE_HEIGHT = 96, 54
SAMPLE_FPS = 1.0
MIN_SHARPNESS = 5.0
LUMA_RANGE = (24, 232)
CLIPPED_LIMIT = 0.5

SAME_PICTURE = 0.5
LAYOUT_BLOCK = 6

UNREADABLE = "unreadable"
BLUR = "blur"
EXPOSURE = "exposure"

log = logging.getLogger("montage.video.frames")


def sampled(path, per_second=None):
    sampling = f"fps={per_second}," if per_second else ""
    framing = () if per_second else ("-frames:v", "1")
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
    pixels = numpy.frombuffer(frame, dtype=numpy.uint8).reshape(SAMPLE_HEIGHT, SAMPLE_WIDTH)
    pixels = pixels.astype(numpy.int16)
    laplacian = (
        4 * pixels[1:-1, 1:-1]
        - pixels[1:-1, :-2]
        - pixels[1:-1, 2:]
        - pixels[:-2, 1:-1]
        - pixels[2:, 1:-1]
    )
    return float(laplacian.var())


def _exposed(frame):
    histogram = collections.Counter(frame)
    pixels = len(frame)
    mean = sum(value * count for value, count in histogram.items()) / pixels
    clipped = sum(count for value, count in histogram.items() if value < 16 or value > 239) / pixels
    return LUMA_RANGE[0] <= mean <= LUMA_RANGE[1] and clipped <= CLIPPED_LIMIT


def alike(frame, other):
    if not frame or not other:
        return False
    return float(numpy.abs(_layout(frame) - _layout(other)).mean()) < SAME_PICTURE


def _layout(frame):
    blocks = (
        numpy.frombuffer(frame, dtype=numpy.uint8)
        .reshape(
            SAMPLE_HEIGHT // LAYOUT_BLOCK, LAYOUT_BLOCK, SAMPLE_WIDTH // LAYOUT_BLOCK, LAYOUT_BLOCK
        )
        .mean(axis=(1, 3))
    )
    return (blocks - blocks.mean()) / max(blocks.std(), 1.0)


def best_moment(frames):
    best = usable = None
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
    return best, BLUR if exposed else EXPOSURE
