"""Кадр сам по себе: резкость, экспозиция и лучший момент файла — `montage/video/frames.py`.

Ни базы, ни бакета, ни ffmpeg: крошечные серые кадры собраны здесь руками, ровно того размера,
какой отдаёт ffmpeg. См. `docs/jobs/montage.md`.
"""

import random

import pytest

from montage.video import frames, selection

FRAME_SIZE = frames.SAMPLE_WIDTH * frames.SAMPLE_HEIGHT


def grey(*values):
    """`FRAME_SIZE` grey bytes, `values` repeated to fill it — one value is a flat frame, two
    alternated is a checkerboard."""
    return bytes(values[i % len(values)] for i in range(FRAME_SIZE))


def test_a_flat_frame_has_no_sharpness():
    """No edges anywhere: the Laplacian is zero at every pixel, so its variance is too — this is
    what a smeared, out-of-focus frame looks like to `_sharpness`."""
    assert frames._sharpness(grey(128)) == 0.0


def test_a_checkerboard_is_sharp():
    """Every neighbouring pixel is as different as it gets — the opposite end of the scale from a
    flat frame, and well past `MIN_SHARPNESS`."""
    assert frames._sharpness(grey(0, 255)) > frames.MIN_SHARPNESS


def laplacian_variance(frame):
    """The same number `_sharpness` promises, written out the long obvious way and owing nothing to
    how `_sharpness` happens to compute it: the Laplacian at every pixel that has four neighbours,
    then the variance of those values over the whole population."""
    width, height = frames.SAMPLE_WIDTH, frames.SAMPLE_HEIGHT

    def at(x, y):
        return frame[y * width + x]

    values = [
        4 * at(x, y) - at(x - 1, y) - at(x + 1, y) - at(x, y - 1) - at(x, y + 1)
        for y in range(1, height - 1)
        for x in range(1, width - 1)
    ]
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


@pytest.mark.parametrize(
    "frame",
    [
        grey(128),  # flat
        grey(0, 255),  # checkerboard
        grey(3, 200, 17, 250, 99),  # a period that does not divide the row: no two rows alike
        random.Random(59).randbytes(FRAME_SIZE),  # noise, the same noise every run
    ],
    ids=["flat", "checkerboard", "irregular", "noise"],
)
def test_sharpness_is_the_variance_of_the_laplacian(frame):
    """«A flat frame scores nothing, a checkerboard scores a lot» is true of half a dozen different
    formulas, so those two alone would let the number drift while the tests stayed green — and the
    number is what decides which frames end up in the video. This one pins it: the value itself,
    against the definition spelled out here rather than against `_sharpness`'s own arithmetic."""
    assert frames._sharpness(frame) == pytest.approx(laplacian_variance(frame), rel=1e-12)


def test_a_mid_grey_varied_frame_is_exposed():
    assert frames._exposed(grey(100, 150)) is True


def test_a_flat_dark_frame_is_not_exposed():
    assert frames._exposed(grey(5)) is False


def test_a_flat_bright_frame_is_not_exposed():
    assert frames._exposed(grey(250)) is False


def test_a_frame_split_between_black_and_blown_is_not_exposed():
    """Mean brightness alone would wave this one through — half black, half blown averages to a mid
    grey. The clipped-pixel count is what catches a frame that is really crushed shadow next to
    blown highlight and nothing usable in between."""
    assert frames._exposed(grey(0, 255)) is False


def test_the_best_frame_of_a_good_file_is_its_sharpest_exposed_one():
    looked = [grey(100, 150), grey(100, 110), grey(100, 200)]

    moment, why = frames.best_moment(looked)

    assert why is None
    assert moment[0] == 2.0, "the third second, sampled one frame a second"


def test_a_file_of_smeared_frames_is_rejected_for_blur():
    """Exposed well enough, nothing in focus anywhere in it."""
    assert frames.best_moment([grey(128), grey(128, 129)])[1] == selection.BLUR


def test_a_file_that_is_all_dark_is_rejected_for_exposure():
    """A different reason from blur, and the one the organizer is shown: nothing in these frames is
    lit well enough for «is it sharp» to even be a question."""
    assert frames.best_moment([grey(5), grey(3, 7)])[1] == selection.EXPOSURE


def test_a_file_with_no_frames_at_all_is_unreadable():
    assert frames.best_moment([]) == (None, selection.UNREADABLE)


def test_a_rejected_file_still_says_where_its_best_frame_is():
    """Because the rejection is only a proposal: «включить» overrules it, and the cut still has to
    land somewhere — on the most detailed frame in the file rather than on its first."""
    moment, why = frames.best_moment([grey(5), grey(3, 7), grey(5)])

    assert why == selection.EXPOSURE
    assert moment[0] == 1.0, "the one frame in it with anything in it at all"
