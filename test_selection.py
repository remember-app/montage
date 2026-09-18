"""Отсев on its own — see README.md.

Needs nothing at all: no database, no bucket, no ffmpeg. `choose` is a function over scored pieces
and the frame judging is arithmetic over bytes, so what either of them does with a burst of shots, a
length it cannot fit, a frame of pure blur or an organizer who overrules it is checked here without
a frame being decoded. `judge` itself, which puts ffmpeg between the two, is checked by the live
montages in `test_montage.py`.
"""

import dataclasses
import datetime
import random

import pytest

import assembly
import selection


# --- choosing ---------------------------------------------------------------


def at(second, sharpness, *, dated=True, media_id=""):
    """A piece shot `second` seconds past midnight, as sharp as it is told to be."""
    when = datetime.datetime(2026, 9, 5) + datetime.timedelta(seconds=second)
    return selection.Piece(
        media_id=media_id, sharpness=sharpness, when=when, captured_at=when if dated else None
    )


def cut(pieces, target_seconds, **marks):
    """Only the pieces that made the video — most of these have nothing to say about the rest."""
    return selection.choose(pieces, target_seconds, **marks)[0]


def test_a_burst_of_shots_leaves_only_its_sharpest():
    burst = [at(0, 10), at(1, 90), at(2, 30)]  # three shots inside two seconds — one moment
    apart = at(30, 20)

    chosen, rejected = selection.choose([*burst, apart], target_seconds=60)

    assert chosen == [burst[1], apart], "the sharpest of the burst stands for all three"
    # And the two it stands for come back saying so, rather than quietly disappearing: this is what
    # the organizer is shown before an order, and what they can put back.
    assert [piece.rejected for piece in rejected] == [selection.DUPLICATE] * 2


def test_a_long_sequence_stays_several_moments():
    """Measured from the first of a burst and not from the shot before it: a whole evening of
    photographs a second apart is not one moment."""
    every_second = [at(second, 10 + second) for second in range(6)]

    chosen = cut(every_second, target_seconds=60)

    # 0-1-2 is one moment and 3-4-5 the next, so the sharpest of each survives.
    assert chosen == [every_second[2], every_second[5]]


def test_a_file_with_no_capture_date_is_never_deduped():
    """Nothing says those two were taken together, and dropping one would drop material over a
    missing EXIF field."""
    undated = [at(0, 10, dated=False), at(1, 90, dated=False)]

    assert cut(undated, target_seconds=60) == undated


def test_only_what_fits_the_asked_for_length_gets_in_and_it_gets_in_in_order():
    pieces = [at(second * 10, sharpness) for second, sharpness in enumerate([10, 90, 20, 80, 30])]

    chosen, rejected = selection.choose(pieces, target_seconds=8)

    # Three seconds each, half a second eaten by every crossfade: 3, 5.5, 8 — a fourth would be 10.5.
    assert assembly.length_of(chosen) == 8
    assert chosen == [pieces[1], pieces[3], pieces[4]], "the sharpest three, back in date order"
    assert [piece.rejected for piece in rejected] == [selection.NO_ROOM] * 2


def test_the_first_piece_is_taken_however_long_it_is():
    """A montage shorter than what was asked for is still a montage; an empty one is not a file."""
    assert cut([at(0, 10, dated=False)], target_seconds=1) == [at(0, 10, dated=False)]


def test_a_file_the_algorithm_threw_away_stays_thrown_away_with_its_reason():
    """`judge` has already had its say on every piece by the time `choose` sees it, and a piece it
    rejected is not silently dropped on the floor here — it comes back out the other side."""
    dark = dataclasses.replace(at(0, 0, dated=False), rejected=selection.EXPOSURE)

    chosen, rejected = selection.choose([dark, at(30, 10), at(60, 20)], target_seconds=60)

    assert dark not in chosen
    assert [piece.rejected for piece in rejected] == [selection.EXPOSURE]


# --- what the organizer said ------------------------------------------------
# «включить» and «исключить», marked before the order and beating the algorithm both ways.


def test_an_excluded_shot_is_not_in_the_video_however_sharp_it_is():
    best = at(0, 900, media_id="no")

    chosen, rejected = selection.choose(
        [best, at(30, 10, media_id="a"), at(60, 20, media_id="b")],
        target_seconds=60, exclude=["no"],
    )

    assert [piece.media_id for piece in chosen] == ["a", "b"]
    assert [(piece.media_id, piece.rejected) for piece in rejected] == [("no", selection.EXCLUDED)]


def test_a_shot_the_organizer_asked_for_goes_in_however_bad_the_frames_are():
    """The algorithm called it too dark to use. The organizer looked at it and said otherwise, and
    the organizer is the one who was there."""
    dark = dataclasses.replace(at(0, 1, media_id="mine"), rejected=selection.EXPOSURE)

    chosen, rejected = selection.choose(
        [dark, at(30, 10, media_id="a")], target_seconds=60, include=["mine"]
    )

    assert [piece.media_id for piece in chosen] == ["mine", "a"]
    assert chosen[0].rejected is None, "it is in the video, so nothing about it is rejected"
    assert rejected == []


def test_a_file_with_no_frame_in_it_stays_out_even_when_the_organizer_asked_for_it():
    """The one judgement «включить» cannot overrule. Everything else is an opinion about the
    picture; this one is that there is no picture to have an opinion about."""
    broken = dataclasses.replace(at(0, 0, media_id="mine"), rejected=selection.UNREADABLE)

    chosen, rejected = selection.choose(
        [broken, at(30, 10, media_id="a")], target_seconds=60, include=["mine"]
    )

    assert [piece.media_id for piece in chosen] == ["a"]
    assert [(piece.media_id, piece.rejected) for piece in rejected] == [
        ("mine", selection.UNREADABLE)
    ]


def test_told_both_at_once_the_shot_stays_out():
    """An order that contradicts itself. Left out is the answer that can still be undone by
    ordering again; a frame in a finished video cannot be taken back out of it."""
    piece = at(0, 90, media_id="both")

    chosen, rejected = selection.choose(
        [piece, at(30, 10), at(60, 20)], target_seconds=60, include=["both"], exclude=["both"]
    )

    assert piece not in chosen
    assert rejected[0].rejected == selection.EXCLUDED


def test_what_the_organizer_asked_for_is_paid_for_out_of_everyone_elses_time():
    """«Включить» is not weighed against the length — it is spent first, and the algorithm fills
    what is left. Here the dullest shot of the five is in the video and two sharper ones are not."""
    pieces = [at(second * 10, 10 * (second + 1), media_id=f"m{second}") for second in range(5)]

    chosen, rejected = selection.choose(pieces, target_seconds=8, include=["m0"])

    # Three seconds each and half a second off every join: m0 asked for, then the two sharpest that
    # still fit — 3, 5.5, 8 — and the fourth would be 10.5.
    assert [piece.media_id for piece in chosen] == ["m0", "m3", "m4"]
    assert assembly.length_of(chosen) == 8
    assert {piece.media_id: piece.rejected for piece in rejected} == {
        "m1": selection.NO_ROOM, "m2": selection.NO_ROOM
    }


def test_the_shot_the_organizer_asked_for_stands_for_its_own_burst():
    """Three shots of one moment, and the organizer picked the one they wanted. The other two are
    still that same moment, so they go — the burst does not get to be in the video twice."""
    burst = [at(0, 10, media_id="a"), at(1, 90, media_id="b"), at(2, 30, media_id="c")]

    chosen, rejected = selection.choose(burst, target_seconds=60, include=["a"])

    assert [piece.media_id for piece in chosen] == ["a"], "not the sharpest — the asked-for one"
    assert {piece.media_id: piece.rejected for piece in rejected} == {
        "b": selection.DUPLICATE, "c": selection.DUPLICATE
    }


def test_ids_from_the_order_are_matched_as_text():
    """The two lists come out of the order's json as strings and the rows they name come out of
    Postgres as uuids. Nothing in between converts them, so this is where they would quietly stop
    matching and the organizer's word would be ignored without a word in the log."""
    class Uuid:  # anything that is not a str but says the same thing when asked
        def __str__(self):
            return "mine"

    chosen = cut([at(0, 90, media_id="mine"), at(30, 10)], target_seconds=60, exclude=[Uuid()])

    assert [piece.media_id for piece in chosen] == [""]


# --- judging a frame ---------------------------------------------------------
# `choose`'s own tests hand it a sharpness directly and never touch the arithmetic that produces
# one. These do: tiny grey frames built by hand, the same size ffmpeg hands `_sharpness`/`_exposed`
# for real, no ffmpeg needed to check the maths itself is right.

FRAME_SIZE = selection.SAMPLE_WIDTH * selection.SAMPLE_HEIGHT


def grey(*values):
    """`FRAME_SIZE` grey bytes, `values` repeated to fill it — one value is a flat frame, two
    alternated is a checkerboard."""
    return bytes(values[i % len(values)] for i in range(FRAME_SIZE))


def test_a_flat_frame_has_no_sharpness():
    """No edges anywhere: the Laplacian is zero at every pixel, so its variance is too — this is
    what a smeared, out-of-focus frame looks like to `_sharpness`."""
    assert selection._sharpness(grey(128)) == 0.0


def test_a_checkerboard_is_sharp():
    """Every neighbouring pixel is as different as it gets — the opposite end of the scale from a
    flat frame, and well past `MIN_SHARPNESS`."""
    assert selection._sharpness(grey(0, 255)) > selection.MIN_SHARPNESS


def laplacian_variance(frame):
    """The same number `_sharpness` promises, written out the long obvious way and owing nothing to
    how `_sharpness` happens to compute it: the Laplacian at every pixel that has four neighbours,
    then the variance of those values over the whole population."""
    width, height = selection.SAMPLE_WIDTH, selection.SAMPLE_HEIGHT

    def at(x, y):
        return frame[y * width + x]

    values = [
        4 * at(x, y) - at(x - 1, y) - at(x + 1, y) - at(x, y - 1) - at(x, y + 1)
        for y in range(1, height - 1)
        for x in range(1, width - 1)
    ]
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


@pytest.mark.parametrize("frame", [
    grey(128),  # flat
    grey(0, 255),  # checkerboard
    grey(3, 200, 17, 250, 99),  # a period that does not divide the row: no two rows alike
    random.Random(59).randbytes(FRAME_SIZE),  # noise, the same noise every run
], ids=["flat", "checkerboard", "irregular", "noise"])
def test_sharpness_is_the_variance_of_the_laplacian(frame):
    """«A flat frame scores nothing, a checkerboard scores a lot» is true of half a dozen different
    formulas, so those two alone would let the number drift while the tests stayed green — and the
    number is what decides which frames end up in the video. This one pins it: the value itself,
    against the definition spelled out here rather than against `_sharpness`'s own arithmetic."""
    assert selection._sharpness(frame) == pytest.approx(laplacian_variance(frame), rel=1e-12)


def test_a_mid_grey_varied_frame_is_exposed():
    assert selection._exposed(grey(100, 150)) is True


def test_a_flat_dark_frame_is_not_exposed():
    assert selection._exposed(grey(5)) is False


def test_a_flat_bright_frame_is_not_exposed():
    assert selection._exposed(grey(250)) is False


def test_a_frame_split_between_black_and_blown_is_not_exposed():
    """Mean brightness alone would wave this one through — half black, half blown averages to a mid
    grey. The clipped-pixel count is what catches a frame that is really crushed shadow next to
    blown highlight and nothing usable in between."""
    assert selection._exposed(grey(0, 255)) is False


# --- and what the file as a whole is rejected for ---------------------------


def test_the_best_frame_of_a_good_file_is_its_sharpest_exposed_one():
    frames = [grey(100, 150), grey(100, 110), grey(100, 200)]

    moment, why = selection._best_moment(frames)

    assert why is None
    assert moment[0] == 2.0, "the third second, sampled one frame a second"


def test_a_file_of_smeared_frames_is_rejected_for_blur():
    """Exposed well enough, nothing in focus anywhere in it."""
    assert selection._best_moment([grey(128), grey(128, 129)])[1] == selection.BLUR


def test_a_file_that_is_all_dark_is_rejected_for_exposure():
    """A different reason from blur, and the one the organizer is shown: nothing in these frames is
    lit well enough for «is it sharp» to even be a question."""
    assert selection._best_moment([grey(5), grey(3, 7)])[1] == selection.EXPOSURE


def test_a_file_with_no_frames_at_all_is_unreadable():
    assert selection._best_moment([]) == (None, selection.UNREADABLE)


def test_a_rejected_file_still_says_where_its_best_frame_is():
    """Because the rejection is only a proposal: «включить» overrules it, and the cut still has to
    land somewhere — on the most detailed frame in the file rather than on its first."""
    moment, why = selection._best_moment([grey(5), grey(3, 7), grey(5)])

    assert why == selection.EXPOSURE
    assert moment[0] == 1.0, "the one frame in it with anything in it at all"
