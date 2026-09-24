"""Отсев on its own — see README.md.

Needs nothing at all: no database, no bucket, no ffmpeg. `choose` is a function over scored pieces
and the frame judging is arithmetic over bytes, so what either of them does with a burst of shots, a
length it cannot fit, a frame of pure blur or an organizer who overrules it is checked here without
a frame being decoded. `judge` itself, which puts ffmpeg between the two, is checked by the live
montages in `test_montage.py`.
"""

import dataclasses
import datetime
import math
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

    # The sharpest of the burst stands for all three — and carries how many it stands for, which
    # is what the journal of the montage tells the organizer as «взял 1 из 3».
    assert chosen == [dataclasses.replace(burst[1], series_size=3), apart]
    # And the two it stands for come back saying so, rather than quietly disappearing: this is what
    # the organizer is shown before an order, and what they can put back.
    assert [piece.rejected for piece in rejected] == [selection.DUPLICATE] * 2


def test_a_long_sequence_stays_several_moments():
    """Measured from the first of a burst and not from the shot before it: a whole evening of
    photographs a second apart is not one moment."""
    every_second = [at(second, 10 + second) for second in range(6)]

    chosen = cut(every_second, target_seconds=60)

    # 0-1-2 is one moment and 3-4-5 the next, so the sharpest of each survives — each standing for
    # the three of its own moment.
    assert chosen == [dataclasses.replace(every_second[2], series_size=3),
                      dataclasses.replace(every_second[5], series_size=3)]


def test_a_file_with_no_capture_date_is_never_deduped():
    """Nothing says those two were taken together, and dropping one would drop material over a
    missing EXIF field."""
    undated = [at(0, 10, dated=False), at(1, 90, dated=False)]

    assert cut(undated, target_seconds=60) == undated


def test_only_what_fits_the_asked_for_length_gets_in_and_it_gets_in_in_order():
    pieces = [at(second * 10, sharpness) for second, sharpness in enumerate([10, 90, 20, 80, 30])]

    chosen, rejected = selection.choose(pieces, target_seconds=7)

    # Two and a half seconds each, half a second eaten by every crossfade: 2.5, 4.5, 6.5 — a fourth
    # would be 8.5, further past seven than 6.5 is short of it.
    assert assembly.length_of(chosen) == 6.5
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

    chosen, rejected = selection.choose(pieces, target_seconds=7, include=["m0"])

    # Two and a half seconds each and half a second off every join: m0 asked for, then the two
    # sharpest that still fit — 2.5, 4.5, 6.5 — and the fourth would be 8.5.
    assert [piece.media_id for piece in chosen] == ["m0", "m3", "m4"]
    assert assembly.length_of(chosen) == 6.5
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


# --- the length that was ordered ---------------------------------------------
# What `choose` picks is usually shorter than the order, and past it only by a still that nothing
# else could make up for; `stretch` is what turns the ordered length from a wish into a promise.
# Both are arithmetic over pieces, so a ten-minute montage is checked here in milliseconds instead
# of being encoded — that a real video comes out that long is `test_montage.py`'s.


def clip(second, sharpness, *, duration=60.0, media_id=""):
    """A piece as `judge` cuts one out of a video file that runs `duration` seconds: the default
    four seconds around a best moment in the middle of it."""
    return dataclasses.replace(
        at(second, sharpness, media_id=media_id), kind="video", duration=duration,
        best_second=duration / 2, seconds=min(selection.CLIP_SECONDS, duration),
    )


@pytest.mark.parametrize("target_seconds", [60, 300, 600])
def test_a_video_ordered_out_of_material_to_spare_comes_out_the_length_it_was_ordered(
    target_seconds
):
    """The promise itself, at the three lengths the product talks in: a minute, five minutes and
    the ten-minute ceiling. Twenty minute-long clips and ten photographs is material with room to
    spare at all three — so the answer is the length that was asked for, not «as much as the
    default four seconds a clip happened to add up to»."""
    material = [clip(minute * 60, 100 - minute, media_id=f"v{minute}") for minute in range(20)]
    material += [at(minute * 60 + 30, 50 - minute, media_id=f"p{minute}") for minute in range(10)]

    chosen, _ = selection.choose(material, target_seconds)
    video = selection.stretch(chosen, target_seconds)

    # Five per cent of the order, and never less than three seconds: the ordered length is kept to
    # within a join or two, not to the frame.
    assert assembly.length_of(video) == pytest.approx(
        target_seconds, abs=max(target_seconds * 0.05, 3)
    )


def test_a_ten_minute_order_out_of_stills_alone_takes_every_one_it_can_hold_and_ends_on_it():
    """Stills never stretch, so an order they alone must fill is filled by how many of them go in,
    and 2.5 seconds each is the arithmetic the api counts the ceiling by. Four hundred photographs
    of material against ten minutes: 299 fall a second and a half short, and the 300th would run
    half a second past — the edge of any video, so that one is held half a second shorter and the
    video ends on ten minutes exactly."""
    material = [at(number * 10, 50, media_id=f"p{number}") for number in range(400)]

    chosen, rejected = selection.choose(material, 600)
    video = selection.stretch(chosen, 600)

    assert sorted(piece.seconds for piece in video) == [2.0] + [selection.PHOTO_SECONDS] * 299
    assert len(rejected) == 100
    assert assembly.length_of(video) == selection.MAX_SECONDS


def ceiling(material):
    """The longest video the app lets anyone order out of this material — the api's number
    (`materialLeft`, api src/montages/montages.ts): every clip whole and every still its 2.5
    seconds, less half a second for every join, down to the whole second, ten minutes at most."""
    whole = sum(piece.duration or piece.seconds for piece in material)
    return math.floor(min(600, whole - assembly.CROSSFADE_SECONDS * (len(material) - 1)))


@pytest.mark.parametrize("photos", [3, 40, 300])
def test_an_order_for_every_still_there_is_gets_every_still(photos):
    """k photographs make a video of 2k and a half seconds — 2.5 each, half a second of every join
    shared by two — and only whole seconds can be ordered, so the most the app offers is 2k. The
    last still cannot be cut to fit that and nothing else can be held longer to make up for it, so
    it goes in half a second past the order rather than leaving the video a second and a half short
    of it: three photographs ordered at six seconds are a video of 6.5 out of all three, not of 4.5
    out of two. At three hundred the ceiling is ten minutes, which no video runs past: that half
    second comes off the last still instead."""
    material = [at(number * 10, 50 + number, media_id=f"p{number}") for number in range(photos)]
    target = ceiling(material)

    chosen, rejected = selection.choose(material, target)
    video = selection.stretch(chosen, target)

    assert target == 2 * photos
    assert rejected == [] and len(video) == photos
    assert assembly.length_of(video) == min(target + 0.5, selection.MAX_SECONDS)


@pytest.mark.parametrize("material", [
    [clip(60 * n, 50 + n, duration=4.0, media_id=f"v{n}") for n in range(4)],
    [clip(0, 10, duration=4.0), clip(60, 20, duration=4.0), at(120, 90)],
    [clip(60 * n, 50 + n, duration=3.4, media_id=f"v{n}") for n in range(250)],
    [*(clip(60 * n, 50 + n, duration=10.0, media_id=f"v{n}") for n in range(100)),
     *(at(60 * n + 30, 40, media_id=f"p{n}") for n in range(100))],
], ids=["четыре целых клипа: 14", "два целых клипа и фото: 9", "250 клипов по 3,4 с: 600",
        "100 клипов по 10 с и 100 фото: 600"])
def test_an_order_for_the_ceiling_out_of_clips_comes_out_the_ceiling(material):
    """A clip that has run whole cannot be held longer, and one left out adds its whole length or
    nothing — so four whole four-second clips ordered at their ceiling of fourteen seconds were a
    video of eleven. Unlike a still, a clip can be cut: the last one goes in cut to end the video on
    the order, whether that is the length of everything or ten minutes of more than that."""
    target = ceiling(material)

    chosen, _ = selection.choose(material, target)
    video = selection.stretch(chosen, target)

    assert assembly.length_of(video) == pytest.approx(target)


def test_a_clip_held_whole_cannot_make_up_for_the_last_still_either():
    """A clip already running from its first frame to its last has nothing left to be held longer
    by, so against it the last still goes in past the order just as among stills alone."""
    material = [clip(0, 90, duration=3.3, media_id="v"),
                *(at(60 * minute, 50 - minute, media_id=f"p{minute}") for minute in (1, 2, 3))]
    target = ceiling(material)  # 3.3 + 3 × 2.5 less three joins is 9.3, and 9 can be ordered

    chosen, rejected = selection.choose(material, target)
    video = selection.stretch(chosen, target)

    assert rejected == [] and [piece.media_id for piece in video] == ["v", "p1", "p2", "p3"]
    assert assembly.length_of(video) == pytest.approx(9.3)


def test_a_still_does_not_run_past_the_order_while_the_clips_can_make_up_the_rest():
    """Two clips and two stills against eleven seconds: the clips and one still come to 9.5, and
    the other still would make 11.5. Both clips have most of a minute left in their files, which
    brings the video to the eleven seconds exactly — so that still stays out and they are held
    longer instead: running past the order is only for when nothing else can reach it."""
    material = [clip(0, 90, media_id="v0"), clip(60, 80, media_id="v1"),
                at(120, 70, media_id="p0"), at(180, 60, media_id="p1")]

    chosen, rejected = selection.choose(material, 11)
    video = selection.stretch(chosen, 11)

    assert [piece.media_id for piece in video] == ["v0", "v1", "p0"]
    assert [(piece.media_id, piece.rejected) for piece in rejected] == [("p1", selection.NO_ROOM)]
    assert assembly.length_of(video) == 11.0


def test_an_order_for_everything_there_is_takes_every_piece_whole():
    """The longest video the app offers, and it means exactly what it says: every clip runs from its
    first frame to its last and every photograph is held its 2.5 seconds. The joins are counted into
    that number — a join is half a second of two pieces at once — so the finished file runs it to
    the frame."""
    videos = [clip(minute * 60, 10, duration=length, media_id=f"v{minute}")
              for minute, length in enumerate((12.0, 30.0, 7.0))]
    photos = [at(minute * 60 + 30, 10, media_id=f"p{minute}") for minute in range(3)]
    everything = ceiling(videos + photos)

    chosen, rejected = selection.choose(videos + photos, everything)
    video = selection.stretch(chosen, everything)

    assert rejected == [], "nothing was left out of an order for everything"
    assert assembly.length_of(video) == everything == 54
    assert {piece.media_id: piece.seconds for piece in video} == {
        "v0": 12.0, "v1": 30.0, "v2": 7.0, "p0": 2.5, "p1": 2.5, "p2": 2.5
    }
    assert [piece.start for piece in video if piece.kind == "video"] == [0.0] * 3, (
        "a clip held whole starts at its own beginning"
    )


def test_the_missing_seconds_are_shared_out_equally_and_a_photograph_gets_none_of_them():
    """Two clips and a still against an order none of them can fill on its own. The clips take an
    equal share of what is missing until one of them runs out of file, and the seconds it could not
    hold go to the other rather than out of the video. The still is held its 2.5 seconds either
    way: a photograph stretched to fill time is a stopped video, not a longer one."""
    short = clip(0, 10, duration=5.0, media_id="short")  # a second left in it past the default cut
    long_one = clip(60, 10, duration=60.0, media_id="long")
    still = at(120, 10, media_id="photo")

    video = selection.stretch([short, long_one, still], target_seconds=20.0)

    # 4 + 4 + 2.5 less two joins is 9.5, so 10.5 is missing: the short clip can only take one of
    # the 5.25 it was offered, and the other 9.5 all go to the long one.
    assert [piece.seconds for piece in video] == [5.0, 13.5, 2.5]
    assert assembly.length_of(video) == 20.0


def test_material_that_runs_out_makes_a_shorter_video_and_not_a_failure(caplog):
    """Ten minutes ordered out of eleven seconds of material. The video is eleven seconds long —
    no error, and nothing shown twice to make up the difference. Only the worker's log says, with
    the numbers, that the order was not filled."""
    material = [clip(0, 10, duration=6.0, media_id="v"),
                at(60, 9, media_id="p0"), at(120, 8, media_id="p1")]

    chosen, _ = selection.choose(material, 600)
    video = selection.stretch(chosen, 600)

    assert [piece.media_id for piece in video] == ["v", "p0", "p1"]
    assert assembly.length_of(video) == 10.0, "the whole clip and both stills, less two joins"
    assert "590.0s short of the 600s ordered" in caplog.text


def test_nothing_that_was_left_out_comes_back_to_fill_the_length():
    """The length is kept by holding what is in the video longer, never by putting back what was
    left out of it: a frame the organizer struck out is not worth more because there is time going
    spare, and neither is one the отсев turned down. The same step fills the length for the
    director's film and the отбор's alike (`montage.py` calls it after both), so this is the answer
    on both paths."""
    struck = at(0, 900, media_id="no")
    blurred = dataclasses.replace(at(30, 5, media_id="blur"), rejected=selection.BLUR)
    material = [clip(60, 10, duration=8.0, media_id="v"),
                at(120, 10, media_id="p0"), at(180, 10, media_id="p1")]

    chosen, rejected = selection.choose([struck, blurred, *material], 600, exclude=["no"])
    video = selection.stretch(chosen, 600)

    assert [piece.media_id for piece in video] == ["v", "p0", "p1"]
    assert {piece.media_id: piece.rejected for piece in rejected} == {
        "no": selection.EXCLUDED, "blur": selection.BLUR
    }


# --- ten minutes, the edge of any video --------------------------------------
# «Включить», the director and the beats can each carry a video past the ceiling; `cap` is the one
# place it is held, after all of them (`montage.py`). That the beats keep a ten-minute order on it
# is `test_rhythm.py`'s, and that the file itself comes out no longer is `test_montage.py`'s.


def test_everything_the_organizer_asked_for_goes_in_up_to_ten_minutes_and_no_further():
    """Four hundred photographs, every one marked «включить»: spent before anything else and weighed
    against nothing, they come to 800.5 seconds. Ten minutes is the edge of any video all the same —
    the first three hundred stay in their own order, the three-hundredth is held two seconds so the
    video ends on the ceiling, and the hundred after it come off the end."""
    material = [at(number * 10, 50, media_id=f"p{number}") for number in range(400)]

    chosen, _ = selection.choose(material, 600, include=[piece.media_id for piece in material])
    video = selection.cap(chosen)

    assert assembly.length_of(chosen) == 800.5, "«включить» did not take the video past the ceiling"
    assert assembly.length_of(video) == selection.MAX_SECONDS
    assert video[:-1] == chosen[:299]
    assert video[-1] == dataclasses.replace(chosen[299], seconds=2.0)


def test_a_clip_that_crosses_ten_minutes_is_cut_around_its_best_moment_and_what_is_past_it_goes():
    """The piece that crosses the ceiling is held shorter whatever it is — a clip the way `cut_to`
    cuts any clip, around the moment it was chosen for — and a piece that would only begin past the
    ceiling is not in the video at all."""
    stills = [at(number * 10, 50, media_id=f"p{number}") for number in range(299)]  # 598.5 s
    tail = clip(3000, 50, media_id="v")  # four seconds out of the middle of a minute-long file
    late = at(3010, 50, media_id="late")

    video = selection.cap([*stills, tail, late])

    assert [piece.media_id for piece in video] == [*(piece.media_id for piece in stills), "v"]
    assert (video[-1].seconds, video[-1].start) == (2.0, 29.0)
    assert assembly.length_of(video) == selection.MAX_SECONDS


@pytest.mark.parametrize("photos", [3, 300])
def test_a_video_inside_ten_minutes_is_left_exactly_as_it_was(photos):
    """Short of the ceiling, or ending on it exactly — three hundred stills ordered for ten minutes
    (above) — nothing comes off and nothing is held any shorter."""
    material = [at(number * 10, 50, media_id=f"p{number}") for number in range(photos)]
    chosen, _ = selection.choose(material, ceiling(material))
    video = selection.stretch(chosen, ceiling(material))

    assert selection.cap(video) == video


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
