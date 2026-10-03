"""Отбор on its own — see README.md.

Needs nothing at all: no database, no bucket, no ffmpeg. `choose` is a function over scored pieces
and the frame judging is arithmetic over bytes, so what either of them does with a burst of shots, a
length it cannot fit, a frame the algorithm disliked or a frame the organizer struck out is checked
here without a frame being decoded. `judge` itself, which puts ffmpeg between the two, is checked by
the live montages in `test_montage.py`.

**Отсева внутри сборки нет** (doc-03, «What it doesn't»): кадры, оставленные организатором при
выборе, равны, и «обязательно включить» больше нет. Что осталось от прежних правил — «исключить»,
длина ролика и единственный отказ, который нельзя переиграть: в файле нет ни кадра, ни звука.
Дедуп серии уехал в предложение на экране заказа (`media/screening.py`) и проверяется здесь
отдельно — он больше не часть выбора в ролик.
"""

import dataclasses
import math

import pytest

from montage.video import assembly, selection

from pieces import at, clip, picture

# --- choosing ---------------------------------------------------------------


def cut(pieces, target_seconds, **marks):
    """Only the pieces that made the video — most of these have nothing to say about the rest."""
    return selection.choose(pieces, target_seconds, **marks)[0]


def test_a_burst_of_shots_stays_three_equal_pieces():
    """Три снимка одного кадра подряд — и ни один из них не выпадает из выбора за повтор: кадры,
    оставленные при выборе, равны, и решает режиссёр (doc-03, «What it doesn't»)."""
    burst = [at(0, 10, frame=picture(1)), at(1, 90, frame=picture(1)), at(2, 30, frame=picture(1))]
    apart = at(60, 20, frame=picture(1))

    chosen, rejected = selection.choose([*burst, apart], target_seconds=60)

    assert chosen == [*burst, apart], "все четыре, в порядке съёмки"
    assert rejected == []


def test_a_frame_the_algorithm_disliked_is_offered_like_any_other():
    """Отбор считал кадр браком — тёмным, смазанным. В ролике он стоит наравне с остальными, и
    «включить» больше не нужно, чтобы он туда попал: решение человека уже было — он его оставил."""
    dark = dataclasses.replace(at(0, 90, media_id="dark"), rejected=selection.EXPOSURE)
    blurry = dataclasses.replace(at(30, 10, media_id="blurry"), rejected=selection.BLUR)

    chosen, rejected = selection.choose([dark, blurry], target_seconds=60)

    assert [piece.media_id for piece in chosen] == ["dark", "blurry"]
    assert rejected == []


def test_only_what_fits_the_asked_for_length_gets_in_and_it_gets_in_in_order():
    pieces = [at(second * 10, sharpness) for second, sharpness in enumerate([10, 90, 20, 80, 30])]

    chosen, rejected = selection.choose(pieces, target_seconds=7)

    # Two and a half seconds each, half a second eaten by every crossfade: 2.5, 4.5, 6.5 — a fourth
    # would be 8.5, further past seven than 6.5 is short of it.
    assert assembly.length_of(chosen) == 6.5
    assert chosen == [pieces[1], pieces[3], pieces[4]], "the sharpest three, back in date order"
    assert [piece.rejected for piece in rejected] == [selection.LEFT_OUT] * 2


def test_the_first_piece_is_taken_however_long_it_is():
    """A montage shorter than what was asked for is still a montage; an empty one is not a file."""
    assert cut([at(0, 10, dated=False)], target_seconds=1) == [at(0, 10, dated=False)]


def test_a_file_with_no_frame_in_it_stays_out_with_its_reason():
    """Единственный отказ, который ничем не переигрывается: файла, в котором нет ни кадра, ни
    звука, нет вовсе. Это отсутствие, а не мнение о картинке."""
    broken = dataclasses.replace(at(0, 0, media_id="mine"), rejected=selection.UNREADABLE)

    chosen, rejected = selection.choose([broken, at(30, 10, media_id="a")], target_seconds=60)

    assert [piece.media_id for piece in chosen] == ["a"]
    assert [(piece.media_id, piece.rejected) for piece in rejected] == [
        ("mine", selection.UNREADABLE)
    ]


# --- what the organizer said ------------------------------------------------
# «исключить» — единственное слово заказа, которое до сих пор решает (doc-03, «What it doesn't»).


def test_an_excluded_shot_is_not_in_the_video_however_sharp_it_is():
    best = at(0, 900, media_id="no")

    chosen, rejected = selection.choose(
        [best, at(30, 10, media_id="a"), at(60, 20, media_id="b")],
        target_seconds=60,
        exclude=["no"],
    )

    assert [piece.media_id for piece in chosen] == ["a", "b"]
    assert [(piece.media_id, piece.rejected) for piece in rejected] == [("no", selection.EXCLUDED)]


def test_ids_from_the_order_are_matched_as_text():
    """The list comes out of the order's json as strings and the rows it names come out of
    Postgres as uuids. Nothing in between converts them, so this is where they would quietly stop
    matching and the organizer's word would be ignored without a word in the log."""

    class Uuid:  # anything that is not a str but says the same thing when asked
        def __str__(self):
            return "mine"

    chosen = cut([at(0, 90, media_id="mine"), at(30, 10)], target_seconds=60, exclude=[Uuid()])

    assert [piece.media_id for piece in chosen] == [""]


# --- the series of one moment (the proposal on the order screen) ------------
# Дедуп больше не сворачивает кадры в ролике — он предлагает убрать их на экране заказа, где
# решает человек (`media/screening.py`). Само правило серий — то же.


def test_a_burst_of_shots_is_one_moment_whose_sharpest_stands_for_it():
    # Three shots of the same picture inside two seconds — one moment.
    burst = [at(0, 10, frame=picture(1)), at(1, 90, frame=picture(1)), at(2, 30, frame=picture(1))]
    apart = at(60, 20, frame=picture(1))

    kept, retakes = selection._dedup([*burst, apart])

    # The sharpest of the burst stands for all three — and carries how many it stands for, which
    # is what the gallery tells the organizer as «взял 1 из 3».
    assert kept == [dataclasses.replace(burst[1], series_size=3), apart]
    # And the two it stands for come back saying so, rather than quietly disappearing: this is what
    # the organizer is shown before an order, and what they can put back.
    assert [piece.rejected for piece in retakes] == [selection.DUPLICATE] * 2


def test_a_retake_seconds_later_is_the_same_moment():
    """The shot taken again ten seconds on, to catch the smile: the camera an inch aside and the
    exposure the phone picked this time, and still the same picture — one moment, which a stopwatch
    alone called two."""
    first = at(0, 10, media_id="first", frame=picture(1))
    retake = at(10, 90, media_id="retake", frame=picture(1, moved=(4, 2), light=0.8))

    kept, retakes = selection._dedup([first, retake])

    assert [(piece.media_id, piece.series_size) for piece in kept] == [("retake", 2)]
    assert [(piece.media_id, piece.rejected) for piece in retakes] == [
        ("first", selection.DUPLICATE)
    ]


def test_two_different_pictures_a_second_apart_are_two_moments():
    """The cake, then the guests a second later: taken as close as a burst, and nothing alike."""
    cake = at(0, 10, media_id="cake", frame=picture(1))
    guests = at(1, 90, media_id="guests", frame=picture(2))

    assert selection._dedup([cake, guests])[0] == [cake, guests]


def test_a_retake_is_one_moment_with_other_shots_taken_in_between():
    """A shot of the cake, one of the guests, and the cake again: the guests are a moment of their
    own, and they do not split the cake's."""
    cake = at(0, 10, media_id="cake", frame=picture(1))
    guests = at(4, 50, media_id="guests", frame=picture(2))
    cake_again = at(8, 90, media_id="cake again", frame=picture(1, moved=(-3, 1)))

    kept, retakes = selection._dedup([cake, guests, cake_again])

    assert sorted(piece.media_id for piece in kept) == ["cake again", "guests"]
    assert [(piece.media_id, piece.rejected) for piece in retakes] == [
        ("cake", selection.DUPLICATE)
    ]


def test_a_long_sequence_stays_several_moments():
    """Measured from the first of a series and not from the shot before it: a table photographed
    every ten seconds through the evening is not one moment, however alike it looks."""
    every_ten = [at(10 * number, 10 + number, frame=picture(1)) for number in range(8)]

    kept = selection._dedup(every_ten)[0]

    # 0–30 seconds is one moment and 40–70 the next, so the sharpest of each survives — each
    # standing for the four of its own moment.
    assert kept == [
        dataclasses.replace(every_ten[3], series_size=4),
        dataclasses.replace(every_ten[7], series_size=4),
    ]


def test_a_file_with_no_capture_date_is_never_deduped():
    """Nothing says those two were taken together, and folding one into the other would drop
    material over a missing EXIF field — however alike they look."""
    undated = [at(0, 10, dated=False, frame=picture(1)), at(1, 90, dated=False, frame=picture(1))]

    assert selection._dedup(undated)[0] == undated


# --- the length that was ordered ---------------------------------------------
# What `choose` picks is usually shorter than the order, and past it only by a still that nothing
# else could make up for; `stretch` is what turns the ordered length from a wish into a promise.
# Both are arithmetic over pieces, so a ten-minute montage is checked here in milliseconds instead
# of being encoded — that a real video comes out that long is `test_montage.py`'s.


@pytest.mark.parametrize("target_seconds", [60, 300, 600])
def test_a_video_ordered_out_of_material_to_spare_comes_out_the_length_it_was_ordered(
    target_seconds,
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


@pytest.mark.parametrize(
    "material",
    [
        [clip(60 * n, 50 + n, duration=4.0, media_id=f"v{n}") for n in range(4)],
        [clip(0, 10, duration=4.0), clip(60, 20, duration=4.0), at(120, 90)],
        [clip(60 * n, 50 + n, duration=3.4, media_id=f"v{n}") for n in range(250)],
        [
            *(clip(60 * n, 50 + n, duration=10.0, media_id=f"v{n}") for n in range(100)),
            *(at(60 * n + 30, 40, media_id=f"p{n}") for n in range(100)),
        ],
    ],
    ids=[
        "четыре целых клипа: 14",
        "два целых клипа и фото: 9",
        "250 клипов по 3,4 с: 600",
        "100 клипов по 10 с и 100 фото: 600",
    ],
)
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
    material = [
        clip(0, 90, duration=3.3, media_id="v"),
        *(at(60 * minute, 50 - minute, media_id=f"p{minute}") for minute in (1, 2, 3)),
    ]
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
    material = [
        clip(0, 90, media_id="v0"),
        clip(60, 80, media_id="v1"),
        at(120, 70, media_id="p0"),
        at(180, 60, media_id="p1"),
    ]

    chosen, rejected = selection.choose(material, 11)
    video = selection.stretch(chosen, 11)

    assert [piece.media_id for piece in video] == ["v0", "v1", "p0"]
    assert [(piece.media_id, piece.rejected) for piece in rejected] == [("p1", selection.LEFT_OUT)]
    assert assembly.length_of(video) == 11.0


def test_an_order_for_everything_there_is_takes_every_piece_whole():
    """The longest video the app offers, and it means exactly what it says: every clip runs from its
    first frame to its last and every photograph is held its 2.5 seconds. The joins are counted into
    that number — a join is half a second of two pieces at once — so the finished file runs it to
    the frame."""
    videos = [
        clip(minute * 60, 10, duration=length, media_id=f"v{minute}")
        for minute, length in enumerate((12.0, 30.0, 7.0))
    ]
    photos = [at(minute * 60 + 30, 10, media_id=f"p{minute}") for minute in range(3)]
    everything = ceiling(videos + photos)

    chosen, rejected = selection.choose(videos + photos, everything)
    video = selection.stretch(chosen, everything)

    assert rejected == [], "nothing was left out of an order for everything"
    assert assembly.length_of(video) == everything == 54
    assert {piece.media_id: piece.seconds for piece in video} == {
        "v0": 12.0,
        "v1": 30.0,
        "v2": 7.0,
        "p0": 2.5,
        "p1": 2.5,
        "p2": 2.5,
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
    material = [
        clip(0, 10, duration=6.0, media_id="v"),
        at(60, 9, media_id="p0"),
        at(120, 8, media_id="p1"),
    ]

    chosen, _ = selection.choose(material, 600)
    video = selection.stretch(chosen, 600)

    assert [piece.media_id for piece in video] == ["v", "p0", "p1"]
    assert assembly.length_of(video) == 10.0, "the whole clip and both stills, less two joins"
    assert "590.0s short of the 600s ordered" in caplog.text


def test_nothing_that_was_left_out_comes_back_to_fill_the_length():
    """The length is kept by holding what is in the video longer, never by putting back what was
    left out of it: a frame the organizer struck out is not worth more because there is time going
    spare. The same step fills the length for the director's film and the отбор's alike
    (`montage.py` calls it after both), so this is the answer on both paths."""
    struck = at(0, 900, media_id="no")
    material = [
        clip(60, 10, duration=8.0, media_id="v"),
        at(120, 10, media_id="p0"),
        at(180, 10, media_id="p1"),
    ]

    chosen, rejected = selection.choose([struck, *material], 600, exclude=["no"])
    video = selection.stretch(chosen, 600)

    assert [piece.media_id for piece in video] == ["v", "p0", "p1"]
    assert {piece.media_id: piece.rejected for piece in rejected} == {"no": selection.EXCLUDED}


# --- ten minutes, the edge of any video --------------------------------------
# Ничто в заказе больше не выводит ролик за потолок: кадры равны, «включить» больше нет. Вывести
# может биты дорожки — `cap` держит предел один раз, после них (`montage.py`). Что биты держат
# десятиминутный заказ на нём, — дело `test_rhythm.py`, а что файл выходит не длиннее, —
# `test_montage.py`.


def test_nothing_in_the_order_carries_a_video_past_ten_minutes():
    """Четыреста фотографий заказа в 600 секунд: отбор набирает ровно до потолка, и `cap` не имеет
    что срезать — видео и так не длиннее десяти минут."""
    material = [at(number * 10, 50, media_id=f"p{number}") for number in range(400)]

    chosen, _ = selection.choose(material, 600)
    video = selection.cap(chosen)

    assert assembly.length_of(chosen) == selection.MAX_SECONDS
    assert video == chosen, "никто не вывел ролик за предел — резать нечего"


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
