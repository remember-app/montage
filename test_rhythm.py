"""Ритм on its own — see README.md.

Needs `ffmpeg` to draw the audio and nothing else: no database, no bucket. The tempo is known here
because the track is drawn to it — a click every 0.6 seconds is a hundred beats a minute — so both
halves can be checked against a number instead of against a listening ear: that the beats found in
a file are the ones put into it, and that the joins between pieces land on them.

A whole montage assembled to an uploaded track of a known tempo is `test_montage.py`'s.
"""

import bisect
import dataclasses

import pytest

import assembly
import media
import rhythm
import selection
from test_selection import at, clip

GAP = 0.6  # seconds between the clicks of the drawn track: a hundred beats a minute
SECONDS = 30  # and this is how long it runs


@pytest.fixture(scope="module")
def click(tmp_path_factory):
    """A metronome: a short click every 0.6 seconds and silence in between."""
    path = tmp_path_factory.mktemp("rhythm") / "click.wav"
    media._ffmpeg(
        "-f", "lavfi",
        "-i", f"aevalsrc='0.8*sin(2*PI*1200*t)*exp(-40*mod(t,{GAP}))':d={SECONDS}", str(path),
    )
    return path


def photos(*lengths):
    """Pieces cut from photographs, as long as they are told to be — a still may be held at will."""
    return [selection.Piece(media_id=str(index), kind="photo", seconds=length)
            for index, length in enumerate(lengths)]


def joins(pieces):
    """Where each join is heard in the assembled montage: the middle of the crossfade."""
    at, cuts = 0.0, []
    for piece in pieces[:-1]:
        cuts.append(at + piece.seconds - assembly.CROSSFADE_SECONDS / 2)
        at += piece.seconds - assembly.CROSSFADE_SECONDS
    return cuts


# --- what the track says -----------------------------------------------------


def test_the_beats_found_in_a_track_are_the_ones_put_into_it(click):
    moments = rhythm.beats(click)
    inside = [moment for moment in moments if moment < SECONDS]

    assert len(inside) > 40, "thirty seconds of metronome is about fifty beats"
    assert list(moments) == sorted(moments), "in order, the way `fit` reads them"
    # Every beat found is on a click rather than between two. 60 ms of tolerance: the analysis hears
    # an attack a couple of windows late, and for a join that difference is not a difference.
    grid = [min(abs(moment - GAP * beat) for beat in range(int(SECONDS / GAP) + 1))
            for moment in inside]
    assert max(grid) < 0.06


def test_the_beats_go_round_with_a_track_shorter_than_the_montage(click):
    """A track shorter than the montage is played again from the top, and its beats go round with
    it. Without that, the pieces past the end of the track would have nothing to land on — and a
    montage under a one-minute track would come apart exactly where the music went round again."""
    moments = rhythm.beats(click)

    assert moments[-1] > 5 * 60, "the grid is counted well past the end of a thirty-second track"
    # And the second time round is the same metronome, moved on by the length of the track.
    second = [moment for moment in moments if SECONDS <= moment < 2 * SECONDS]
    assert len(second) == len([moment for moment in moments if moment < SECONDS])


def test_a_track_longer_than_the_horizon_is_heard_only_up_to_it(click, monkeypatch):
    """A list of tracks is one soundtrack as long as all of them together, and nothing past the
    horizon ever plays: analysed whole, a few hours of it would take the worker's memory."""
    monkeypatch.setattr(rhythm, "HORIZON_SECONDS", SECONDS / 3)

    assert max(rhythm.beats(click)) < SECONDS / 3


def test_the_tempo_of_a_track_is_the_one_it_was_drawn_to(click):
    """Темп дорожки — то единственное, что из ритма уходит в журнал монтажа: «подогнал под
    «Тёплую», 100 ударов». Считается он по тем же битам, что двигают склейки, поэтому у
    метронома в сто ударов ответ — сто; битов нет — нет и темпа, и журнал о нём молчит.

    Удар-другой в минуту — не разница: биты библиотека кладёт на сетку своих окон анализа, и та
    же сотня выходит из неё то 99, то 101. Число это показывают человеку, а не режут по нему.
    """
    assert abs(rhythm.tempo(rhythm.beats(click)) - round(60 / GAP)) <= 2
    assert rhythm.tempo(()) is None


def test_a_track_with_no_tempo_in_it_gives_no_beats(tmp_path):
    """The fallback: an even drone is not a tempo, and a rhythm must not be invented out of it."""
    drone = tmp_path / "drone.wav"
    media._ffmpeg("-f", "lavfi", "-i", "sine=frequency=220:duration=30", str(drone))

    assert rhythm.beats(drone) == ()


def test_a_file_that_is_not_audio_at_all_gives_no_beats(tmp_path):
    """An uploaded track is somebody else's bytes. They would not open — the montage is cut in even
    pieces rather than lost: there will be music in it either way, ffmpeg puts it there itself."""
    broken = tmp_path / "broken.mp3"
    broken.write_bytes(b"this is not music and never was")

    assert rhythm.beats(broken) == ()


# --- what it does to the pieces ----------------------------------------------


def test_every_join_lands_on_a_beat():
    """What the module exists for: the joins do not cut the track in the middle of a beat."""
    moments = tuple(GAP * beat for beat in range(50))
    pieces = photos(3.0, 3.0, 3.0, 3.0)

    fitted = rhythm.fit(pieces, moments)

    assert [piece.seconds for piece in pieces] == [3.0] * 4, "the pieces handed in are untouched"
    for cut in joins(fitted):
        assert min(abs(cut - moment) for moment in moments) < 1e-6, f"the join at {cut} misses"
    # And the fitting does not run away with the montage: every piece stayed near its own length.
    assert all(2.5 <= piece.seconds <= 3.5 for piece in fitted)


def test_the_last_piece_takes_up_what_the_joins_moved():
    """Nothing is joined to it — behind it are the end of the video and the music fading out — so
    the beats do not set its length: it is held as much longer or shorter as ends the video on the
    length it was handed, which is the length that was ordered."""
    moments = tuple(GAP * beat for beat in range(50))
    pieces = photos(3.0, 3.0, 4.2)

    fitted = rhythm.fit(pieces, moments)

    assert fitted[-1].seconds != 4.2, "the joins moved and the last piece took up none of it"
    assert assembly.length_of(fitted) == pytest.approx(assembly.length_of(pieces))


def test_a_last_clip_with_nothing_left_in_its_file_ends_the_video_short_and_says_so(caplog):
    """The joins land early, and the last piece would have to be held longer to make up for it — but
    it is a clip already running whole. The video comes out that fraction of a beat short, and the
    worker's log says by how much instead of keeping quiet about it."""
    moments = tuple(GAP * beat for beat in range(50))
    whole = dataclasses.replace(photos(3.0)[0], kind="video", duration=3.0)
    pieces = [*photos(2.9, 2.9), whole]

    fitted = rhythm.fit(pieces, moments)

    assert fitted[-1].seconds == 3.0, "a clip was held past its own file"
    assert assembly.length_of(fitted) < assembly.length_of(pieces)
    assert "instead of" in caplog.text, "the video came out short without a word in the log"


def test_a_track_without_a_tempo_leaves_the_pieces_alone():
    pieces = photos(3.0, 3.0, 3.0)

    assert rhythm.fit(pieces, ()) == pieces
    assert rhythm.fit(pieces, (0.6, 1.2, 3.0)) == pieces, "three beats in a montage is not a tempo"


def test_a_clip_is_never_stretched_past_its_own_file():
    """A photograph may be held as long as it needs to be, a clip only while its file has anything
    left to show. Stretched further it would come back from ffmpeg shorter than asked for, and the
    miss would take not only that join but every one after it."""
    # The beat nearest the end of the piece is the one at 3.3 — but reaching it would take 3.55
    # seconds out of a file with three left in it. The one that fits is taken: 2.0, a 2.25s piece.
    moments = (0.0, 2.0, 3.3, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0)
    clip = dataclasses.replace(
        photos(3.0)[0], kind="video", seconds=3.0, start=1.0, duration=4.0
    )

    fitted = rhythm.fit([clip, *photos(3.0, 3.0)], moments)

    assert fitted[0].seconds == pytest.approx(2.25)


def test_a_piece_keeps_its_own_length_when_no_beat_fits():
    """A sparse grid with no beat inside what the piece may be: then it stays what it was. Same as
    under a track with no tempo — an even piece beats a montage skipping along somebody's grid."""
    moments = tuple(30.0 * beat for beat in range(10))

    fitted = rhythm.fit(photos(3.0, 3.0, 3.0), moments)

    assert [piece.seconds for piece in fitted] == [3.0, 3.0, 3.0]


def test_a_clip_shrunk_to_the_beat_stays_cut_around_its_best_moment():
    """`selection.judge` cuts a clip centred on its best moment; a beat that shrinks the piece must
    not slide that moment out of what is left on screen — the two modules agree on where a clip
    starts (`cut_to`) for exactly this reason, and `fit` has to go through it too, not just swap
    the length in and leave the old, now off-centre, `start` behind."""
    moments = (1.5, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0)
    clip = dataclasses.replace(
        photos(4.0)[0], kind="video", seconds=4.0, start=0.0, best_second=2.0, duration=10.0
    )

    fitted = rhythm.fit([clip, *photos(3.0)], moments)

    shrunk = fitted[0]
    assert shrunk.seconds < clip.seconds, "the beat picked here does shrink the piece"
    assert shrunk.start <= clip.best_second <= shrunk.start + shrunk.seconds, (
        "the best moment fell outside the clip once it was shrunk"
    )


def test_a_piece_is_never_cut_shorter_than_a_glance():
    """A dense grid of beats must not turn the montage into a flicker: the nearest beat can be half
    a second from the start of the piece, and that one may not be taken. Nor is the last piece cut
    down past a glance to give back what the joins before it put on."""
    moments = tuple(0.2 * beat for beat in range(100))

    fitted = rhythm.fit(photos(0.3, 0.3, 3.0), moments)

    assert all(piece.seconds >= rhythm.MIN_PIECE_SECONDS for piece in fitted)


def test_a_ten_minute_order_keeps_the_length_it_was_stretched_to_under_a_beat_grid():
    """The ceiling order, past both of `selection`'s steps: twenty minute-long clips and ten stills
    ordered for 600 seconds are stretched to it, and the beats then pull each join to its nearest
    one. The window of a piece (`MAX_PIECE_SECONDS`) bounds what a beat may do to it, not how long
    the stretch made it — a clip held for half a minute is not cut back to six seconds — and the
    pulls do not add up: the video stays within a second of the order."""
    material = [clip(minute * 60, 100 - minute, media_id=f"v{minute}") for minute in range(20)]
    material += [at(minute * 60 + 30, 50 - minute, media_id=f"p{minute}") for minute in range(10)]
    chosen, _ = selection.choose(material, 600)
    stretched = selection.stretch(chosen, 600)
    moments = tuple(GAP * beat for beat in range(1, 1500))

    fitted = rhythm.fit(stretched, moments)

    assert max(piece.seconds for piece in stretched) > rhythm.MAX_PIECE_SECONDS * 3
    for before, after in zip(stretched[:-1], fitted[:-1]):
        assert after.seconds == pytest.approx(before.seconds, abs=GAP), "a stretched piece was cut back"
    assert assembly.length_of(fitted) == pytest.approx(600, abs=1.0)
    for cut in joins(fitted):
        assert min(abs(cut - moment) for moment in moments) < 1e-6, f"the join at {cut} misses"


def test_a_ten_minute_order_under_a_fast_track_still_ends_on_ten_minutes():
    """Three hundred stills ordered for the ceiling come to 600 seconds exactly. Under a hundred and
    forty beats a minute a grid that pulled every join from where the one before it landed would
    pull them all the same way and run the video some forty seconds long; each join is pulled from
    its own place instead, the last still takes up the rest, and the video ends on ten minutes —
    not past them — with every join in it on a beat."""
    material = [at(number * 10, 50, media_id=f"p{number}") for number in range(300)]
    chosen, _ = selection.choose(material, 600)
    gap = 60 / 140
    moments = tuple(gap * beat for beat in range(1, int(rhythm.HORIZON_SECONDS / gap)))

    video = selection.cap(rhythm.fit(selection.stretch(chosen, 600), moments))

    assert len(video) == 300, "a still came off the tail"
    assert assembly.length_of(video) <= selection.MAX_SECONDS
    assert assembly.length_of(video) == pytest.approx(selection.MAX_SECONDS)
    for cut in joins(video):
        assert min(abs(cut - moment) for moment in moments) < 1e-6, f"the join at {cut} misses"


@pytest.fixture(scope="module")
def shelf():
    """The beats of two tracks of the library, heard the way a montage hears them: «Тёплая», the
    default, at about eighty beats a minute, and «Ночь» at about a hundred. Under one of them every
    join pulled the same way would run the video long, under the other short."""
    return {track: rhythm.beats(assembly.track_file(track)) for track in ("warm", "night")}


@pytest.mark.parametrize("target_seconds", [60, 300, 600])
@pytest.mark.parametrize("track", ["warm", "night"])
@pytest.mark.parametrize("material", [
    [at(number * 10, 50, media_id=f"p{number}") for number in range(300)],
    [*(clip(minute * 60, 100 - minute, media_id=f"v{minute}") for minute in range(20)),
     *(at(minute * 60 + 30, 50 - minute, media_id=f"p{minute}") for minute in range(10))],
], ids=["300 фото", "20 клипов по минуте и 10 фото"])
def test_a_video_under_music_comes_out_the_length_it_was_ordered(
    shelf, track, target_seconds, material
):
    """The promise under music, at the three lengths the product talks in: out of material with
    enough in it, the video that the beats have cut is the length that was ordered to within a
    second — under a slow track and a fast one alike, out of stills that cannot be held longer and
    out of clips that can — and no longer than ten minutes. Every join is on a beat of the track,
    except where the track has none for longer than a piece can reach: «Тёплая» fades out and
    starts again with nine seconds of no pulse in between."""
    moments = shelf[track]
    chosen, _ = selection.choose(material, target_seconds)

    video = selection.cap(rhythm.fit(selection.stretch(chosen, target_seconds), moments))

    assert abs(assembly.length_of(video) - target_seconds) <= 1.0
    assert assembly.length_of(video) <= selection.MAX_SECONDS
    for cut in joins(video):
        following = bisect.bisect_left(moments, cut)
        around = moments[max(following - 1, 0):following + 1]
        silence = moments[following] - (moments[following - 1] if following else 0.0)
        assert min(abs(cut - moment) for moment in around) < 1e-6 or silence > (
            rhythm.MAX_PIECE_SECONDS - rhythm.MIN_PIECE_SECONDS
        ), f"the join at {cut:.2f} misses a beat it could have reached"
