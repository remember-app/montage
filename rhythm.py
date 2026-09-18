"""Ритм: how long each piece is held, so the joins between them land on the track's beats.

A join that falls on a beat reads as meant; the same join a quarter of a second early reads as «this
piece just ran out». That is the whole difference between a video and a pile of clips, and it costs
one shifted duration per piece: what goes into the montage and in what order was settled before this
module (`selection`), and the joining itself happens after it (`assembly`). Nothing here decodes a
frame — it only moves numbers.

Where the track comes from is not this module's business either: the library's own file and the one
the organizer uploaded arrive the same way, as a path (doc-03, «Ролик можно собрать под трек из
библиотеки и под свой, и склейки идут в ритм»).

The beats are found by librosa (ISC), which is what doc-11 names: a ready beat tracker for tenths of
a second of CPU per track, and no signal processing of our own in the worker because of it.
"""

import bisect
import dataclasses
import logging
import pathlib
import tempfile

import librosa

import assembly
import media

ANALYSIS_RATE = 22050  # the pulse is there at half rate too, and half the samples is half the work

# Fewer beats than this in a whole track is not a tempo but a handful of accidents — rain, a voice,
# a drone. A montage under one of those is cut the way it was cut before this module existed, which
# is the понятный фолбэк doc-03 asks for rather than a mess.
MIN_BEATS = 8

# The window a beat is looked for in: shorter and the frame is gone before it is seen, longer and a
# photograph hangs for half a minute under a slow track. A clip's upper edge is its own file too.
MIN_PIECE_SECONDS = 1.5
MAX_PIECE_SECONDS = 6.0

# How far past the end of the track its beats are counted. A track shorter than the montage is
# played again from the top by `assembly`, and the beats go round with it — the grid simply repeats
# every track length. Half an hour is well past the longest montage the api will take an order for
# (`MONTAGE_TARGET_SECONDS_MAX`, 300 seconds by default), and it is a few thousand numbers: cheaper
# than working out whether it is enough.
HORIZON_SECONDS = 30 * 60

log = logging.getLogger("montage.rhythm")


def beats(path):
    """The track's beats, in seconds and in order — or nothing at all when there is no tempo in it.

    Nothing is an answer and not a failure: under a track with no pulse the montage is assembled in
    even pieces, exactly as it was before there was a rhythm to cut to.

    The beats run past the end of the track because a short track is looped under a long montage
    (see `HORIZON_SECONDS`): without that, every piece after the track's last beat would have
    nothing to land on, and the montage would come apart precisely where the music went round again.
    """
    try:
        with tempfile.TemporaryDirectory() as workdir:
            # Through ffmpeg rather than librosa's own reading: behind that is libsndfile, which
            # opens neither the m4a off a phone nor half of what an organizer may upload as their
            # track. ffmpeg is in this worker anyway, it opens everything, and it is the same door
            # every other piece of somebody else's media comes through.
            wav = pathlib.Path(workdir) / "track.wav"
            media._ffmpeg("-i", str(path), "-ac", "1", "-ar", str(ANALYSIS_RATE), str(wav))
            samples, rate = librosa.load(wav, sr=ANALYSIS_RATE, mono=True)
        _, moments = librosa.beat.beat_track(y=samples, sr=rate, units="time")
        length = len(samples) / rate
    except Exception as exc:
        # Deliberately broad: a montage must not die over the analysis of its music. Anything that
        # goes wrong here — a track that will not decode, librosa disliking the data — costs even
        # joins and nothing else.
        log.warning("cannot make out the rhythm of %s, cutting in even pieces instead: %s",
                    path, exc)
        return ()
    if len(moments) < MIN_BEATS or not length:
        log.info("no tempo to speak of in %s (%s beats), cutting in even pieces",
                 path, len(moments))
        return ()
    circles = max(int(HORIZON_SECONDS / length), 1)
    return tuple(float(moment) + length * circle
                 for circle in range(circles) for moment in moments)


def fit(pieces, moments):
    """The same pieces in the same order, each held for as long as the beats want it held.

    It is the length of an already chosen piece that is adjusted, never the other way round: what is
    in the montage, and how much of it, does not depend on the music. Which is also why it does not
    matter what level librosa heard the pulse at — quarters or eighths: a dense grid moves a piece
    less and a sparse one more, and neither can move it away from the music.

    The last piece is left alone: nothing is joined to it, there is the end of the video and the
    music fading out.

    ponytail: beats, not bars. A join on a strong beat would read better still, but `beat_track`
    does not mark the first beat of a bar — that is a separate analysis. Should it ever be needed,
    only the choice of moment below changes.
    """
    if len(moments) < MIN_BEATS or len(pieces) < 2:
        return list(pieces)

    fitted, at = [], 0.0  # `at` is where the piece begins in the assembled montage
    for piece in pieces[:-1]:
        seconds = _snap(moments, piece, at)
        fitted.append(dataclasses.replace(piece, seconds=seconds))
        at += seconds - assembly.CROSSFADE_SECONDS
    fitted.append(pieces[-1])
    log.info("cut to the beat: %s became %s",
             [round(piece.seconds, 2) for piece in pieces],
             [round(piece.seconds, 2) for piece in fitted])
    return fitted


def _snap(moments, piece, at):
    """The length at which the join after this piece falls on a beat — or the piece's own length,
    when no beat fits inside what the piece may be.

    The beat looked for is not the nearest one but the nearest one that leaves the piece a length it
    is allowed to have: the bounds choose a beat rather than trim the answer to a beat already
    chosen. Trimming is what would make the mess — piece after piece of a second and a half wherever
    the grid has run ahead of the previous join.
    """
    # A join is heard at the middle of the crossfade rather than at the end of the piece: for half
    # of that dissolve the next piece is already on screen. That middle is what lands on the beat,
    # which is why half the crossfade is in every line below.
    half_fade = assembly.CROSSFADE_SECONDS / 2
    least, most = MIN_PIECE_SECONDS, _room(piece)
    if most < least:
        return piece.seconds  # a clip shorter than the minimum: its length is its file's to decide
    # The beats that leave a length between `least` and `most`. `moments` is sorted — librosa
    # hands it over that way and `beats` keeps it so.
    inside = moments[bisect.bisect_left(moments, at + least - half_fade):
                     bisect.bisect_right(moments, at + most - half_fade)]
    if not inside:
        return piece.seconds
    wanted = at + piece.seconds - half_fade  # where the join would be if there were no music
    return min(inside, key=lambda moment: abs(moment - wanted)) - at + half_fade


def _room(piece):
    """How long a piece may be held at most: a photograph as long as you like — it is a still — and
    a clip only while there is anything left in its file past the point it is cut from. Stretched
    further, it comes back from ffmpeg shorter than asked for, and the join misses the beat by
    exactly that difference."""
    if piece.kind == "video" and piece.duration:
        return min(MAX_PIECE_SECONDS, piece.duration - piece.start)
    return MAX_PIECE_SECONDS
