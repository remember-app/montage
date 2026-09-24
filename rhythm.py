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
import logging
import math
import pathlib
import statistics
import tempfile

import librosa

import assembly
import media
import selection

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
# every track length. Half an hour is well past the longest montage anyone can order — ten minutes
# (doc-03, «не больше 10 минут») — and it is a few thousand numbers: cheaper than working out
# whether it is enough.
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
            # every other piece of somebody else's media comes through. Only up to the horizon: past
            # it nothing ever plays, and an hour of track takes about two gigabytes to analyse.
            wav = pathlib.Path(workdir) / "track.wav"
            media._ffmpeg("-i", str(path), "-t", str(HORIZON_SECONDS), "-ac", "1",
                          "-ar", str(ANALYSIS_RATE), str(wav))
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


def tempo(moments):
    """Темп дорожки в ударах в минуту — по тем же битам, что двигают склейки, или None, когда
    битов нет.

    Считается по медиане расстояний между соседними битами, а не по среднему: сетка повторяется
    вместе с зациклённым треком (`beats`), и на шве круга один промежуток всегда не такой, как
    остальные. Число идёт в журнал монтажа и больше никуда — на сборку темп не влияет, склейки
    попадают в сами биты.
    """
    beat = _beat(moments)
    return round(60 / beat) if beat else None


def _beat(moments):
    """How long one beat of the track lasts, in seconds, or None when there are no beats."""
    steps = [after - before for before, after in zip(moments, moments[1:]) if after > before]
    return statistics.median(steps) if steps else None


def fit(pieces, moments):
    """The same pieces in the same order, each held for as long as the beats want it held — and the
    video as long as it was handed in, which is the length that was ordered (`selection.stretch`).

    It is the length of an already chosen piece that is adjusted, never the other way round: what is
    in the montage, and how much of it, does not depend on the music. Which is also why it does not
    matter what level librosa heard the pulse at — quarters or eighths: a dense grid moves a piece
    less and a sparse one more, and neither can move it away from the music.

    Each join goes to the beat nearest the place it had before the music, not to the one nearest its
    piece's own length. Measured from wherever the join before it landed, every piece under a steady
    tempo would be pulled the same way and the pulls would add up — to a minute on a ten-minute
    video. Measured from its own place, a join is never more than a beat off it (`_snap`).

    The last piece has no join after it — there is the end of the video and the music fading out —
    so it takes up what the joins moved: held that much longer or shorter, it ends the video on the
    length it was handed. What it cannot take up — a clip with nothing left in its file, a piece
    already down to a glance — is less than a beat, and the log says how much.

    ponytail: beats, not bars. A join on a strong beat would read better still, but `beat_track`
    does not mark the first beat of a bar — that is a separate analysis. Should it ever be needed,
    only the choice of moment below changes.
    """
    if len(moments) < MIN_BEATS or len(pieces) < 2:
        return list(pieces)

    # Where the piece begins in the assembled montage, and where it began before the beats moved it.
    fitted, at, planned, beat = [], 0.0, 0.0, _beat(moments)
    for piece in pieces[:-1]:
        # Through `cut_to` and not a bare length swap: a beat moves how long a piece is held, and
        # `cut_to` is what keeps it cut *around* its best moment at that new length. Without it a
        # piece shrunk here would keep the `start` centred for its old length and could show
        # nothing of what `selection` cut it around in the first place.
        fitted.append(selection.cut_to(piece, _snap(moments, piece, at, planned, beat)))
        at += fitted[-1].seconds - assembly.CROSSFADE_SECONDS
        planned += piece.seconds - assembly.CROSSFADE_SECONDS
    last = pieces[-1]
    fitted.append(selection.cut_to(
        last, max(_whole_frames(last.seconds + planned - at), min(last.seconds, MIN_PIECE_SECONDS))
    ))
    before, after = assembly.length_of(pieces), assembly.length_of(fitted)
    log.info("cut to the beat: %s became %s, the video %.2fs → %.2fs",
             [round(piece.seconds, 2) for piece in pieces],
             [round(piece.seconds, 2) for piece in fitted], before, after)
    if abs(after - before) > 1 / assembly.FPS:
        log.warning("cut to the beat, the video runs %.2fs instead of %.2fs: the last piece could "
                    "only be held %.2fs of the %.2fs that would end it there", after, before,
                    fitted[-1].seconds, last.seconds + planned - at)
    return fitted


def _snap(moments, piece, at, planned, beat):
    """The length at which the join after this piece falls on a beat — or the piece's own length,
    when no beat fits inside what the piece may be.

    The beat looked for is the one nearest the join's place before the music, `planned` into the
    montage, among those that leave the piece a length it is allowed to have: the bounds choose a
    beat rather than trim the answer to a beat already chosen. Trimming is what would make the mess
    — piece after piece of a second and a half wherever the grid has run ahead of the previous join.

    And no further from that place than one `beat`: past it is not this join's beat but the far
    side of a stretch with none — a fade, the seam of a looped track — and a join pulled seconds
    across it takes more than the last piece can give back, and the video runs long.
    """
    # A join is heard at the middle of the crossfade rather than at the end of the piece: for half
    # of that dissolve the next piece is already on screen. That middle is what lands on the beat,
    # which is why half the crossfade is in every line below.
    half_fade = assembly.CROSSFADE_SECONDS / 2
    least, most = MIN_PIECE_SECONDS, _room(piece)
    if most < least:
        return piece.seconds  # a clip shorter than the minimum: its length is its file's to decide
    wanted = planned + piece.seconds - half_fade  # where the join would be if there were no music
    # The beats that leave a length between `least` and `most`. `moments` is sorted — librosa
    # hands it over that way and `beats` keeps it so.
    inside = moments[bisect.bisect_left(moments, max(at + least - half_fade, wanted - beat)):
                     bisect.bisect_right(moments, min(at + most - half_fade, wanted + beat))]
    if not inside:
        return piece.seconds
    return _whole_frames(min(inside, key=lambda moment: abs(moment - wanted)) - at + half_fade)


def _whole_frames(seconds):
    """A length the encoder can hold exactly: to the nearest of the video's frames (`assembly.FPS`).

    Every piece is encoded to a whole number of frames, each rounded on its own, and the lengths the
    beats ask for are no whole number of them. Left to the encoder, those roundings add up over
    hundreds of pieces — a second and more on a ten-minute video, and every join past them that far
    off its beat. Rounded here, the next join is measured from where this piece really ends: a join
    is within half a frame of its beat, and the video within half a frame of its length.
    """
    return round(seconds * assembly.FPS) / assembly.FPS


def _room(piece):
    """How long a piece may be held at most: a photograph as long as you like — it is a still — and
    a clip only while there is anything left in its file past the point it is cut from. Stretched
    further, it comes back from ffmpeg shorter than asked for, and the join misses the beat by
    exactly that difference.

    A piece that is already longer than the window is not pulled back into it: the length it
    arrived with was given to it to make the video as long as it was ordered
    (`selection.stretch`), and a ceiling for the sake of the music would quietly take those seconds
    out of the montage again. The window bounds what the *beats* may do to a piece, not how long a
    piece may be. Nor is it frozen at that length: held only shorter, every such piece in a row
    could only reach back for its beat, and in a video of stretched clips most joins missed theirs.
    """
    ceiling = MAX_PIECE_SECONDS if piece.seconds <= MAX_PIECE_SECONDS else math.inf
    if piece.kind == "video" and piece.duration:
        return min(ceiling, piece.duration - piece.start)
    return ceiling
