import bisect
import logging
import math
import pathlib
import statistics
import tempfile

import librosa

from montage import ffmpeg
from montage.video import assembly, selection

ANALYSIS_RATE = 22050

MIN_BEATS = 8

MIN_PIECE_SECONDS = 1.5
MAX_PIECE_SECONDS = 6.0

HORIZON_SECONDS = 30 * 60

log = logging.getLogger("montage.rhythm")


def beats(path):
    try:
        with tempfile.TemporaryDirectory() as workdir:
            wav = pathlib.Path(workdir) / "track.wav"
            ffmpeg.run(
                "-i",
                str(path),
                "-t",
                str(HORIZON_SECONDS),
                "-ac",
                "1",
                "-ar",
                str(ANALYSIS_RATE),
                str(wav),
            )
            samples, rate = librosa.load(wav, sr=ANALYSIS_RATE, mono=True)
        _, moments = librosa.beat.beat_track(y=samples, sr=rate, units="time")
        length = len(samples) / rate
    except Exception as exc:
        log.warning(
            "cannot make out the rhythm of %s, cutting in even pieces instead: %s", path, exc
        )
        return ()
    if len(moments) < MIN_BEATS or not length:
        log.info(
            "no tempo to speak of in %s (%s beats), cutting in even pieces", path, len(moments)
        )
        return ()
    circles = max(int(HORIZON_SECONDS / length), 1)
    return tuple(float(moment) + length * circle for circle in range(circles) for moment in moments)


def tempo(moments):
    beat = _beat(moments)
    return round(60 / beat) if beat else None


def _beat(moments):
    steps = [after - before for before, after in zip(moments, moments[1:]) if after > before]
    return statistics.median(steps) if steps else None


def fit(pieces, moments):
    if len(moments) < MIN_BEATS or len(pieces) < 2:
        return list(pieces)

    fitted, at, planned, beat = [], 0.0, 0.0, _beat(moments)
    for piece in pieces[:-1]:
        fitted.append(selection.cut_to(piece, _snap(moments, piece, at, planned, beat)))
        at += fitted[-1].seconds - assembly.CROSSFADE_SECONDS
        planned += piece.seconds - assembly.CROSSFADE_SECONDS
    last = pieces[-1]
    fitted.append(
        selection.cut_to(
            last,
            max(_whole_frames(last.seconds + planned - at), min(last.seconds, MIN_PIECE_SECONDS)),
        )
    )
    before, after = assembly.length_of(pieces), assembly.length_of(fitted)
    log.info(
        "cut to the beat: %s became %s, the video %.2fs → %.2fs",
        [round(piece.seconds, 2) for piece in pieces],
        [round(piece.seconds, 2) for piece in fitted],
        before,
        after,
    )
    if abs(after - before) > 1 / assembly.FPS:
        log.warning(
            "cut to the beat, the video runs %.2fs instead of %.2fs: the last piece could "
            "only be held %.2fs of the %.2fs that would end it there",
            after,
            before,
            fitted[-1].seconds,
            last.seconds + planned - at,
        )
    return fitted


def _snap(moments, piece, at, planned, beat):
    half_fade = assembly.CROSSFADE_SECONDS / 2
    least, most = MIN_PIECE_SECONDS, _room(piece)
    if most < least:
        return piece.seconds
    wanted = planned + piece.seconds - half_fade
    inside = moments[
        bisect.bisect_left(
            moments, max(at + least - half_fade, wanted - beat)
        ) : bisect.bisect_right(moments, min(at + most - half_fade, wanted + beat))
    ]
    if not inside:
        return piece.seconds
    return _whole_frames(min(inside, key=lambda moment: abs(moment - wanted)) - at + half_fade)


def _whole_frames(seconds):
    return round(seconds * assembly.FPS) / assembly.FPS


def _room(piece):
    ceiling = MAX_PIECE_SECONDS if piece.seconds <= MAX_PIECE_SECONDS else math.inf
    if piece.kind == "video" and piece.duration:
        return min(ceiling, piece.duration - piece.start)
    return ceiling
