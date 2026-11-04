import dataclasses
import datetime
import logging
import pathlib

from montage import ffmpeg
from montage.video import assembly, frames

DEFAULT_TARGET_SECONDS = 60.0
MAX_SECONDS = 600.0

PHOTO_SECONDS = 2.5
CLIP_SECONDS = 4.0
DEDUP_SECONDS = 30.0

UNREADABLE, BLUR, EXPOSURE = frames.UNREADABLE, frames.BLUR, frames.EXPOSURE
DUPLICATE = "duplicate"
EXCLUDED = "excluded"
LEFT_OUT = "left_out"

log = logging.getLogger("montage.selection")


@dataclasses.dataclass
class Piece:
    media_id: str = ""
    sharpness: float = 0.0
    seconds: float = PHOTO_SECONDS
    when: datetime.datetime = datetime.datetime.min
    captured_at: datetime.datetime | None = None
    kind: str = "photo"
    path: pathlib.Path = pathlib.Path()
    start: float = 0.0
    best_second: float = 0.0
    duration: float | None = None
    rejected: str | None = None
    series_size: int = 1
    frame: bytes = dataclasses.field(default=b"", repr=False)
    live: float = 0.0


def judge(piece):
    try:
        ffmpeg.refuse_playlists(piece.path)
        looked = frames.sampled(piece.path, frames.SAMPLE_FPS if piece.kind == "video" else None)
        duration = ffmpeg.duration(piece.path) if piece.kind == "video" else None
    except RuntimeError as exc:
        log.info("cannot read the file of %s, leaving it out: %s", piece.media_id, exc)
        return dataclasses.replace(piece, rejected=UNREADABLE)

    moment, why = frames.best_moment(looked)
    if moment is None:
        return dataclasses.replace(piece, rejected=UNREADABLE)
    second, sharpness = moment
    looked_at = dataclasses.replace(
        piece,
        sharpness=sharpness,
        best_second=second,
        frame=looked[round(second * frames.SAMPLE_FPS)],
        duration=duration,
        rejected=why,
    )
    return cut_to(looked_at, CLIP_SECONDS if piece.kind == "video" else PHOTO_SECONDS)


def cut_to(piece, seconds):
    if piece.kind != "video":
        return dataclasses.replace(piece, seconds=seconds, start=0.0)
    duration = piece.duration or seconds
    seconds = min(seconds, duration)
    return dataclasses.replace(
        piece,
        seconds=seconds,
        start=min(max(piece.best_second - seconds / 2, 0.0), max(duration - seconds, 0.0)),
    )


def _dedup(pieces, always=()):
    always = set(always)
    kept = [piece for piece in pieces if piece.captured_at is None]
    dropped, series = [], []

    for piece in sorted((p for p in pieces if p.captured_at), key=lambda p: p.captured_at):
        ours = next(
            (
                one
                for one in series
                if (piece.captured_at - one[0].captured_at).total_seconds() <= DEDUP_SECONDS
                and any(frames.alike(piece.frame, shot.frame) for shot in one)
            ),
            None,
        )
        if ours:
            ours.append(piece)
        else:
            series.append([piece])

    for one in series:
        stands = [piece for piece in one if piece.media_id in always]
        stands = stands or [max(one, key=lambda piece: piece.sharpness)]
        held = {id(piece) for piece in stands}
        kept.extend(dataclasses.replace(piece, series_size=len(one)) for piece in stands)
        dropped.extend(
            dataclasses.replace(piece, rejected=DUPLICATE) for piece in one if id(piece) not in held
        )
    return kept, dropped


def choose(pieces, target_seconds, exclude=()):
    exclude = {str(one) for one in exclude}
    rejected, candidates = [], []
    for piece in pieces:
        if piece.media_id in exclude:
            rejected.append(dataclasses.replace(piece, rejected=EXCLUDED))
        elif piece.rejected == UNREADABLE:
            rejected.append(piece)
        else:
            candidates.append(piece)

    chosen, too_long = [], []
    for piece in sorted(candidates, key=lambda piece: -piece.sharpness):
        if chosen and assembly.length_of(chosen + [piece]) > target_seconds:
            too_long.append(piece)
            continue
        chosen.append(piece)

    short = target_seconds - assembly.length_of(chosen) - sum(map(_room, chosen))
    extra = next(
        (
            piece
            for piece in too_long
            if assembly.length_of(chosen + [piece]) - target_seconds < short
        ),
        None,
    )
    if extra:
        too_long.remove(extra)
        past = assembly.length_of(chosen + [extra]) - (
            target_seconds if extra.kind == "video" else MAX_SECONDS
        )
        chosen.append(cut_to(extra, extra.seconds - max(past, 0.0)))
    rejected += (dataclasses.replace(piece, rejected=LEFT_OUT) for piece in too_long)
    return sorted(chosen, key=lambda piece: piece.when), rejected


def stretch(pieces, target_seconds):
    missing = target_seconds - assembly.length_of(pieces)
    if missing <= 0:
        return list(pieces)
    lengths = [piece.seconds for piece in pieces]
    room = [_room(piece) for piece in pieces]
    growing = sorted((index for index, free in enumerate(room) if free > 0), key=room.__getitem__)
    log.info(
        "%.1fs short of the %.0fs ordered, %s clip(s) can be held longer",
        missing,
        target_seconds,
        len(growing),
    )
    for served, index in enumerate(growing):
        grown = min(missing / (len(growing) - served), room[index])
        lengths[index] += grown
        missing -= grown
    if missing > 0.01:
        log.warning(
            "material runs out %.1fs short of the %.0fs ordered: the video runs %.1fs → "
            "%.1fs, nothing in it can be held longer",
            missing,
            target_seconds,
            assembly.length_of(pieces),
            target_seconds - missing,
        )
    return [
        cut_to(piece, seconds) if seconds > piece.seconds else piece
        for piece, seconds in zip(pieces, lengths)
    ]


def cap(pieces):
    before = assembly.length_of(pieces)
    if before <= MAX_SECONDS:
        return list(pieces)
    kept = list(pieces)
    while assembly.length_of(kept[:-1]) >= MAX_SECONDS:
        kept.pop()
    last = kept[-1]
    kept[-1] = cut_to(last, last.seconds - (assembly.length_of(kept) - MAX_SECONDS))
    log.warning(
        "the video runs %.1fs, past the %.0fs no video may run: cutting it to %.1fs — "
        "%s piece(s) off the tail, the last held %.2fs instead of %.2fs",
        before,
        MAX_SECONDS,
        assembly.length_of(kept),
        len(pieces) - len(kept),
        kept[-1].seconds,
        last.seconds,
    )
    return kept


def _room(piece):
    return piece.duration - piece.seconds if piece.kind == "video" and piece.duration else 0.0
