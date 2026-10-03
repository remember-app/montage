"""Отбор: which of a memory's material goes into a video, and why the rest of it does not.

Nothing here encodes anything. One file at a time is looked at frame by frame (`judge`, on what
`frames.py` reads off it), and out of the judged pieces the ones the video is made of are picked
(`choose`) — which hands back both halves of the answer, the cuts and, on every piece left out, the
reason it was. Both halves are the point: the organizer is shown what did not go into the video and
why, and the one overruling of theirs honoured here and nowhere else — «исключить» — so no later way
of directing a video can get around it.

**Отсева внутри сборки нет.** Кадры, оставленные организатором при выборе, равны: смазанный и
тёмный идут в ролик так же, как резкий и светлый, и решать это не отбору (doc-03, «What it
doesn't») — брак предлагается убрать на экране заказа (`media/screening.py`), и там решает
человек. «Включить» больше ничего не закрепляет: режиссёр выбирает из всех оставленных сам.

The pieces handed back carry their own length and their place in the finished video, and `assembly`
takes them from there knowing nothing about sharpness or exposure. The one thing borrowed
back from it is `length_of`: what the pieces have to fit into is the length of the *assembled*
video, and the crossfades that eat into it are assembly's own business.
"""

import dataclasses
import datetime
import logging
import pathlib

from montage import ffmpeg
from montage.video import assembly, frames

DEFAULT_TARGET_SECONDS = 60.0  # what the api fills `params.target_seconds` with when nobody says
MAX_SECONDS = 600.0  # the api's `maxSeconds`: ten minutes, as long as a video may ever run

# A still held on screen; a plain frame, no Ken Burns. The same 2.5 seconds the api counts the
# longest orderable video by — every video whole and 2.5 seconds a photo, less a crossfade a join,
# down to the whole second (`materialLeft`, api src/montages/montages.ts) — so this number is not
# this module's alone to move: changed here and nowhere else, the app would offer a length the
# worker cannot reach.
PHOTO_SECONDS = 2.5
CLIP_SECONDS = 4.0  # «3–5 сек вокруг лучшего момента»
# How far from the first shot of a series a shot of the same picture is still that moment: a burst,
# or a few retakes five to ten seconds apart to catch the smile. Past it, the same picture is a new
# moment in the same place — the table at the start of the dinner and at its end.
DEDUP_SECONDS = 30.0

# Why a piece is not in the video — one of these on every piece left out. Codes and not sentences:
# they are read by other code and, in the end, shown to the organizer in their own language, so
# they must not change with the wording of a screen.
# Три из них — о самом кадре, и считает их `frames.py`; остальные три о куске в ролике.
UNREADABLE, BLUR, EXPOSURE = frames.UNREADABLE, frames.BLUR, frames.EXPOSURE
DUPLICATE = "duplicate"  # another shot of the same moment stands for it — только у предложения на
# экране заказа (`media/screening.py`); в ролике отсева нет и быть не может
EXCLUDED = "excluded"  # the organizer said no
NO_ROOM = "no_room"  # good enough, but the video is only so long

log = logging.getLogger("montage.selection")


@dataclasses.dataclass
class Piece:
    """One cut of one file, ready to be laid into the montage. Every field has a default so a test
    can name only the ones the choosing actually reads."""

    media_id: str = ""  # the `media_items` row it is cut from — what the organizer's lists name
    sharpness: float = 0.0
    seconds: float = PHOTO_SECONDS
    when: datetime.datetime = datetime.datetime.min  # chronological place in the finished video
    captured_at: datetime.datetime | None = None  # None → this file takes no part in the dedup
    kind: str = "photo"
    # On disk in the workdir, not in R2 — every other `…_key` in this worker is a bucket key, and
    # this one is what ffmpeg is pointed at: first the downloaded file the отсев looks at (a
    # photograph's large copy where there is one, the original otherwise), then the cut of it.
    path: pathlib.Path = pathlib.Path()
    start: float = 0.0  # where in the original the cut begins
    # The moment `frames.best_moment` scored, in the original's own timeline — not where the cut
    # begins. Kept because the cover is taken from it: `start` is that moment pulled back by half
    # a clip.
    best_second: float = 0.0
    # How long the file itself runs, for a video that ffmpeg would tell — None for a photo, which
    # runs for as long as it is held. What `cut_to` may not ask past (so a piece can be cut longer
    # than the default when the director or `rhythm` asks for it), and how far `rhythm` may stretch
    # the piece to put a join on a beat.
    duration: float | None = None
    rejected: str | None = None  # one of the reasons above, None while the piece is still in
    # Сколько кадров было в серии, за которую этот кусок остался один («взял 1 из 4»). Ставит
    # `_dedup`, и дальше это едет с куском до журнала монтажа. Единица — кусок сам по себе.
    series_size: int = 1
    # The sampled frame of the best moment (`frames.sampled`), what `_dedup` tells a retake of the
    # same picture by. Empty until `judge` has looked at the file — and then nothing is like it.
    frame: bytes = dataclasses.field(default=b"", repr=False)
    # Сколько на куске слышно его собственного звука, долей от 0 до 1: 0 — одна музыка, 1 — один
    # живой звук, между — смесь, где музыке достаётся остальное. Выбирает режиссёр, кусок за
    # куском; отсев не выбирает ничего, и под его кусками играет музыка. Кусок, которому нечем
    # звучать, — фотография, клип без звука — идёт под музыку при любом числе здесь, а ролик без
    # музыки весь звучит живым звуком (`assembly.assemble`).
    live: float = 0.0


def judge(piece):
    """One downloaded file looked at: the piece with its best moment on it, or with `rejected`
    saying why there is nothing in it worth showing.

    Everything that decides whether a file *can* go into a video happens here and only here, on the
    file alone — so the same judgement can be passed on material long before anybody orders a video
    out of it.
    """
    try:
        # Before ffmpeg is pointed at it: these bytes came off somebody's phone, and a playlist
        # dressed up as a photo would have ffmpeg go and read this worker's own disk.
        ffmpeg.refuse_playlists(piece.path)
        looked = frames.sampled(piece.path, frames.SAMPLE_FPS if piece.kind == "video" else None)
        duration = ffmpeg.duration(piece.path) if piece.kind == "video" else None
    except RuntimeError as exc:
        # Bytes we do have and cannot use — a refused playlist, a file ffmpeg will not decode.
        # Fetching them again would fail identically, so one bad file costs the memory that file
        # and not its whole video: anyone who can upload could otherwise deny the organizer a
        # montage for good, and «повторить» would meet the same file.
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
    """The same piece cut to `seconds` — centred on its best moment, never past what the file holds.

    Where the length of a piece is settled, wherever the number comes from: `judge` asks for the
    three or four seconds this module cuts by default, the director asks for what its plan says.
    A photograph is held for exactly as long as it is asked to be; a clip can only give what it has,
    so the cut is pulled back inside the file at both ends — which is also why it may start earlier
    than the moment it is built around.
    """
    if piece.kind != "video":
        return dataclasses.replace(piece, seconds=seconds, start=0.0)
    # A clip whose length ffmpeg would not report is taken at its word: the cut is what was asked
    # for, from its start, which is what this module did before it measured anything.
    duration = piece.duration or seconds
    seconds = min(seconds, duration)
    return dataclasses.replace(
        piece,
        seconds=seconds,
        start=min(max(piece.best_second - seconds / 2, 0.0), max(duration - seconds, 0.0)),
    )


# --- choosing what goes in ---------------------------------------------------


def _dedup(pieces, always=()):
    """A series of shots of one moment, and the sharpest of them stands for it — unless the
    organizer marked one of the series «включить», and then theirs does, all of them.

    **В ролике этого больше нет:** дедуп сворачивал серию кадров за спиной режиссёра, а кадры,
    оставленные при выборе, равны (doc-03, «What it doesn't»). Остался он в предложении на
    экране заказа (`media/screening.py`), где человек решает, что убрать.

    One moment is two things at once: taken within `DEDUP_SECONDS` of the first shot of the series,
    and the same picture as a shot already in it (`frames.alike`). Time alone is not it — two
    shots a second apart of the cake and of the guests are two moments, and a retake ten seconds
    later is one, however many other shots were taken between the two. A file with no capture date
    is left alone, exactly as architecture.md has it: «нет даты съёмки — дедуп по дате пропускаем».

    Returns what is kept and what its series swallowed, the latter marked for why.
    """
    always = set(always)
    kept = [piece for piece in pieces if piece.captured_at is None]
    dropped, series = [], []

    for piece in sorted((p for p in pieces if p.captured_at), key=lambda p: p.captured_at):
        # Measured from the first of a series and not from the one before, so a long sequence of
        # the same picture stays several moments instead of collapsing into one.
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
        # By identity: two pieces of one series can hold equal values and still be two files.
        held = {id(piece) for piece in stands}
        # Сколько кадров стояло за оставшимся, на нём самом: «взял 1 из 4» в журнале монтажа.
        kept.extend(dataclasses.replace(piece, series_size=len(one)) for piece in stands)
        dropped.extend(
            dataclasses.replace(piece, rejected=DUPLICATE) for piece in one if id(piece) not in held
        )
    return kept, dropped


def choose(pieces, target_seconds, exclude=()):
    """The cuts that make the video, in the order they are shown — and everything left out of it,
    each piece carrying the reason it was.

    The sharpest of what is left up to the asked-for length — and one piece more, a clip cut to fit
    or a still past it by less than a second, when that is nearer to it (below) — then back into
    chronological order: «отбор по оценке, порядок в ролике — хронологический» (architecture.md).
    The first piece is taken whatever it costs: a montage shorter than asked for is a montage, an
    empty one is not.

    `exclude` is the organizer's own list of media ids from the order, and it beats the algorithm
    outright: an excluded file is never in the video, however sharp it is. There is no list of the
    other kind — the frames left at the order are equal, and every one of them is offered here
    (doc-03, «What it doesn't»). The only thing that takes a piece out on its own is a file with no
    frame in it at all, which is an absence rather than a judgement, and the ten minutes no video
    runs past, which `cap` holds after every step that could cross it.
    """
    exclude = {str(one) for one in exclude}
    rejected, candidates = [], []
    for piece in pieces:
        if piece.media_id in exclude:
            # The word that keeps material out of a video is the one that cannot be undone
            # afterwards, so where the order names the same file twice it is this one that wins.
            rejected.append(dataclasses.replace(piece, rejected=EXCLUDED))
        elif piece.rejected == UNREADABLE:
            rejected.append(piece)
        else:
            candidates.append(piece)

    chosen, no_room = [], []
    for piece in sorted(candidates, key=lambda piece: -piece.sharpness):
        if chosen and assembly.length_of(chosen + [piece]) > target_seconds:
            # Too long for what is left, but a shorter cut behind it may still fit.
            no_room.append(piece)
            continue
        chosen.append(piece)

    # An order is whole seconds, and a piece left out adds its whole length or nothing. So when the
    # clips in the video cannot be held long enough to make up what is left (`stretch`), the
    # sharpest piece left out that lands the video nearer the order than it is without goes in
    # after all. A clip is then cut to end the video on the order. A still is held its 2.5 seconds,
    # which runs past the order by less than one — the two misses add up to its two — and is cut
    # only at ten minutes, the edge of any video. It is what makes an order for the api's ceiling
    # come out that long: the ceiling is the length of everything, rounded down to the second.
    short = target_seconds - assembly.length_of(chosen) - sum(map(_room, chosen))
    extra = next(
        (
            piece
            for piece in no_room
            if assembly.length_of(chosen + [piece]) - target_seconds < short
        ),
        None,
    )
    if extra:
        no_room.remove(extra)
        past = assembly.length_of(chosen + [extra]) - (
            target_seconds if extra.kind == "video" else MAX_SECONDS
        )
        chosen.append(cut_to(extra, extra.seconds - max(past, 0.0)))
    rejected += (dataclasses.replace(piece, rejected=NO_ROOM) for piece in no_room)
    return sorted(chosen, key=lambda piece: piece.when), rejected


# --- the length that was ordered ---------------------------------------------


def stretch(pieces, target_seconds):
    """The same pieces in the same order, their clips held longer, until the video runs as long as
    the order asked for.

    Whoever directed the montage — the отбор above or the режиссёр beside it — picked these out of
    what the memory holds, and both stop at the first piece that would not fit the ordered length.
    So what comes back from them is usually shorter than the order, by however much the piece that
    did not fit would have taken — and past it only by the one still `choose` lets run over when
    nothing here could make up the rest. When nothing else can be *added*, the length is kept by
    holding what is already in the video for longer, and that is this step: the ordered length is a
    promise and not a wish (doc-03, «Длина ролика — число секунд»).

    Only clips grow. A photograph is held for the `PHOTO_SECONDS` the screen promises — a still
    stretched to a minute is not a longer video, it is a stopped one — and a clip only while its
    own file has anything left to show: the cut widens around the same best moment (`cut_to`), up
    to the whole file and no further. The missing seconds are shared out equally, the shortest file
    first, so what one clip cannot give is given by the others instead of being lost.

    Nothing is added to the video here and nothing is taken out of it: a frame the отсев turned
    down, or the organizer struck out, does not come back because there are seconds going spare —
    the length is filled out of what is already in the video. Material that runs out before the
    order is filled is not an error either: the video is as long as there was material for, and
    that is the whole of it.

    The crossfades are counted in, because `assembly.length_of` counts them: the half-second each
    join eats is missing out of the finished file exactly as an unfilled second is.
    """
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
        # An equal share of what is still missing, out of the clips that can still take one. The
        # shortest file is served first, and what it cannot hold stays in `missing` for the longer
        # ones behind it — which is what makes the shares equal wherever they can be.
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
    """The same pieces in the same order, cut from the tail to the ten minutes no video runs past.

    Every step before this one can carry a video over `MAX_SECONDS`: «включить» is spent whatever
    it costs (`choose`), the director puts back what its plan left out, and the beats by the
    fraction of one that the last piece cannot give back (`rhythm.fit`). Ten minutes is a hard
    edge and not a wish, so it is held once, after all of them: whole pieces come off the end while
    the ones before them already reach the ceiling, and the piece that crosses it is held shorter —
    a clip cut around its best moment like any other, a still for less than its 2.5 seconds.
    Everything ahead of that is left exactly as it was.
    """
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
    """How much longer `stretch` can hold a piece than it is cut. A photograph has nowhere to grow
    to, and neither has a clip whose length ffmpeg would not tell: asked past what is really in the
    file, ffmpeg hands back a shorter piece than it was told to, and every join after it lands
    wrong."""
    return piece.duration - piece.seconds if piece.kind == "video" and piece.duration else 0.0
