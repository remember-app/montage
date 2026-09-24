"""Отсев: which of a memory's material goes into a video, and why the rest of it does not.

Nothing here encodes anything. One file at a time is looked at frame by frame (`judge`), and out of
the judged pieces the ones the video is made of are picked (`choose`) — which hands back both
halves of the answer, the cuts and, on every piece left out, the reason it was. Both halves are the
point: the organizer is shown what the algorithm threw away and can overrule it in either
direction, and those two overrulings — «включить» and «исключить» — are honoured here and nowhere
else, so no later way of directing a video can get around them.

The pieces handed back carry their own length and their place in the finished video, and `assembly`
takes them from there knowing nothing about sharpness, exposure or bursts. The one thing borrowed
back from it is `length_of`: what the pieces have to fit into is the length of the *assembled*
video, and the crossfades that eat into it are assembly's own business.
"""

import collections
import dataclasses
import datetime
import logging
import pathlib
import subprocess

import numpy

import assembly
import media

DEFAULT_TARGET_SECONDS = 60.0  # what the api fills `params.target_seconds` with when nobody says
MAX_SECONDS = 600.0  # the api's `maxSeconds`: ten minutes, as long as a video may ever run

# A still held on screen; a plain frame, no Ken Burns. The same 2.5 seconds the api counts the
# longest orderable video by — every video whole and 2.5 seconds a photo, less a crossfade a join,
# down to the whole second (`materialLeft`, api src/montages/montages.ts) — so this number is not
# this module's alone to move: changed here and nowhere else, the app would offer a length the
# worker cannot reach.
PHOTO_SECONDS = 2.5
CLIP_SECONDS = 4.0  # «3–5 сек вокруг лучшего момента»
DEDUP_SECONDS = 2.0  # two shots this close together are one moment, not two

# Every frame is judged on a small grey copy of itself: blur and exposure both survive the scaling,
# and 96×54 is 5k pixels rather than 2M — cheap enough to judge every sampled second of every file
# in the memory.
#
# What is handed in matters to these numbers, though, and not only to how long they take: swscale
# cannot fit its filter past about a 64× reduction, so a 12032px original squeezed straight to 96
# is *sampled* rather than averaged. That reads high on sharpness (aliasing counts as detail) and
# wrong on exposure — a frame whose true mean luma is 25 came out at 13. A photograph therefore
# arrives here as the 2048px copy the preview made of it (`media.large_copy`), which is a plain
# average of every pixel, and the thresholds below are set against that (TASK-184).
SAMPLE_WIDTH, SAMPLE_HEIGHT = 96, 54
SAMPLE_FPS = 1.0  # «семпл кадров ≈1/сек»
MIN_SHARPNESS = 5.0  # variance of the Laplacian under which a frame is just blur
LUMA_RANGE = (24, 232)  # mean brightness outside this is a black or a blown-out frame
CLIPPED_LIMIT = 0.5  # and so is one whose histogram is mostly at either end

# Why a piece is not in the video — one of these on every piece left out. Codes and not sentences:
# they are read by other code and, in the end, shown to the organizer in their own language, so
# they must not change with the wording of a screen.
UNREADABLE = "unreadable"  # ffmpeg would not open it, or found no frame in it at all
BLUR = "blur"
EXPOSURE = "exposure"
DUPLICATE = "duplicate"  # another shot of the same moment stands for it
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
    # The moment `_best_moment` scored, in the original's own timeline — not where the cut begins.
    # Kept because the cover is taken from it: `start` is that moment pulled back by half a clip.
    best_second: float = 0.0
    # How long the file itself runs, for a video that ffmpeg would tell — None for a photo, which
    # runs for as long as it is held. What `cut_to` may not ask past (so a piece can be cut longer
    # than the default when the director or `rhythm` asks for it), and how far `rhythm` may stretch
    # the piece to put a join on a beat.
    duration: float | None = None
    rejected: str | None = None  # one of the reasons above, None while the piece is still in
    # Сколько кадров было в серии, за которую этот кусок остался один («взял 1 из 4»). Ставит
    # `_dedup`, и дальше это едет с куском до журнала монтажа: серия — одна запись в рассказе о
    # сборке, а не четыре, из которых три «мимо, дубль». Единица — кусок сам по себе.
    series_size: int = 1


# --- looking at frames -------------------------------------------------------


def _grey_frames(path, per_second=None):
    """The file as raw 96×54 grey frames — one a second for a video, its only one for a photo.

    One ffmpeg call and one bytes object: everything below reads those pixels directly, so nothing
    is decoded twice and no image library is needed to ask how sharp a frame is.
    """
    sampling = f"fps={per_second}," if per_second else ""
    framing = () if per_second else ("-frames:v", "1")  # a photo has the one frame to look at
    # Run rather than `media._ffmpeg`: this is the only call whose *output* is the point, and it
    # comes back as raw pixels on stdout, which `media._run`'s `text=True` would mangle.
    done = subprocess.run(
        (
            "ffmpeg", "-nostdin", "-loglevel", "error", "-protocol_whitelist", "file",
            "-i", str(path), *framing,
            "-vf", f"{sampling}scale={SAMPLE_WIDTH}:{SAMPLE_HEIGHT},format=gray",
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ),
        capture_output=True,
    )
    if done.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {done.stderr.decode(errors='replace').strip()[-500:]}")
    frame = SAMPLE_WIDTH * SAMPLE_HEIGHT
    return [done.stdout[at:at + frame] for at in range(0, len(done.stdout) - frame + 1, frame)]


def _sharpness(frame):
    """Variance of the Laplacian: how much the frame changes from pixel to pixel. A photograph in
    focus is full of edges and scores high; a smeared one has almost none and scores near zero.

    The shifted slices are the same five neighbours a per-pixel loop would visit, over the same
    interior, and `.var()` (ddof=0) is `statistics.pvariance` under another name — this is the old
    Python loop's number, ~100x faster. That win is the whole reason the worker depends on numpy.
    """
    # Signed before the subtraction: as `uint8` the neighbours would wrap around instead of going
    # negative, and a dark pixel next to a bright one is exactly what this is looking for.
    pixels = numpy.frombuffer(frame, dtype=numpy.uint8).reshape(SAMPLE_HEIGHT, SAMPLE_WIDTH)
    pixels = pixels.astype(numpy.int16)
    laplacian = (
        4 * pixels[1:-1, 1:-1] - pixels[1:-1, :-2] - pixels[1:-1, 2:]
        - pixels[:-2, 1:-1] - pixels[2:, 1:-1]  # the one-pixel border has no four neighbours
    )
    return float(laplacian.var())


def _exposed(frame):
    """False for a frame that is mostly black or blown out, off its own histogram."""
    histogram = collections.Counter(frame)
    pixels = len(frame)
    mean = sum(value * count for value, count in histogram.items()) / pixels
    clipped = sum(count for value, count in histogram.items() if value < 16 or value > 239) / pixels
    return LUMA_RANGE[0] <= mean <= LUMA_RANGE[1] and clipped <= CLIPPED_LIMIT


def _best_moment(frames):
    """`(second, sharpness)` of the frame the piece is worth cutting around, and `None` beside it —
    or, when no frame in the file is worth showing, the sharpest one there is and the reason.

    A rejected file gets a moment back too, because the rejection is only a proposal: the organizer
    can overrule it («включить»), and a dark clip that goes into the video anyway should still go in
    cut around the most detailed thing in it rather than around whatever its first frame happens to
    be. `None` for the moment means there was no frame at all to have an opinion about.

    ponytail: sharpness is standing in for the ranking architecture.md asks for — NIMA over ONNX,
    which is «how good is this picture» rather than «how in focus is it». Deliberately deferred:
    the model's weights, provenance and licence are a question for the human, not a default to pick
    here. When it is answered, the score returned from this function is the only thing that changes.
    """
    best = usable = None  # the sharpest frame of all, and the sharpest one worth showing
    exposed = False
    for index, frame in enumerate(frames):
        second, sharpness = index / SAMPLE_FPS, _sharpness(frame)
        if best is None or sharpness > best[1]:
            best = (second, sharpness)
        if not _exposed(frame):
            continue
        exposed = True
        if sharpness >= MIN_SHARPNESS and (usable is None or sharpness > usable[1]):
            usable = (second, sharpness)
    if usable:
        return usable, None
    if best is None:
        return None, UNREADABLE
    # Both bars were tried on every frame, so the reason is whichever one nothing got past: a file
    # with frames that are exposed and still score nothing is blurred, one with no exposed frame at
    # all is too dark or too blown out to say anything about focus.
    return best, BLUR if exposed else EXPOSURE


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
        media._refuse_playlists(piece.path)
        frames = _grey_frames(piece.path, SAMPLE_FPS if piece.kind == "video" else None)
        duration = media._duration(piece.path) if piece.kind == "video" else None
    except RuntimeError as exc:
        # Bytes we do have and cannot use — a refused playlist, a file ffmpeg will not decode.
        # Fetching them again would fail identically, so one bad file costs the memory that file
        # and not its whole video: anyone who can upload could otherwise deny the organizer a
        # montage for good, and «повторить» would meet the same file.
        log.info("cannot read the file of %s, leaving it out: %s", piece.media_id, exc)
        return dataclasses.replace(piece, rejected=UNREADABLE)

    moment, why = _best_moment(frames)
    if moment is None:
        return dataclasses.replace(piece, rejected=UNREADABLE)
    second, sharpness = moment
    looked_at = dataclasses.replace(
        piece, sharpness=sharpness, best_second=second, duration=duration, rejected=why
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
        piece, seconds=seconds,
        start=min(max(piece.best_second - seconds / 2, 0.0), max(duration - seconds, 0.0)),
    )


# --- choosing what goes in ---------------------------------------------------


def _dedup(pieces, always=()):
    """A burst of shots two seconds apart is one moment, and the sharpest of them stands for it —
    unless the organizer marked one of the burst «включить», and then theirs does, all of them.

    Only the capture date can say that — a file that carries none is left alone, exactly as
    architecture.md has it: «нет даты съёмки — дедуп по дате пропускаем».

    Returns what is kept and what its burst swallowed, the latter marked for why.
    """
    always = set(always)
    kept = [piece for piece in pieces if piece.captured_at is None]
    dropped, burst = [], []

    def settle(burst):
        stands = [piece for piece in burst if piece.media_id in always]
        stands = stands or [max(burst, key=lambda piece: piece.sharpness)]
        # By identity: two pieces of one burst can hold equal values and still be two files.
        held = {id(piece) for piece in stands}
        # Сколько кадров стояло за оставшимся, на нём самом: журналу монтажа серия — одна запись
        # «взял 1 из 4», а не четыре, из которых три «дубль» (`montage.py`).
        kept.extend(dataclasses.replace(piece, series_size=len(burst)) for piece in stands)
        dropped.extend(dataclasses.replace(piece, rejected=DUPLICATE)
                       for piece in burst if id(piece) not in held)

    for piece in sorted((p for p in pieces if p.captured_at), key=lambda p: p.captured_at):
        # Measured from the first of the burst and not from the one before, so a long sequence of
        # shots a second apart stays several moments instead of collapsing into one.
        if burst and (piece.captured_at - burst[0].captured_at).total_seconds() <= DEDUP_SECONDS:
            burst.append(piece)
            continue
        if burst:
            settle(burst)
        burst = [piece]
    if burst:
        settle(burst)
    return kept, dropped


def choose(pieces, target_seconds, include=(), exclude=()):
    """The cuts that make the video, in the order they are shown — and everything left out of it,
    each piece carrying the reason it was.

    Dedup, then the best of what is left up to the asked-for length — and one piece more, a clip cut
    to fit or a still past it by less than a second, when that is nearer to it (below) — then back
    into chronological order: «отбор по оценке, порядок в ролике — хронологический»
    (architecture.md). The first piece is taken whatever it costs: a montage shorter than asked for
    is a montage, an empty one is not.

    `include` and `exclude` are the organizer's own two lists of media ids, from the order, and they
    beat the algorithm in both directions. An excluded file is never in the video, however sharp it
    is. An included one is always in it, blurred or dark or both — the only things that word cannot
    overrule are a file with no frame in it at all, which is not a judgement but an absence, and the
    ten minutes no video runs past, which `cap` holds after every step that could cross it. What the
    included pieces take is spent before anything else is chosen, so the rest of the material fits
    into the time left rather than crowding them out.
    """
    include, exclude = {str(one) for one in include}, {str(one) for one in exclude}
    rejected, wanted, candidates = [], [], []
    for piece in pieces:
        if piece.media_id in exclude:
            # Named in both lists is the order contradicting itself; the word that keeps material
            # out of a video is the one that cannot be undone afterwards, so it is the one that wins.
            rejected.append(dataclasses.replace(piece, rejected=EXCLUDED))
        elif piece.rejected == UNREADABLE:
            rejected.append(piece)
        elif piece.media_id in include:
            wanted.append(dataclasses.replace(piece, rejected=None))
        elif piece.rejected:
            rejected.append(piece)
        else:
            candidates.append(piece)

    # The wanted go through the dedup with the rest so that a burst they belong to is still one
    # moment — theirs, and not also the sharpest shot standing next to it.
    kept, duplicates = _dedup(candidates + wanted, always=include)
    rejected += duplicates

    chosen, no_room = [piece for piece in kept if piece.media_id in include], []
    for piece in sorted((p for p in kept if p.media_id not in include), key=lambda p: -p.sharpness):
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
    extra = next((piece for piece in no_room
                  if assembly.length_of(chosen + [piece]) - target_seconds < short), None)
    if extra:
        no_room.remove(extra)
        past = assembly.length_of(chosen + [extra]) - (
            target_seconds if extra.kind == "video" else MAX_SECONDS)
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
    log.info("%.1fs short of the %.0fs ordered, %s clip(s) can be held longer",
             missing, target_seconds, len(growing))
    for served, index in enumerate(growing):
        # An equal share of what is still missing, out of the clips that can still take one. The
        # shortest file is served first, and what it cannot hold stays in `missing` for the longer
        # ones behind it — which is what makes the shares equal wherever they can be.
        grown = min(missing / (len(growing) - served), room[index])
        lengths[index] += grown
        missing -= grown
    if missing > 0.01:
        log.warning("material runs out %.1fs short of the %.0fs ordered: the video runs %.1fs → "
                    "%.1fs, nothing in it can be held longer", missing, target_seconds,
                    assembly.length_of(pieces), target_seconds - missing)
    return [cut_to(piece, seconds) if seconds > piece.seconds else piece
            for piece, seconds in zip(pieces, lengths)]


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
    log.warning("the video runs %.1fs, past the %.0fs no video may run: cutting it to %.1fs — "
                "%s piece(s) off the tail, the last held %.2fs instead of %.2fs", before,
                MAX_SECONDS, assembly.length_of(kept), len(pieces) - len(kept), kept[-1].seconds,
                last.seconds)
    return kept


def _room(piece):
    """How much longer `stretch` can hold a piece than it is cut. A photograph has nowhere to grow
    to, and neither has a clip whose length ffmpeg would not tell: asked past what is really in the
    file, ffmpeg hands back a shorter piece than it was told to, and every join after it lands
    wrong."""
    return piece.duration - piece.seconds if piece.kind == "video" and piece.duration else 0.0
