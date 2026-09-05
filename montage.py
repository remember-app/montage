"""The `montage` job: one video assembled out of a memory's material.

The organizer orders a video, the api writes a `montages` row and queues `('montage', <id>)`; from
`processing` onward the row is this worker's (see the api's migration `020.do.montages.sql`). Here
the memory's ready originals are pulled out of R2, looked at frame by frame, the good moments are
cut and crossfaded into one 1080p file with a track over it, and the row is left `ready` with the
key on it — then the api is knocked on so the phones get told.

The steps are architecture.md, «Montage-пайплайн», with one departure: selection (its step 4) runs
before normalisation (its step 3), so only the pieces that made the cut are ever encoded to 1080p
instead of every file the memory holds. `montages` and `media_items` belong to the api's migrations;
this module only reads and writes rows by id. Encoding goes through `media.py`'s helpers — one
worker, one way of running the tool; only `_grey_frames` runs ffmpeg itself, because it needs raw
pixels back on stdout.
"""

import collections
import dataclasses
import datetime
import json
import logging
import os
import pathlib
import subprocess
import tempfile
import urllib.request

import numpy

import media

MIN_ITEMS = 3  # «Меньше 3 → сразу failed» — the api refuses the order at this number too
DEFAULT_TARGET_SECONDS = 60.0  # what the api fills `params.target_seconds` with when nobody says

# The common canvas everything is normalised onto. 1920×1080 rather than the phone's own 1080×1920:
# the finished video leaves the app through «поделиться» and 16:9 is what plays everywhere. Vertical
# material is padded into it, horizontal fits as it is.
CANVAS_WIDTH, CANVAS_HEIGHT, FPS = 1920, 1080, 30
CANVAS = (
    f"scale={CANVAS_WIDTH}:{CANVAS_HEIGHT}:force_original_aspect_ratio=decrease,"
    f"pad={CANVAS_WIDTH}:{CANVAS_HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS},format=yuv420p"
)

PHOTO_SECONDS = 3.0  # a still held on screen; a plain frame, no Ken Burns
CLIP_SECONDS = 4.0  # «3–5 сек вокруг лучшего момента»
CROSSFADE_SECONDS = 0.5
MUSIC_FADE_SECONDS = 1.5
DEDUP_SECONDS = 2.0  # two shots this close together are one moment, not two

# Every frame is judged on a small grey copy of itself: blur and exposure both survive the scaling,
# and 96×54 is 5k pixels rather than 2M — cheap enough to judge every sampled second of every file
# in the memory.
SAMPLE_WIDTH, SAMPLE_HEIGHT = 96, 54
SAMPLE_FPS = 1.0  # «семпл кадров ≈1/сек»
MIN_SHARPNESS = 5.0  # variance of the Laplacian under which a frame is just blur
LUMA_RANGE = (24, 232)  # mean brightness outside this is a black or a blown-out frame
CLIPPED_LIMIT = 0.5  # and so is one whose histogram is mostly at either end

# The tracks, drawn by ffmpeg itself out of nothing (`aevalsrc`): three sine tones a chord apart
# under a slow swell. Not a licence, not a file in the repository, not an account to buy them from —
# a placeholder that plays.
# ponytail: synthesized tones stand in for music. Three real royalty-free tracks replace this dict
# and nothing else, once the human has settled where they come from and under what licence.
_SWELL = "(0.75+0.25*sin(2*PI*0.15*t))"
TRACKS = {
    "warm": f"{_SWELL}*(0.25*sin(2*PI*220*t)+0.18*sin(2*PI*277.18*t)+0.12*sin(2*PI*329.63*t))",
    "bright": f"{_SWELL}*(0.22*sin(2*PI*329.63*t)+0.16*sin(2*PI*415.3*t)+0.12*sin(2*PI*493.88*t))",
    "calm": f"{_SWELL}*(0.2*sin(2*PI*174.61*t)+0.14*sin(2*PI*261.63*t)+0.1*sin(2*PI*349.23*t))",
}
DEFAULT_TRACK = "warm"

API_TIMEOUT_SECONDS = 10

log = logging.getLogger("montage.montage")


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


def _best_moment(path, kind):
    """`(second, sharpness)` of the best frame in the file, or None when it has no usable one.

    ponytail: sharpness is standing in for the ranking architecture.md asks for — NIMA over ONNX,
    which is «how good is this picture» rather than «how in focus is it». Deliberately deferred:
    the model's weights, provenance and licence are a question for the human, not a default to pick
    here. When it is answered, the score returned from this function is the only thing that changes.
    """
    best = None
    for index, frame in enumerate(_grey_frames(path, SAMPLE_FPS if kind == "video" else None)):
        if not _exposed(frame):
            continue
        sharpness = _sharpness(frame)
        if sharpness < MIN_SHARPNESS:
            continue
        if best is None or sharpness > best[1]:
            best = (index / SAMPLE_FPS, sharpness)
    return best


# --- choosing what goes in ---------------------------------------------------


@dataclasses.dataclass
class Piece:
    """One cut of one file, ready to be laid into the montage. Every field has a default so a test
    can name only the ones the choosing actually reads."""

    sharpness: float = 0.0
    seconds: float = PHOTO_SECONDS
    when: datetime.datetime = datetime.datetime.min  # chronological place in the finished video
    captured_at: datetime.datetime | None = None  # None → this file takes no part in the dedup
    kind: str = "photo"
    # On disk in the workdir, not in R2 — every other `…_key` in this worker is a bucket key, and
    # this one is what ffmpeg is pointed at: first the downloaded original, then the cut of it.
    path: pathlib.Path = pathlib.Path()
    start: float = 0.0  # where in the original the cut begins


def _dedup(pieces):
    """A burst of shots two seconds apart is one moment, and the sharpest of them stands for it.

    Only the capture date can say that — a file that carries none is left alone, exactly as
    architecture.md has it: «нет даты съёмки — дедуп по дате пропускаем».
    """
    kept = [piece for piece in pieces if piece.captured_at is None]
    burst = []
    for piece in sorted((p for p in pieces if p.captured_at), key=lambda p: p.captured_at):
        # Measured from the first of the burst and not from the one before, so a long sequence of
        # shots a second apart stays several moments instead of collapsing into one.
        if burst and (piece.captured_at - burst[0].captured_at).total_seconds() <= DEDUP_SECONDS:
            burst.append(piece)
            continue
        if burst:
            kept.append(max(burst, key=lambda p: p.sharpness))
        burst = [piece]
    if burst:
        kept.append(max(burst, key=lambda p: p.sharpness))
    return kept


def length_of(pieces):
    """How long these pieces run once crossfaded — each join eats half a second of the total."""
    return sum(piece.seconds for piece in pieces) - CROSSFADE_SECONDS * (len(pieces) - 1)


def choose(pieces, target_seconds):
    """The cuts that make the video, in the order they are shown.

    Dedup, then the best of what is left up to the asked-for length, then back into chronological
    order — «отбор по оценке, порядок в ролике — хронологический» (architecture.md). The first piece
    is taken whatever it costs: a montage shorter than asked for is a montage, an empty one is not.
    """
    chosen = []
    for piece in sorted(_dedup(pieces), key=lambda p: -p.sharpness):
        if chosen and length_of(chosen + [piece]) > target_seconds:
            continue  # too long for what is left, but a shorter cut behind it may still fit
        chosen.append(piece)
    return sorted(chosen, key=lambda p: p.when)


# --- cutting and assembling --------------------------------------------------


def _normalise(piece, out):
    """One piece onto the common canvas: 1080p, 30 fps, padded, and silent.

    Silent because the finished video's only sound is the track (architecture.md: «звук исходных
    клипов по умолчанию убираем»), which also makes every clip's audio identical by having none.
    """
    if piece.kind == "photo":
        opened = ("-loop", "1", "-t", f"{piece.seconds}", "-i", str(piece.path))
    else:
        opened = ("-ss", f"{piece.start}", "-t", f"{piece.seconds}", "-i", str(piece.path))
    media._ffmpeg(
        *opened, "-vf", CANVAS, "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(out),
    )


def _assemble(clips, track_id, out):
    """The normalised clips into one file: crossfaded end to end, the track faded over the top."""
    total = length_of(clips)
    fade = min(MUSIC_FADE_SECONDS, total / 3)

    # Each xfade joins everything built so far to the next clip, starting half a second before the
    # chain ends — so the chain grows by the clip's length minus the overlap, which is `length_of`.
    chain, label, ends_at = [], "0:v", clips[0].seconds
    for index, clip in enumerate(clips[1:], start=1):
        offset = ends_at - CROSSFADE_SECONDS
        chain.append(
            f"[{label}][{index}:v]xfade=transition=fade:"
            f"duration={CROSSFADE_SECONDS}:offset={offset:.3f}[v{index}]"
        )
        label, ends_at = f"v{index}", offset + clip.seconds
    music = len(clips)  # the generated track is the input after the last clip
    chain.append(
        f"[{music}:a]afade=t=in:st=0:d={fade:.3f},"
        f"afade=t=out:st={total - fade:.3f}:d={fade:.3f}[a]"
    )

    # `-protocol_whitelist` is an *input* option: ffmpeg scopes it to the group it precedes, so the
    # one `media._ffmpeg` puts at the front guards input 0 and nothing else. This is the worker's
    # only multi-input call, so it is the only place that has to repeat it — every clip and the
    # generated track get their own, and the whitelist means the same thing on all of them.
    inputs = [
        argument
        for clip in clips
        for argument in ("-protocol_whitelist", "file", "-i", str(clip.path))
    ]
    media._ffmpeg(
        *inputs, "-protocol_whitelist", "file",
        "-f", "lavfi", "-i", f"aevalsrc={TRACKS[track_id]}:d={total:.3f}",
        "-filter_complex", ";".join(chain),
        # A single clip is never crossfaded, so there is no filter output to map — the input itself
        # is the video, and it is named the way ffmpeg names inputs rather than filter outputs.
        "-map", f"[{label}]" if len(clips) > 1 else "0:v", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-shortest", str(out),
    )


# --- the handler -------------------------------------------------------------


def montage(conn, ref):
    """One ordered video: the memory's material out of R2, the assembled file back in, the row
    to `ready` — and then the api, so the phones hear about it."""
    row = conn.execute("SELECT memory_id, params FROM montages WHERE id = %s", (ref,)).fetchone()
    if row is None:
        # The memory was deleted while the job waited, and the montage with it. Nothing to assemble
        # and nothing wrong: retrying would only fail three times.
        log.info("montage %s is gone, nothing to assemble", ref)
        return
    memory_id, params = row
    params = params or {}
    conn.execute("UPDATE montages SET status = 'processing', updated_at = now() WHERE id = %s", (ref,))

    target_seconds = float(params.get("target_seconds") or DEFAULT_TARGET_SECONDS)
    track_id = params.get("music_track_id") or DEFAULT_TRACK
    if track_id not in TRACKS:
        # The api refuses an unknown id at the order, so this is the two lists having drifted apart.
        # Retrying would ask for the same missing track three times.
        return _give_up_for_good(conn, ref, f"неизвестный музыкальный трек «{track_id}»")

    items = conn.execute(
        # Ordered the way the gallery orders the same rows (api, `src/media/gallery.ts`): the
        # finished video runs in the order the memory is already read in.
        """SELECT kind, original_key, captured_at, created_at
           FROM media_items WHERE memory_id = %s AND status = 'ready'
           ORDER BY coalesce(captured_at, created_at), id""",
        (memory_id,),
    ).fetchall()
    if len(items) < MIN_ITEMS:
        # Material may have been deleted between the order and here, so the api's check at the order
        # is not the last word. Not a transient failure — three retries would find the same library.
        return _give_up_for_good(conn, ref, "недостаточно материала для ролика")

    video_key = f"montages/{ref}.mp4"
    s3 = media.storage()
    with tempfile.TemporaryDirectory() as workdir:
        workdir = pathlib.Path(workdir)
        pieces = []
        for index, (kind, original_key, captured_at, created_at) in enumerate(items):
            original = workdir / f"original-{index}"
            # A download that fails is left fatal on purpose: the bytes may well arrive next time,
            # and swallowing an unreachable bucket would quietly cut a video out of half a memory.
            s3.download_file(media.bucket(), original_key, str(original))
            try:
                # Before ffmpeg is pointed at it: these bytes came off somebody's phone, and a
                # playlist dressed up as a photo would have ffmpeg go and read this worker's own
                # disk.
                media._refuse_playlists(original)
                best = _best_moment(original, kind)
                duration = media._duration(original) if kind == "video" else None
            except RuntimeError as exc:
                # Bytes we do have and cannot use — a refused playlist, a file ffmpeg will not
                # decode. Fetching them again would fail identically, so one bad file costs the
                # memory that file and not its whole video: anyone who can upload could otherwise
                # deny the organizer a montage for good, and «повторить» would meet the same file.
                log.info("cannot read %s, leaving it out: %s", original_key, exc)
                continue
            if best is None:
                log.info("nothing usable in %s, leaving it out", original_key)
                continue
            second, sharpness = best
            seconds, start = PHOTO_SECONDS, 0.0
            if kind == "video":
                duration = duration or CLIP_SECONDS
                seconds = min(CLIP_SECONDS, duration)
                # Centred on the best moment, then pulled back inside the file at both ends.
                start = min(max(second - seconds / 2, 0.0), max(duration - seconds, 0.0))
            pieces.append(Piece(
                sharpness=sharpness, seconds=seconds, when=captured_at or created_at,
                captured_at=captured_at, kind=kind, path=original, start=start,
            ))

        # The floor again, now against what survived. The count above was of rows; files drop out
        # here for being unreadable or having no usable frame in them, and three rows that come to
        # one still would otherwise be assembled into a three-second «ролик» against a sixty-second
        # order — the very thing the floor is there to refuse.
        if len(pieces) < MIN_ITEMS:
            return _give_up_for_good(
                conn, ref, "материал не годится для ролика: файлы не читаются или кадры испорчены"
            )

        clips = []
        for index, piece in enumerate(choose(pieces, target_seconds)):
            clip = workdir / f"clip-{index}.mp4"
            _normalise(piece, clip)
            # Measured, not the length that was asked for. `-t` cuts against the *video* stream
            # while `_duration` reads the container, which is the longest stream in it — a clip
            # whose audio outruns its picture normalises shorter than declared. `_assemble` steps
            # the crossfade offsets along by these numbers, so a stale one walks the whole chain
            # off the end and ffmpeg truncates the montage without a word of complaint.
            clips.append(dataclasses.replace(
                piece, path=clip, seconds=media._duration(clip) or piece.seconds
            ))
        assembled = workdir / "montage.mp4"
        _assemble(clips, track_id, assembled)
        duration = media._duration(assembled)
        media._upload(s3, assembled, video_key, "video/mp4")

    landed = conn.execute(
        """UPDATE montages
           SET status = 'ready', video_key = %s, duration_seconds = %s, error_text = null,
               updated_at = now()
           WHERE id = %s
           RETURNING id""",
        (video_key, duration, ref),
    ).fetchone()
    if landed is None:
        # The memory went away mid-assembly and took the row with it. Nothing else will ever name
        # this key, so it is taken back out — same as a preview that outlives its item.
        log.info("montage %s went away mid-assembly, taking its video back", ref)
        s3.delete_object(Bucket=media.bucket(), Key=video_key)
        return
    log.info("montage %s is ready: %s clips, %.1fs", ref, len(clips), duration or 0)
    _tell_the_api(ref)


def give_up(conn, ref, error_text):
    """Out of retries. The montage is failed too, so the organizer stops waiting on it and gets
    «повторить» under it (architecture.md, «Обработка ошибок»).

    `error_text` is shown to people — the api hands the whole row to every member of the memory —
    so it is a sentence about the video, never an exception's text.
    """
    conn.execute(
        "UPDATE montages SET status = 'failed', error_text = %s, updated_at = now() WHERE id = %s",
        (error_text, ref),
    )


def _give_up_for_good(conn, ref, why):
    """A refusal rather than an accident: the answer would be the same on all three attempts, so
    the montage is failed here and the job finishes cleanly instead of retrying into it."""
    log.info("montage %s cannot be made: %s", ref, why)
    give_up(conn, ref, why)


def _tell_the_api(montage_id):
    """«Готов, скажи людям». The row is already right — this call only reaches the phones, which
    the worker cannot do itself, so a knock nobody answers is a warning and not a failure."""
    url, token = os.environ.get("API_URL"), os.environ.get("INTERNAL_WORKER_TOKEN")
    if not (url and token):
        log.warning("API_URL/INTERNAL_WORKER_TOKEN unset: montage %s is ready and nobody is told",
                    montage_id)
        return
    request = urllib.request.Request(
        f"{url.rstrip('/')}/internal/montages/{montage_id}/ready",
        data=json.dumps({}).encode(), method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
            response.read()
    except Exception as exc:
        # Raising here would retry the whole job — rebuilding a video that already exists, for a
        # push. The organizer sees the montage in the list either way, it is only the push that
        # is lost.
        log.warning("could not tell the api that montage %s is ready: %s", montage_id, exc)
