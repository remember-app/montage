"""Сборка: the chosen pieces laid onto one canvas and cut into one file.

Everything in here is ffmpeg, and nothing in here asks why these pieces and not others. It is
handed a list of pieces that already know their length, where in their own file each is cut from
and where in the video it goes; it normalises every one of them onto the montage's canvas and
crossfades them end to end with the track over the top. Sharpness, exposure, bursts, the
organizer's two lists, a plan from a neural network — all of that is `selection`'s, and how long a
piece is held so the join lands on a beat is `rhythm`'s, so the way a video is directed can change
without a line of this module moving.

A piece, as far as this module is concerned, is anything that carries `path`, `kind`, `seconds` and
`start`. That is the whole contract — it is deliberately not an import.

Every call here goes through `media.py`'s helpers: one worker, one way of running the tool.
"""

import dataclasses
import pathlib

import media

# The common canvas everything in one montage is normalised onto — the organizer picks which at the
# order (`params.orientation`). Landscape by default: the finished video leaves the app through
# «поделиться» and 16:9 is what plays everywhere; portrait is the same 1080p turned on its side, for
# a montage meant to be watched on the phone it was shot on. Either way material shot the other way
# round is padded into the chosen canvas rather than cropped to it.
CANVASES = {"landscape": (1920, 1080), "portrait": (1080, 1920)}
DEFAULT_ORIENTATION = "landscape"  # what an order that says nothing about orientation means
FPS = 30

CROSSFADE_SECONDS = 0.5
MUSIC_FADE_SECONDS = 1.5

# The library: the five tracks of the «Под что монтировать» screen
# (`design/screens/MontageMusic.html`), in its order and under its names — «Яркая», «Тёплая»,
# «Спокойная», «Ночь», «Дорога». Ids only, because the worker shows no names: which name is which
# track, who wrote it, where the file came from and under what licence is `tracks/README.md`, kept
# beside the files themselves.
TRACKS = ("bright", "warm", "calm", "night", "road")
DEFAULT_TRACK = "warm"  # what an order that says nothing about music means

TRACKS_DIR = pathlib.Path(__file__).parent / "tracks"


def track_file(track_id):
    """The library's file for a track. The organizer's own track never comes through here — that
    one arrives out of R2 and reaches `assemble` the same way, just from somewhere else."""
    return TRACKS_DIR / f"{track_id}.mp3"


def track_key(track_id):
    """Where the same track lives in the bucket, for «Послушать» on the «Под что монтировать»
    screen to play before any video is ordered. The montage itself never reads these objects — it
    cuts from the file on disk beside it — but that disk is this worker's alone, and the api and the
    phone have to be able to reach the music too. Same id in the key as in `TRACKS`, so the api
    builds it without knowing anything more about the library."""
    return f"tracks/{track_id}.mp3"


def publish_tracks(s3=None):
    """The library into the bucket. Returns the ids that were actually uploaded.

    What is already there is left alone rather than written over: this runs on deploy, a track's
    file never changes under an id, and re-uploading five mp3s on every run would only spend the
    bucket's write quota. The flip side is that a track *replaced* under an id that is already
    published needs its object deleted first — README.md, «Треки».
    """
    s3 = s3 or media.storage()
    published = []
    for track_id in TRACKS:
        try:
            s3.head_object(Bucket=media.bucket(), Key=track_key(track_id))
            continue
        except s3.exceptions.ClientError:
            pass  # not there — and if it is the bucket that is unreachable, the upload says so
        media._upload(s3, track_file(track_id), track_key(track_id), "audio/mpeg")
        published.append(track_id)
    return published


def length_of(pieces):
    """How long these pieces run once crossfaded — each join eats half a second of the total."""
    return sum(piece.seconds for piece in pieces) - CROSSFADE_SECONDS * (len(pieces) - 1)


def assemble(pieces, orientation, track, workdir, out):
    """The chosen pieces as one video at `out`, and the normalised clips it was made of.

    `track` is a file — one of the library's own or the one the organizer uploaded. Which of the two
    it is was settled before this module was called, and nothing here can tell them apart.

    The clips come back because they are the only place the lengths are true: `_normalise` is asked
    for a length and the file that comes out of it is measured rather than believed, and whoever
    cuts a cover off the montage needs those measured numbers to find its opening piece in it.
    """
    clips = []
    for index, piece in enumerate(pieces):
        clip = workdir / f"clip-{index}.mp4"
        _normalise(piece, orientation, clip)
        # Measured, not the length that was asked for. `-t` cuts against the *video* stream while
        # `_duration` reads the container, which is the longest stream in it — a clip whose audio
        # outruns its picture normalises shorter than declared. `_assemble` steps the crossfade
        # offsets along by these numbers, so a stale one walks the whole chain off the end and
        # ffmpeg truncates the montage without a word of complaint.
        clips.append(dataclasses.replace(
            piece, path=clip, seconds=media._duration(clip) or piece.seconds
        ))
    _assemble(clips, track, out)
    return clips


def _normalise(piece, orientation, out):
    """One piece onto this montage's canvas: 1080p the way it was ordered, 30 fps, padded, silent.

    The canvas is built here rather than held as a constant because it belongs to the order and not
    to the module — but every clip of one montage is normalised with the same `orientation`, which
    `_assemble`'s crossfades require: xfade joins two streams of one size or nothing.

    Silent because the finished video's only sound is the track (architecture.md: «звук исходных
    клипов по умолчанию убираем»), which also makes every clip's audio identical by having none.
    """
    width, height = CANVASES[orientation]
    canvas = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={FPS},format=yuv420p"
    )
    if piece.kind == "photo":
        opened = ("-loop", "1", "-t", f"{piece.seconds}", "-i", str(piece.path))
    else:
        opened = ("-ss", f"{piece.start}", "-t", f"{piece.seconds}", "-i", str(piece.path))
    media._ffmpeg(
        *opened, "-vf", canvas, "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", str(out),
    )


def _assemble(clips, track, out):
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
    music = len(clips)  # the track is the input after the last clip
    chain.append(
        f"[{music}:a]afade=t=in:st=0:d={fade:.3f},"
        f"afade=t=out:st={total - fade:.3f}:d={fade:.3f}[a]"
    )

    # `-protocol_whitelist` is an *input* option: ffmpeg scopes it to the group it precedes, so the
    # one `media._ffmpeg` puts at the front guards input 0 and nothing else. This is the worker's
    # only multi-input call, so it is the only place that has to repeat it — every clip and the
    # track get their own, and the whitelist means the same thing on all of them.
    inputs = [
        argument
        for clip in clips
        for argument in ("-protocol_whitelist", "file", "-i", str(clip.path))
    ]
    media._ffmpeg(
        *inputs, "-protocol_whitelist", "file",
        # A track shorter than the montage is played again from the top (`-stream_loop -1`), a
        # longer one is cut off with the picture, because `-shortest` is down below. Either way the
        # music never has to be told how long to play: the montage is cut to the material, not to
        # the track.
        "-stream_loop", "-1", "-i", str(track),
        "-filter_complex", ";".join(chain),
        # A single clip is never crossfaded, so there is no filter output to map — the input itself
        # is the video, and it is named the way ffmpeg names inputs rather than filter outputs.
        "-map", f"[{label}]" if len(clips) > 1 else "0:v", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", "-shortest", str(out),
    )
