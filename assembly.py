"""Сборка: the chosen pieces laid onto one canvas and cut into one file.

Everything in here is ffmpeg, and nothing in here asks why these pieces and not others. It is
handed a list of pieces that already know their length, where in their own file each is cut from
and where in the video it goes; it normalises every one of them onto the montage's canvas and
crossfades them end to end with the track over the top. Sharpness, exposure, bursts, the
organizer's two lists, a plan from a neural network — all of that is `selection`'s, so the way a
video is directed can change without a line of this module moving.

A piece, as far as this module is concerned, is anything that carries `path`, `kind`, `seconds` and
`start`. That is the whole contract — it is deliberately not an import.

Every call here goes through `media.py`'s helpers: one worker, one way of running the tool.
"""

import dataclasses

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


def length_of(pieces):
    """How long these pieces run once crossfaded — each join eats half a second of the total."""
    return sum(piece.seconds for piece in pieces) - CROSSFADE_SECONDS * (len(pieces) - 1)


def assemble(pieces, orientation, track_id, workdir, out):
    """The chosen pieces as one video at `out`, and the normalised clips it was made of.

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
    _assemble(clips, track_id, out)
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
