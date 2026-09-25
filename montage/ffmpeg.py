"""The one place ffmpeg is run: everything the worker asks a file about, and nothing else.

`ffmpeg` and `ffprobe` are two names for one tool as far as this worker is concerned — one way of
calling them, one way of turning what they say into an answer, one way of turning a failure into
an exception whose text is worth reading. Nothing here knows what a preview, a piece or a montage
is; the callers do.
"""

import contextlib
import json
import logging
import os
import subprocess
import tempfile

# Containers that are not media but a list of *other* files to go and fetch. ffmpeg picks its
# demuxer by probing the bytes, and the api never looks inside an upload — `kind` is the phone's
# word for it — so a «photo» whose bytes are an HLS playlist would have ffmpeg open whatever paths
# the playlist names and encode them into a preview the uploader can then open from the gallery.
# That is a read of this worker's own disk, which is where DATABASE_URL and the R2 keys live.
#
# A denylist and not an allowlist on purpose: format names vary by build and by camera, and being
# wrong here must mean «one exotic container slips through», never «every photo is refused».
PLAYLIST_FORMATS = {"hls", "applehttp", "concat", "ffconcat", "dash", "sdp", "rtsp", "rtp"}
PROBE_SECONDS = 60  # ffprobe reads a header; one still at it a minute on is not coming back

log = logging.getLogger("montage.ffmpeg")


def _threads():
    """How many threads one ffmpeg may run each decoder, filter graph and encoder on.

    Left to itself ffmpeg takes as many as the node has processors, not as many as the pod was
    given, and every thread of an encoder or a decoder holds frames of its own: the same montage
    took 1.1 ГиБ on four processors and 1.45 on ten (running.md, «Образ»). A number here keeps the
    memory the same on any node.
    """
    threads = os.environ.get("FFMPEG_THREADS") or "2"
    if not threads.isdigit() or int(threads) < 1:
        raise ValueError(f"FFMPEG_THREADS must be a whole number from 1, not {threads!r}")
    return threads


THREADS = _threads()


def _held(args):
    """`args` with every input's decoder, every filter graph and the encoder held to `THREADS`.

    `-threads` belongs to the input it precedes, or to the output when it is the last option before
    the file, so it goes in front of every `-i` and in front of the output — the last argument of
    every call here. The two filter options are global.
    """
    *inputs_and_options, output = args
    held = ["-filter_threads", THREADS, "-filter_complex_threads", THREADS]
    for arg in inputs_and_options:
        if arg == "-i":
            held += ["-threads", THREADS]
        held.append(arg)
    return (*held, "-threads", THREADS, output)


def _ffmpeg(*args):
    """The command line of every ffmpeg here: silent unless it fails, deaf to the keyboard and held
    to `THREADS`."""
    return ("ffmpeg", "-nostdin", "-loglevel", "error", "-y", *_held(args))


def _tool(*args, timeout=None, into=None):
    """A tool, its output, and its stderr in the exception when it fails — that text is what ends
    up in `jobs.last_error`, so it is the only account of what went wrong."""
    done = subprocess.run(
        args,
        stdout=subprocess.PIPE if into is None else into,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    if done.returncode != 0:
        raise RuntimeError(f"{args[0]} failed: {done.stderr.strip()[-500:]}")
    return done.stdout


def run(*args, into=None):
    """ffmpeg, told to write a file and to say nothing unless it failed — or, for an output named
    `-`, to write it `into` a pipe that `encoding` holds open."""
    # `-protocol_whitelist file`: whatever the input turns out to be, it may not reach the network.
    # Belt to the braces of `refuse_playlists` — that one stops the demuxers that go looking, this
    # one stops where they could look.
    _tool(*_ffmpeg("-protocol_whitelist", "file", *args), into=into)


def raw(*args):
    """The same call, with what ffmpeg wrote to stdout handed back as bytes.

    For the one caller whose *output* is the point rather than a file — the отсев, which reads the
    pixels it is judging straight off the pipe. `_tool`'s `text=True` would mangle them.
    """
    done = subprocess.run(_ffmpeg("-protocol_whitelist", "file", *args), capture_output=True)
    if done.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {done.stderr.decode(errors='replace').strip()[-500:]}")
    return done.stdout


@contextlib.contextmanager
def encoding(frames, *args):
    """An ffmpeg encoding what is written into it for as long as the block runs: the pipe to write
    into, for `run(..., into=)`.

    `frames` are the input options that say what is written — its format, size and rate — and
    `args` what becomes of it, the file last. Leaving the block closes the pipe and waits for the
    file to be finished. A failure here is raised over anything that broke the block: when the
    encoder falls over, whatever was writing into it only sees its pipe break.
    """
    with tempfile.TemporaryFile() as said:
        # Its complaints go to a file and not a pipe: a pipe nobody reads until the end would fill
        # and stop the encoder mid-frame, with the writers stopped behind it.
        encoder = subprocess.Popen(
            _ffmpeg("-protocol_whitelist", "pipe", *frames, "-i", "pipe:0", *args),
            stdin=subprocess.PIPE,
            stderr=said,
        )
        try:
            yield encoder.stdin
        finally:
            encoder.stdin.close()
            if encoder.wait() != 0:
                said.seek(0)
                raise RuntimeError(
                    f"ffmpeg failed: {said.read().decode(errors='replace').strip()[-500:]}"
                )


def probe(entry, path):
    """One `format=` field as ffprobe reports it, stripped. Empty when it does not say."""
    return _tool(
        "ffprobe",
        "-v",
        "error",
        "-protocol_whitelist",
        "file",
        "-show_entries",
        f"format={entry}",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
        timeout=PROBE_SECONDS,
    ).strip()


def format_tags(path):
    """The container's own tags, as the file carries them. Nothing is decoded for these."""
    return (
        json.loads(
            _tool(
                "ffprobe",
                "-v",
                "error",
                "-protocol_whitelist",
                "file",
                "-show_entries",
                "format_tags",
                "-of",
                "json",
                str(path),
                timeout=PROBE_SECONDS,
            )
        )
        .get("format", {})
        .get("tags", {})
    )


def duration(path):
    """Seconds as the container reports them, or None when it does not say."""
    try:
        return float(probe("duration", path))
    except ValueError:
        return None


def dimensions(path):
    """`(width, height)` of the picture as it is *seen*, or None when ffprobe does not say.

    A clip the phone was held upright for is stored on its side with a display matrix beside it
    saying to stand it up, and answers here the way it plays — taller than wide. Everything that
    decodes it sees the same picture: ffmpeg turns it on the way out unless told not to.

    A photograph's turn is written in EXIF instead, which ffprobe does not hand over as a
    rotation — those are measured by Pillow (`photo.read`).
    """
    probed = _tool(
        "ffprobe",
        "-v",
        "error",
        "-protocol_whitelist",
        "file",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height:stream_side_data=rotation",
        "-of",
        "json",
        str(path),
        timeout=PROBE_SECONDS,
    )
    try:
        (stream,) = json.loads(probed)["streams"]
        width, height = stream["width"], stream["height"]
    except (ValueError, KeyError):
        # Nothing to read the size off: no picture in the file at all, or more than one answer.
        # Callers treat that as «cannot say» and fall back on what they had.
        return None
    if width < 1 or height < 1:
        # A header ffmpeg will not believe — a size it would refuse to allocate — comes back as
        # zeros, and «0 × 0» on the sheet is no better an answer than none.
        return None
    # A quarter turn either way swaps the sides; half a turn and none leave them as they are.
    upended = any(
        abs(side.get("rotation", 0)) % 180 == 90 for side in stream.get("side_data_list", ())
    )
    return (height, width) if upended else (width, height)


def has_sound(path):
    """Whether the file carries an audio stream at all. A clip shot with the microphone off is
    common, and ffmpeg answers a request for its sound with an error rather than with silence."""
    return bool(
        _tool(
            "ffprobe",
            "-v",
            "error",
            "-protocol_whitelist",
            "file",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(path),
        ).strip()
    )


def refuse_playlists(path):
    """Stop before encoding anything if the «media» is really a list of other files to fetch.

    Raising here fails the job, which after three tries fails the item — the gallery then shows the
    placeholder it shows for anything that could not be prepared.
    """
    # ffprobe answers with a comma-separated list of every name the container goes by.
    named = {name for name in probe("format_name", path).split(",") if name}
    playlists = named & PLAYLIST_FORMATS
    if playlists:
        raise RuntimeError(f"refusing {'/'.join(sorted(playlists))}: a playlist is not media")
