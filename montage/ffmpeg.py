import contextlib
import json
import logging
import os
import subprocess
import tempfile

PLAYLIST_FORMATS = {"hls", "applehttp", "concat", "ffconcat", "dash", "sdp", "rtsp", "rtp"}
PROBE_SECONDS = 60

log = logging.getLogger("montage.ffmpeg")


def _threads():
    threads = os.environ.get("FFMPEG_THREADS") or "2"
    if not threads.isdigit() or int(threads) < 1:
        raise ValueError(f"FFMPEG_THREADS must be a whole number from 1, not {threads!r}")
    return threads


THREADS = _threads()


def _held(args):
    *inputs_and_options, output = args
    held = ["-filter_threads", THREADS, "-filter_complex_threads", THREADS]
    for arg in inputs_and_options:
        if arg == "-i":
            held += ["-threads", THREADS]
        held.append(arg)
    return (*held, "-threads", THREADS, output)


def _ffmpeg(*args):
    return ("ffmpeg", "-nostdin", "-loglevel", "error", "-y", *_held(args))


def _tool(*args, timeout=None, into=None):
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
    _tool(*_ffmpeg("-protocol_whitelist", "file", *args), into=into)


def raw(*args):
    done = subprocess.run(_ffmpeg("-protocol_whitelist", "file", *args), capture_output=True)
    if done.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {done.stderr.decode(errors='replace').strip()[-500:]}")
    return done.stdout


@contextlib.contextmanager
def encoding(frames, *args):
    with tempfile.TemporaryFile() as said:
        encoder = subprocess.Popen(
            _ffmpeg("-protocol_whitelist", "pipe", *frames, "-i", "pipe:0", *args),
            stdin=subprocess.PIPE,
            stderr=said,
        )
        interrupted = False
        try:
            yield encoder.stdin
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            encoder.stdin.close()
            if encoder.wait() != 0 and not interrupted:
                said.seek(0)
                raise RuntimeError(
                    f"ffmpeg failed: {said.read().decode(errors='replace').strip()[-500:]}"
                )


def probe(entry, path, stream=None):
    asked = (
        ("-select_streams", stream, "-show_entries", f"stream={entry}")
        if stream
        else ("-show_entries", f"format={entry}")
    )
    return _tool(
        "ffprobe",
        "-v",
        "error",
        "-protocol_whitelist",
        "file",
        *asked,
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
        timeout=PROBE_SECONDS,
    ).strip()


def format_tags(path):
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


def duration(path, stream=None):
    try:
        return float(probe("duration", path, stream))
    except ValueError:
        return None


def dimensions(path):
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
        return None
    if width < 1 or height < 1:
        return None
    upended = any(
        abs(side.get("rotation", 0)) % 180 == 90 for side in stream.get("side_data_list", ())
    )
    return (height, width) if upended else (width, height)


def has_sound(path):
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
    named = {name for name in probe("format_name", path).split(",") if name}
    playlists = named & PLAYLIST_FORMATS
    if playlists:
        raise RuntimeError(f"refusing {'/'.join(sorted(playlists))}: a playlist is not media")
