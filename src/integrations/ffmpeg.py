import json
import pathlib
import subprocess


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(("ffmpeg", *args), check=True, capture_output=True, text=True)


def probe(path: pathlib.Path) -> dict:
    result = subprocess.run(
        (
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout or "{}")


def duration(path: pathlib.Path) -> float | None:
    value = probe(path).get("format", {}).get("duration")
    return float(value) if value else None
