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


def dimensions(path: pathlib.Path) -> tuple[int, int] | None:
    for stream in probe(path).get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        width, height = stream.get("width"), stream.get("height")
        if not width or not height:
            return None
        rotation = 0
        for side in stream.get("side_data_list", []):
            if "rotation" in side:
                rotation = int(side["rotation"])
        if rotation % 180 != 0:
            width, height = height, width
        return width, height
    return None


def format_tags(path: pathlib.Path) -> dict:
    return probe(path).get("format", {}).get("tags", {})


def has_audio(path: pathlib.Path) -> bool:
    return any(stream.get("codec_type") == "audio" for stream in probe(path).get("streams", []))


def refuse_playlists(path: pathlib.Path) -> None:
    with path.open("rb") as file:
        head = file.read(64).lstrip()
    if head.startswith(b"#EXTM3U"):
        raise ValueError(f"{path} is a playlist, not a media file")
