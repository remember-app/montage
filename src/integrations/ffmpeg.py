import subprocess


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(("ffmpeg", *args), check=True, capture_output=True, text=True)
