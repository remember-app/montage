"""Чем тесты смотрят на то, что получилось: длина, громкость и цвет пикселя.

Ни базы, ни бакета: файл уже лежит на диске, вопрос только в том, что в нём.
"""

import re
import subprocess

# Насколько ролик может разойтись с заказанной длиной: ритм тянет склейки к битам,
# и последний кусок добирает остаток (`video/rhythm.py`).
RHYTHM_SLACK = 1.0


def about(duration, expected):
    """The montage came out about as long as it should, with room for the pull towards the beat."""
    return abs(duration - expected) <= RHYTHM_SLACK


def loudness(path, at, seconds=1.2):
    """Насколько громко звучит кусочек файла: `mean_volume` в децибелах, как его считает ffmpeg.
    Тишина — около −91 дБ (или −inf), живой звук и музыка — на десятки децибел громче."""
    done = subprocess.run(
        (
            "ffmpeg",
            "-nostdin",
            "-y",
            "-ss",
            str(at),
            "-t",
            str(seconds),
            "-i",
            str(path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ),
        capture_output=True,
        text=True,
        check=True,
    )
    (mean,) = re.findall(r"mean_volume: (\S+) dB", done.stderr)
    return float(mean)


def near(pixel, colour):
    return all(abs(a - b) < 40 for a, b in zip(pixel, colour))


def pixel_at(video, at, x, y):
    """The colour at (x, y) of a finished video's frame at `at` seconds, at full size. Cut out two
    pixels square and averaged down to one: a yuv420 frame cannot be cropped to an odd size."""
    return tuple(
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-loglevel",
                "error",
                "-ss",
                f"{at:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-vf",
                f"crop=2:2:{x}:{y},scale=1:1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ],
            capture_output=True,
            check=True,
        ).stdout
    )
