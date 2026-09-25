"""Склейка ролика по частям: память не растёт с числом кусков, а ролик тот же.

См. `montage/video/assembly.py` и running.md, «Образ». Нужен только ffmpeg: куски — одноцветные
кадры на холсте-марке 128×72, и по цвету кадра видно, чей кусок где стоит.
"""

import subprocess

import pytest

from montage import ffmpeg
from montage.video import assembly, selection

from checks import near, pixel_at
from material import draw, probe, tone

COLOURS = {
    "red": (255, 0, 0),
    "lime": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "magenta": (255, 0, 255),
    "cyan": (0, 255, 255),
}
# Not whole frames on purpose: 2.53 seconds is 75.9 of them, and a join placed against the one
# before it rather than against the first frame would drift a little further with every piece.
SECONDS = (2.0, 2.53, 1.5, 3.0, 2.52, 2.2)


@pytest.fixture
def stills(tmp_path, monkeypatch):
    monkeypatch.setitem(assembly.CANVASES, "landscape", (128, 72))
    pieces = []
    for name, seconds in zip(COLOURS, SECONDS):
        path = tmp_path / f"{name}.jpg"
        draw("-f", "lavfi", "-i", f"color=c={name}:s=320x180", "-frames:v", "1", str(path))
        pieces.append(selection.Piece(kind="photo", path=path, seconds=seconds))
    return pieces


def music(tmp_path, wanted):
    return tone(tmp_path / "track.wav", seconds=20) if wanted else None


@pytest.mark.parametrize("with_music", [True, False], ids=["под музыку", "с живым звуком"])
def test_no_ffmpeg_of_a_montage_opens_more_than_two_of_its_clips(
    stills, tmp_path, monkeypatch, with_music
):
    """Every clip an ffmpeg is handed stays open and decoding to the end, at 80–100 МиБ apiece: one
    call over all the clips of a minute-long montage took 3 ГиБ, and a pod gets two. So a montage is
    put together a clip at a time, and this goes red the day one call is handed them all again."""
    opened = []
    run = ffmpeg.run

    def counted(*args, **kwargs):
        inputs = [args[at + 1] for at, arg in enumerate(args) if arg == "-i"]
        opened.append(sum(str(tmp_path / "clip-") in one for one in inputs))
        return run(*args, **kwargs)

    monkeypatch.setattr(ffmpeg, "run", counted)
    assembly.assemble(
        stills, "landscape", music(tmp_path, with_music), tmp_path, tmp_path / "montage.mp4"
    )

    assert max(opened) == 2, f"одному ffmpeg отдали {max(opened)} куска из {len(stills)}"


@pytest.mark.parametrize("with_music", [True, False], ids=["под музыку", "с живым звуком"])
def test_a_montage_put_together_by_parts_is_the_one_chain_of_its_clips(
    stills, tmp_path, with_music
):
    """What the parts add up to is what one crossfade chain over every clip made: as long as the
    clips less half a second a join, to the frame; every clip in its place and in its order; every
    join a dissolve half way through where the plan put it — the pieces here are no whole number of
    frames, so a part cut against the one before it rather than against the first frame would walk
    the later joins off; and the sound as long as the picture."""
    out = tmp_path / "montage.mp4"

    clips = assembly.assemble(stills, "landscape", music(tmp_path, with_music), tmp_path, out)

    length = assembly.length_of(clips)
    streams = {stream["codec_type"]: stream for stream in probe(out)["streams"]}
    assert float(streams["video"]["duration"]) == pytest.approx(length, abs=1 / assembly.FPS)
    assert float(streams["audio"]["duration"]) == pytest.approx(length, abs=0.05)
    starts = [0.0]
    for clip in clips[:-1]:
        starts.append(starts[-1] + clip.seconds - assembly.CROSSFADE_SECONDS)
    colours = list(COLOURS.values())
    for index, start in enumerate(starts):
        alone_from = start + (assembly.CROSSFADE_SECONDS if index else 0)
        alone_to = starts[index + 1] if index + 1 < len(starts) else length
        assert near(pixel_at(out, (alone_from + alone_to) / 2, 64, 36), colours[index]), (
            f"кусок {index} не на своём месте"
        )
        if index:
            half = tuple((a + b) / 2 for a, b in zip(colours[index - 1], colours[index]))
            join = start + assembly.CROSSFADE_SECONDS / 2
            assert near(pixel_at(out, join, 64, 36), half), f"склейка {index} не там, где план"


def test_photographs_and_clips_keep_their_colours_through_the_pipe(tmp_path, monkeypatch):
    """The pieces reach the encoder as bare frames, which say nothing of their colour, while a
    photograph decodes full range with the JPEG matrix and a phone's clip limited with BT.709. Taken
    for one another, a light grey photograph plays back some thirteen levels brighter and a red one
    turns orange; the pipe carries one kind of frame and every piece is converted into it."""
    monkeypatch.setitem(assembly.CANVASES, "landscape", (128, 72))
    grey, red = (192, 192, 192), (255, 0, 0)
    draw(
        "-f", "lavfi", "-i", "color=c=0xC0C0C0:s=320x180", "-frames:v", "1", str(tmp_path / "g.jpg")
    )
    draw("-f", "lavfi", "-i", "color=c=red:s=320x180", "-frames:v", "1", str(tmp_path / "r.jpg"))
    draw(
        "-f", "lavfi", "-i", "color=c=0xC0C0C0:s=320x180:d=3", "-vf", "format=yuv420p",
        "-color_range", "tv", "-colorspace", "bt709", "-c:v", "libx264", str(tmp_path / "g.mp4"),
    )  # fmt: skip
    pieces = [
        selection.Piece(kind="photo", path=tmp_path / "g.jpg", seconds=2.0),
        selection.Piece(kind="photo", path=tmp_path / "r.jpg", seconds=2.0),
        selection.Piece(kind="video", path=tmp_path / "g.mp4", seconds=2.0),
    ]
    out = tmp_path / "montage.mp4"

    assembly.assemble(pieces, "landscape", music(tmp_path, True), tmp_path, out)

    for at, colour in ((0.5, grey), (2.5, red), (4.0, grey)):
        seen = pixel_at(out, at, 64, 36)
        assert all(abs(a - b) <= 8 for a, b in zip(seen, colour)), f"{colour} на {at} с — {seen}"


@pytest.mark.parametrize("with_music", [True, False], ids=["под музыку", "с живым звуком"])
def test_a_clip_shorter_than_a_dissolve_does_not_push_the_rest_of_the_montage_along(
    tmp_path, monkeypatch, with_music
):
    """A phone clip of a third of a second is cut whole, and the dissolve into the next piece
    starts before it is fully in: it has no stretch of its own. It is left no frames rather than
    a negative count of them, which ffmpeg takes for «no limit» and which would shift everything
    after it off its place."""
    monkeypatch.setitem(assembly.CANVASES, "landscape", (128, 72))
    draw("-f", "lavfi", "-i", "color=c=red:s=320x180", "-frames:v", "1", str(tmp_path / "r.jpg"))
    draw("-f", "lavfi", "-i", "color=c=lime:s=320x180", "-frames:v", "1", str(tmp_path / "g.jpg"))
    draw(
        "-f", "lavfi", "-i", "color=c=blue:s=320x180:d=0.3", "-f", "lavfi", "-i", "sine=d=0.3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(tmp_path / "b.mp4"),
    )  # fmt: skip
    pieces = [
        selection.Piece(kind="photo", path=tmp_path / "r.jpg", seconds=2.0),
        selection.Piece(kind="video", path=tmp_path / "b.mp4", seconds=0.3),
        selection.Piece(kind="photo", path=tmp_path / "g.jpg", seconds=2.0),
    ]
    out = tmp_path / "montage.mp4"

    clips = assembly.assemble(pieces, "landscape", music(tmp_path, with_music), tmp_path, out)

    length = assembly.length_of(clips)
    assert float(probe(out)["format"]["duration"]) == pytest.approx(length, abs=0.05)
    assert near(pixel_at(out, 0.7, 64, 36), (255, 0, 0))
    assert near(pixel_at(out, length - 0.3, 64, 36), (0, 255, 0)), "последний кусок не на месте"


def test_every_ffmpeg_of_a_montage_is_held_to_the_threads_it_was_given(
    stills, tmp_path, monkeypatch
):
    """ffmpeg left to itself counts the node's processors, not the pod's, and every thread of a
    decoder or an encoder holds frames of its own. Every one the montage runs — the encoder the
    pieces are written into as well — is told how many: each input, each filter graph, the output.
    """
    track, commands, popen = music(tmp_path, True), [], subprocess.Popen

    def recorded(args, *rest, **kwargs):
        commands.append(list(args))
        return popen(args, *rest, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", recorded)
    monkeypatch.setattr(ffmpeg, "THREADS", "3")
    assembly.assemble(stills, "landscape", track, tmp_path, tmp_path / "montage.mp4")

    ffmpegs = [command for command in commands if command[0] == "ffmpeg"]
    assert any("pipe:0" in command for command in ffmpegs), "кодировщик тоже ffmpeg"
    for command in ffmpegs:
        for option in ("-filter_threads", "-filter_complex_threads"):
            assert command[command.index(option) + 1] == "3", command
        assert command[-3:-1] == ["-threads", "3"], f"кодировщик считает процессоры: {command}"
        for at, arg in enumerate(command):
            if arg == "-i":
                assert command[at - 2 : at] == ["-threads", "3"], f"вход без предела: {command}"


def test_ffmpeg_takes_two_threads_unless_the_environment_says_otherwise(monkeypatch):
    monkeypatch.delenv("FFMPEG_THREADS", raising=False)
    assert ffmpeg._threads() == "2"
    monkeypatch.setenv("FFMPEG_THREADS", "")
    assert ffmpeg._threads() == "2", "пустое значение из чарта — это «не задано»"
    monkeypatch.setenv("FFMPEG_THREADS", "4")
    assert ffmpeg._threads() == "4"


@pytest.mark.parametrize("threads", ["0", "-1", "two", "1.5"])
def test_a_thread_count_that_is_no_count_stops_the_worker_at_start(monkeypatch, threads):
    """Read once, when the worker starts: a count ffmpeg would take for «as many as you see» (0) or
    refuse mid-montage fails the process before its first job instead."""
    monkeypatch.setenv("FFMPEG_THREADS", threads)
    with pytest.raises(ValueError, match="FFMPEG_THREADS"):
        ffmpeg._threads()
