"""Склейка ролика по частям: память не растёт с числом кусков, а ролик тот же.

См. `montage/video/assembly.py` и running.md, «Образ». Нужен только ffmpeg: куски — одноцветные
кадры на холсте-марке 128×72, и по цвету кадра видно, чей кусок где стоит.
"""

import dataclasses
import math
import subprocess

import numpy
import pytest

from montage import ffmpeg
from montage.video import assembly, selection

from checks import loudness, near, pixel_at
from material import draw, mute_clip, probe, spatial_sound_clip, tone

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


def test_the_clips_own_sound_runs_on_through_every_part_without_a_gap_or_a_jump(
    tmp_path, monkeypatch
):
    """A part's sound opens on the half second of the clip before, from where that clip's own part
    stopped. A tail even a sample shorter than the dissolve and acrossfade silences the whole part;
    one picked up a little off where the part before stopped and the sound jumps there, a click.
    One steady tone under every clip: each is heard, and no step from sample to sample is steeper
    than the tone's own."""
    monkeypatch.setitem(assembly.CANVASES, "landscape", (128, 72))
    pieces = []
    for index, seconds in enumerate((2.0, 2.53, 1.5, 3.0, 2.52)):
        path = tmp_path / f"{index}.mp4"
        draw(
            "-f", "lavfi", "-i", "color=c=gray:s=320x180:r=30:d=4", "-f", "lavfi", "-i",
            "sine=f=440:d=4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
            str(path),
        )  # fmt: skip
        pieces.append(selection.Piece(kind="video", path=path, seconds=seconds))
    out = tmp_path / "montage.mp4"

    clips = assembly.assemble(pieces, "landscape", None, tmp_path, out)

    start = 0.0
    for index, clip in enumerate(clips):
        assert loudness(out, start + clip.seconds / 2 - 0.2, 0.4) > -40, f"кусок {index} онемел"
        start += clip.seconds - assembly.CROSSFADE_SECONDS
    steps = jumps(out)
    own = steps[at(0.6) : at(1.2)].max()  # the first clip alone, faded in by then
    assert steps.max() < 2 * own, "звук скачет на стыке частей"


def jumps(path):
    """How far the sound of a file moves from one sample to the next, all the way through: a click
    is one sample far from the one before it, further than any sound the file is made of moves."""
    pcm = subprocess.run(
        ("ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-f", "s16le", "-"),
        capture_output=True,
        check=True,
    ).stdout
    return numpy.abs(numpy.diff(numpy.frombuffer(pcm, "<i2").astype(int)))


def at(second):
    """The sample of the finished sound at this second."""
    return int(second * assembly.SAMPLE_RATE)


# Живой звук клипов и музыка — два тона, которые не спутать: по частоте видно, что из них звучит.
LIVE_HZ, MUSIC_HZ = 440, 1320


def test_every_piece_is_heard_as_the_director_chose_and_the_sound_turns_over_with_the_picture(
    tmp_path, monkeypatch
):
    """Звук куска — выбор режиссёра: музыка, свой живой звук или смесь в названной доле, и в
    готовом файле слышно ровно это. Фотография и клип, снятый без звука, идут под музыку, что бы о
    них ни сказали: своего им сказать нечего, а тишины вместо музыки никто не просил.

    Звук меняется тем же полусекундным перехлёстом, что и картинка: на середине склейки слышно и
    уходящее, и приходящее, каждое вполовину, — и нигде ни щелчка: ни один шаг от сэмпла к сэмплу не
    круче, чем у самих тонов там, где они звучат ровно."""
    monkeypatch.setitem(assembly.CANVASES, "landscape", (128, 72))
    photo, sounding = tmp_path / "photo.jpg", tmp_path / "sounding.mp4"
    draw("-f", "lavfi", "-i", "color=c=red:s=320x180", "-frames:v", "1", str(photo))
    draw(
        "-f", "lavfi", "-i", "color=c=gray:s=320x180:r=30:d=4", "-f", "lavfi", "-i",
        f"sine=f={LIVE_HZ}:d=4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-shortest", str(sounding),
    )  # fmt: skip
    silent = mute_clip(tmp_path / "silent.mp4", seconds=4)
    heard = [  # (кусок, что о нём сказано, что должно звучать: доля живого звука)
        (selection.Piece(kind="photo", path=photo), 1.0, 0.0),
        (selection.Piece(kind="video", path=sounding), 1.0, 1.0),
        (selection.Piece(kind="video", path=sounding), 0.0, 0.0),
        (selection.Piece(kind="video", path=sounding), 0.5, 0.5),
        (selection.Piece(kind="video", path=silent), 1.0, 0.0),
    ]
    pieces = [dataclasses.replace(piece, seconds=3.0, live=said) for piece, said, _ in heard]
    track = tone(tmp_path / "track.wav", seconds=20, frequency=MUSIC_HZ)
    out = tmp_path / "montage.mp4"

    clips = assembly.assemble(pieces, "landscape", track, tmp_path, out)

    starts = [0.0]
    for clip in clips[:-1]:
        starts.append(starts[-1] + clip.seconds - assembly.CROSSFADE_SECONDS)
    # Каждый кусок — там, где он звучит один: после склейки в него и до следующей, в стороне от
    # фейдов музыки на краях ролика. Громкость полного тона — по кускам, где он звучит один.
    alone = [starts[1] - 0.7] + [start + assembly.CROSSFADE_SECONDS + 0.1 for start in starts[1:]]
    live = [loudness(out, second, 0.6, LIVE_HZ) for second in alone]
    music = [loudness(out, second, 0.6, MUSIC_HZ) for second in alone]
    full_live, full_music = live[1], music[2]
    assert full_live > -40 and full_music > -40, "живой звук или музыка не слышны вовсе"
    for index, (_, said, share) in enumerate(heard):
        for what, level, full, wanted in (
            ("живой звук", live[index], full_live, share),
            ("музыка", music[index], full_music, 1 - share),
        ):
            expected = full + 20 * math.log10(wanted) if wanted else full - 30
            assert abs(level - expected) < 2 if wanted else level < expected, (
                f"кусок {index} (сказано {said}): {what} {level:.1f} дБ, "
                f"а при доле {wanted} — {'около' if wanted else 'ниже'} {expected:.1f}"
            )

    # Посреди склейки, где музыка сменяется живым звуком и обратно, слышно и уходящее, и
    # приходящее, и ни то ни другое не в полную силу: звук меняется вместе с картинкой, а не до
    # неё и не после.
    for index in (1, 2):
        middle = starts[index] + assembly.CROSSFADE_SECONDS / 2 - 0.05
        for what, hz, full in (
            ("живой звук", LIVE_HZ, full_live),
            ("музыка", MUSIC_HZ, full_music),
        ):
            level = loudness(out, middle, 0.1, hz)
            assert full - 16 < level < full - 3, (
                f"склейка {index}: {what} {level:.1f} дБ посреди перехлёста — звук сменился не "
                f"вместе с картинкой"
            )

    steps = jumps(out)
    steady = max(steps[at(second) : at(second + 0.6)].max() for second in alone)
    assert steps.max() < 1.5 * steady, "на стыке звук прыгает — щелчок"


def test_a_montage_that_turns_its_sound_at_every_join_is_still_an_expression_ffmpeg_takes():
    """Парсер выражений ffmpeg уходит на уровень глубже с каждым `+` подряд и отказывается после
    сотни, а длинный ролик, где звук меняется на каждой склейке, — это сотни смен. Отказ здесь — это
    упавшая сборка ролика после минут кодирования."""
    clips = [selection.Piece(live=float(index % 2)) for index in range(400)]
    share = assembly._live_share(clips, [index * assembly.FPS for index in range(len(clips) + 1)])

    ffmpeg.run(
        "-f", "lavfi", "-i", "sine=d=1", "-af", f"volume=eval=frame:volume='{share}'",
        "-f", "null", "-",
    )  # fmt: skip


def test_a_clip_with_a_second_sound_track_the_phone_added_is_taken_by_its_first(
    tmp_path, monkeypatch
):
    """iPhone пишет в MOV рядом с обычной дорожкой вторую, пространственную, чей кодек ffmpeg не
    знает. Взять из такого файла все дорожки — значит не получить даже куска: ffmpeg отказывает
    целому вызову, и ролика нет вовсе, что и случилось на живом заказе (TASK-460). Берётся первая, и
    кусок идёт в ролик своим звуком, а не тишиной. `ffmpeg.has_sound` от этого не спасает и
    спасти не может: `a:0` в файле есть."""
    monkeypatch.setitem(assembly.CANVASES, "landscape", (128, 72))
    piece = selection.Piece(
        kind="video", path=spatial_sound_clip(tmp_path / "iphone.mov"), seconds=2.0
    )
    out = tmp_path / "montage.mp4"

    assembly.assemble([piece], "landscape", None, tmp_path, out)

    kinds = [stream["codec_type"] for stream in probe(out)["streams"]]
    assert kinds.count("audio") == 1, f"в ролике дорожек: {kinds}"
    assert loudness(out, 0.6, 0.8) > -40, "кусок онемел: вместо своего звука в ролик ушла тишина"


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


def test_two_stretches_of_one_clip_each_show_their_own(tmp_path, monkeypatch):
    """Один файл стоит в ролике дважды — двумя отрезками, которые назвал режиссёр: каждый кусок
    режется из файла по своему `start`. Клип красный первые пять секунд и синий следующие пять;
    отрезок 6–8 стоит первым, 1–3 — вторым, и ролик идёт сперва синим, потом красным."""
    monkeypatch.setitem(assembly.CANVASES, "landscape", (128, 72))
    clip = tmp_path / "clip.mp4"
    draw(
        "-f", "lavfi", "-i", "color=c=red:s=320x180:r=30:d=10",
        "-vf", "drawbox=w=iw:h=ih:c=blue:t=fill:enable='gte(t,5)'",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip),
    )  # fmt: skip
    pieces = [
        selection.Piece(kind="video", path=clip, start=start, seconds=2.0, duration=10.0)
        for start in (6.0, 1.0)
    ]
    out = tmp_path / "montage.mp4"

    assembly.assemble(pieces, "landscape", music(tmp_path, True), tmp_path, out)

    assert near(pixel_at(out, 0.75, 64, 36), COLOURS["blue"]), "первый кусок — не отрезок 6–8"
    assert near(pixel_at(out, 2.75, 64, 36), COLOURS["red"]), "второй кусок — не отрезок 1–3"


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


def test_ctrl_c_in_the_middle_of_a_montage_stays_an_interrupt(tmp_path):
    """Ctrl-C reaches every ffmpeg of the terminal, the encoder too, and it exits complaining. An
    interrupt hands the job back to the queue (`worker.run_once`); the encoder's failure raised
    over it would count an attempt against a job that did not fail. Here the encoder fails anyway,
    with nowhere to write."""
    with pytest.raises(KeyboardInterrupt):
        with ffmpeg.encoding(assembly.RAW, str(tmp_path / "missing" / "picture.mp4")):
            raise KeyboardInterrupt


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
