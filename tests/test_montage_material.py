"""Что попадает в ролик, а что нет: «исключить», равные кадры и пол с потолком.

См. `docs/jobs/montage.md`. Всё живое: настоящий Postgres, бакет и ffmpeg.
"""

import uuid

import pytest

from montage import ffmpeg, storage, worker
from montage.video import assembly, montage, selection

from checks import about, near, pixel_at
from material import FIXTURES, click_track, crushed, draw, jpeg, lopsided, probe, tone, unopenable
from rows import add_item, entries_of, fetch, layout_of, order, read_montage, ready_material


def test_a_file_that_will_not_open_is_left_out_rather_than_failing_the_video(
    conn, s3, memory, tmp_path
):
    """One corrupt upload is that file's problem, not the memory's. It used to raise straight out
    of the handler: three retries, then a montage failed for good — and «повторить» met the same
    file, so any one member could deny the organizer a video permanently."""
    montage_id = order(conn, memory, {"target_seconds": 30})
    ready_material(conn, s3, memory, tmp_path)
    _, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(unopenable(tmp_path / "broken")), storage.client().bucket, key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assert about(duration, 9.0), "the four that opened went in, the broken one was left out"


def test_a_badly_exposed_photo_is_in_the_video_rather_than_failing_it(conn, s3, memory, tmp_path):
    """A different code path from a file ffmpeg refuses to open: this one decodes fine, and the
    only frame in it is crushed almost to black. Отсева внутри сборки больше нет, и такой кадр стоит
    в ролике наравне с остальными — решение человека, оставившего его при выборе (doc-03, «What it
    doesn't»). Единственное, что в сборку не попадает, — файл, который не открывается вовсе."""
    montage_id = order(conn, memory, {"target_seconds": 30})
    ready_material(conn, s3, memory, tmp_path)
    crushed_id, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(crushed(tmp_path / "crushed.jpg")), storage.client().bucket, key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    # Три снимка по 2.5, пересвеченный снимок по 2.5 и трёхсекундный клип, минус полсекунды на
    # каждую из четырёх склеек.
    assert about(duration, 11.0), "пересвеченный кадр в ролике, а не выброшен за брак"
    assert crushed_id in {media_id for media_id, _, _ in layout_of(conn, montage_id)}


def test_the_floor_counts_every_frame_the_organizer_left_alike(conn, s3, memory, tmp_path):
    """Пол — это «сколько кусков реально войдёт», и кадры, оставленные при выборе, равны: из
    одного нормального снимка и двух пересвеченных ролик собирается, а не отказывается. Раньше
    отказ был возможен, если материал отсевал почти целиком."""
    _, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(jpeg(tmp_path / "good.jpg")), storage.client().bucket, key)
    for index in range(2):
        _, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(
            str(crushed(tmp_path / f"crushed-{index}.jpg")), storage.client().bucket, key
        )
    montage_id = order(conn, memory, {"target_seconds": 30})
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assert about(duration, 6.5), "all three went in: three stills, less half a second for each join"


def test_no_montage_runs_past_the_ceiling_whatever_the_organizer_asked_for_and_the_beats_did(
    conn, s3, memory, tmp_path, monkeypatch
):
    """The ten-minute edge on a real file, at a ceiling lowered to eight seconds: what is asked is
    whether the edge holds, not whether ten minutes can be encoded. Five stills come to 10.5
    seconds, and a fast metronome pulls their joins about on top of that. The video is cut from its
    tail to the ceiling — the stills that stay are the first ones, in their order — and the file
    ffprobe reads runs no longer than it."""
    monkeypatch.setattr(selection, "MAX_SECONDS", 8.0)
    track_key = f"music/{uuid.uuid4()}.wav"
    s3.upload_file(
        str(click_track(tmp_path / "click.wav", gap=0.43)), storage.client().bucket, track_key
    )
    ids = []
    for index in range(5):
        item_id, key = add_item(conn, memory, "photo", status="ready", age_hours=5 - index)
        s3.upload_file(str(jpeg(tmp_path / f"photo-{index}.jpg")), storage.client().bucket, key)
        ids.append(item_id)
    montage_id = order(conn, memory, {"target_seconds": 30, "music_track_key": track_key})
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    seconds = float(probe(fetch(s3, video_key, tmp_path / "m.mp4"))["format"]["duration"])
    assert duration <= 8.0 and seconds <= 8.0, f"the file runs {seconds}s past the ceiling"
    assert seconds == pytest.approx(8.0, abs=0.1), "cut short of the ceiling rather than to it"
    layout = [media_id for media_id, _, _ in layout_of(conn, montage_id)]
    assert layout == ids[: len(layout)] and len(layout) < len(ids), "not cut from the tail"


def test_a_photo_the_organizer_marked_exclude_stays_out_of_the_video(conn, s3, memory, tmp_path):
    """The other half of the same word from the organizer, on material the algorithm is perfectly
    happy with: a photo named in «исключить» is not in the video, and the rest of the material
    fills the time it would have taken."""
    photo_id = ready_material(conn, s3, memory, tmp_path)[0]
    montage_id = order(conn, memory, {"target_seconds": 30, "exclude_media_ids": [photo_id]})
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    # Two stills and the clip — eight seconds of material, less half a second for each of two
    # joins.
    assert about(duration, 7.0), "one of the four pieces was left out on the order's own word"


def test_a_photo_its_owner_hid_stays_out_of_the_video(conn, s3, memory, tmp_path):
    """Hiding is the owner's word, и никакое слово заказа его не отменяет (TASK-424): скрытый снимок
    не материала вовсе, и в ролике его нет — ни в файле, ни в раскладке."""
    hidden_id = ready_material(conn, s3, memory, tmp_path)[0]
    conn.execute("UPDATE media_items SET hidden = true WHERE id = %s", (hidden_id,))
    montage_id = order(conn, memory, {"target_seconds": 30})
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assert about(duration, 7.0), "two stills and the clip, the hidden photo left out"
    assert hidden_id not in {media_id for media_id, _, _ in layout_of(conn, montage_id)}


def test_a_stale_include_in_the_order_no_longer_means_anything(conn, s3, memory, tmp_path):
    """Приложение и api ещё присылают `include_media_ids` в заказе, а воркер их больше не читает:
    кадры, оставленные при выборе, равны, и «обязательно включить» ничего не закрепляет (doc-03,
    «What it doesn't»). Заказ с этим словом собирается ровно так же, как без него."""
    ids = ready_material(conn, s3, memory, tmp_path)
    montage_id = order(conn, memory, {"target_seconds": 30, "include_media_ids": [ids[0]]})
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assert about(duration, 9.0), "все четыре куска на месте, отмеченный — среди прочих"
    assert set(ids) == {media_id for media_id, _, _ in layout_of(conn, montage_id)}
    assert not [one for one in entries_of(conn, montage_id) if one.reason == "included"]


def test_the_order_cuts_a_photograph_from_the_same_copy_the_proposal_judged(
    conn, s3, memory, tmp_path
):
    """Which file заказ смотрит, когда ролик заказан, — видно снаружи единственным способом: большая
    копия рядом с этим снимком чёрная, а оригинал под ней — картинка.

    Кадры равны, и чёрная копия теперь не отсеивается, а попадает в ролик как есть: кусок режется из
    той же копии, на которой смотрел кадр (`_back_to_originals`). Значит, и в готовом файле на месте
    этого кадра чёрное — судил оригинал, была бы картинка.
    """
    photo_id = ready_material(conn, s3, memory, tmp_path)[0]
    black = tmp_path / "black.jpg"
    draw("-f", "lavfi", "-i", "color=black:size=2048x1366", "-frames:v", "1", str(black))
    s3.upload_file(str(black), storage.client().bucket, f"previews/{photo_id}.jpg")
    conn.execute(
        "UPDATE media_items SET preview_key = %s WHERE id = %s",
        (f"previews/{photo_id}.jpg", photo_id),
    )
    montage_id = order(conn, memory, {"target_seconds": 30})
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assert about(duration, 9.0), "все четыре куска на месте, чёрный снимок в том числе"
    where = {media_id: start for media_id, start, _ in layout_of(conn, montage_id)}
    video = fetch(s3, video_key, tmp_path / "montage.mp4")
    assert near(pixel_at(video, where[photo_id] + 0.5, 640, 360), (0, 0, 0)), (
        "снимок с чёрной копией остался в ролике чёрным — судили оригинал"
    )


def test_the_floor_does_not_count_the_pieces_the_organizer_struck_out(conn, s3, memory, tmp_path):
    """Four files the отсев is happy with, two of them named in «исключить». The floor is what can
    actually end up on screen, and `selection.choose` throws the struck-out two away whatever the
    floor thought — so a minute-long order backed by two pieces is refused here rather than
    assembled into a seven-second «ролик»."""
    struck = ready_material(conn, s3, memory, tmp_path)[:2]
    montage_id = order(conn, memory, {"target_seconds": 60, "exclude_media_ids": struck})
    job_id = worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "done"
    status, video_key, _, error_text = read_montage(conn, montage_id)
    assert (status, video_key) == ("failed", None), "nothing was assembled out of the two left"
    assert "не годится" in error_text


def test_the_floor_does_not_count_what_the_organizer_struck_out(conn, s3, memory, tmp_path):
    """«Исключить» — слово, которое нельзя отменить, когда ролик уже собран, и пол считает именно
    то, что может попасть на экран: три нормальных снимка, один из них вычеркнут, — это два куска,
    и ролик из двух не собирается. Слова «включить» больше нет, и вычеркнутый им не вернуть."""
    ids = []
    for index in range(3):
        item_id, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(jpeg(tmp_path / f"photo-{index}.jpg")), storage.client().bucket, key)
        ids.append(item_id)
    montage_id = order(conn, memory, {"target_seconds": 60, "exclude_media_ids": [ids[0], ids[1]]})
    job_id = worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "done"
    status, video_key, _, error_text = read_montage(conn, montage_id)
    assert (status, video_key) == ("failed", None), (
        "the contradiction did not save it from the floor"
    )
    assert "не годится" in error_text


def test_too_little_material_that_actually_opens_fails_without_retrying(conn, s3, memory, tmp_path):
    """The floor is about material that can go in the video, not rows in the table: three rows of
    which two are unreadable would otherwise have been assembled into one three-second still."""
    montage_id = order(conn, memory, {})
    _, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(jpeg(tmp_path / "good.jpg")), storage.client().bucket, key)
    for index in range(montage.MIN_ITEMS - 1):
        _, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(unopenable(tmp_path / f"broken-{index}")), storage.client().bucket, key)
    job_id = worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "done"
    status, _, _, error_text = read_montage(conn, montage_id)
    assert status == "failed"
    assert "не годится" in error_text


def test_a_clip_is_laid_into_the_chain_at_the_length_it_really_is(conn, s3, memory, tmp_path):
    """The crossfade offsets step along by each clip's length. Booking a clip in at the length it
    was *asked* for, when its picture is shorter than its container says, walks the rest of the
    chain off the end — and ffmpeg truncates the montage without a word of complaint."""
    montage_id = order(conn, memory, {"target_seconds": 30})
    # Oldest, so the chronological order puts it first and the drift hits every join after it.
    _, key = add_item(conn, memory, "video", status="ready", age_hours=5)
    s3.upload_file(str(lopsided(tmp_path / "lopsided.mp4")), storage.client().bucket, key)
    for index in range(3):
        _, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(jpeg(tmp_path / f"photo-{index}.jpg")), storage.client().bucket, key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, duration, _ = read_montage(conn, montage_id)
    assert status == "ready"
    # Two seconds of picture and three stills of 2.5, less half a second for each of three
    # joins: 8. Booked at the container's 3.5 instead, ffmpeg truncates the chain and the whole
    # montage comes out at three seconds — nowhere near this window, slack for the rhythm or not.
    assert about(duration, 8.0), "the clip went in at its real two seconds"
    assert about(float(probe(fetch(s3, video_key, tmp_path / "m.mp4"))["format"]["duration"]), 8.0)


def test_a_photograph_is_scaled_once_for_the_piece_and_not_once_a_frame(tmp_path, monkeypatch):
    """The root of TASK-183: looping the *input* (`-loop 1`) hands the file back to the decoder for
    every frame of the clip, so a 46-megapixel snapshot was decoded and scaled eighty-five times
    over for three seconds of video — seventeen minutes for one piece on the stand, with the whole
    montage behind it. The frame is held after the crop instead, by the filter.

    Checked on the command because that is where the decoding is decided, and the cost of the
    difference is hardware: the same three seconds of video comes out of both, one of them fifty
    times slower. The clip itself is made for real below — an infinite loop that `-t` fails to stop
    would hang here rather than in somebody's montage.
    """
    commands = []
    monkeypatch.setattr(ffmpeg, "run", lambda *args: commands.append(args))
    held = selection.Piece(kind="photo", seconds=3.0, path=tmp_path / "photo.jpg")
    assembly._normalise(held, "landscape", tmp_path / "clip.mp4")
    (command,) = commands

    assert "-loop" not in command, "исходник зациклен — фото декодируется заново на каждом кадре"
    filters = command[command.index("-vf") + 1]
    assert filters.count("scale=") == 1, "фото масштабируется больше одного раза"
    assert filters.index("loop=") > filters.index("crop="), "удерживается не обрезанный кадр"

    monkeypatch.undo()
    jpeg(tmp_path / "photo.jpg")
    assembly._normalise(held, "landscape", tmp_path / "clip.mp4")
    clip = probe(tmp_path / "clip.mp4")
    assert float(clip["format"]["duration"]) == pytest.approx(3.0, abs=0.05), "не та длина куска"
    picture = clip["streams"][0]
    assert (picture["width"], picture["height"]) == assembly.CANVASES["landscape"]


def test_a_plan_that_ends_on_the_ceiling_does_not_come_out_frames_past_it(tmp_path, monkeypatch):
    """Every clip is encoded to the nearest frame of what it was asked for, and the beats ask for
    lengths that are no whole number of frames: 2.52 seconds is 75.6 of them and comes out as 76.
    Five such stills planned to end on a ceiling of 10.6 seconds come out two frames past it, and
    those frames are cut off the file rather than left in it. The canvas is a stamp, as above."""
    monkeypatch.setitem(assembly.CANVASES, "landscape", (128, 72))
    still = jpeg(tmp_path / "still.jpg")
    pieces = [selection.Piece(kind="photo", seconds=2.52, path=still) for _ in range(5)]
    out = tmp_path / "capped.mp4"

    clips = assembly.assemble(
        pieces,
        "landscape",
        tone(tmp_path / "track.wav", seconds=20),
        tmp_path,
        out,
        max_seconds=10.6,
    )

    assert assembly.length_of(clips) > 10.6, "the frames did not add up past the ceiling"
    assembled = probe(out)
    assert float(assembled["format"]["duration"]) <= 10.6
    assert {stream["codec_type"] for stream in assembled["streams"]} == {"video", "audio"}


def test_a_photograph_is_cut_from_the_large_copy_it_was_judged_on_unless_it_is_short(tmp_path):
    """The montage is cut from the 2048px copy the preview already made — the same file the отсев
    looked at — and not from the original off somebody's camera: the same picture on the canvas out
    of a tenth of the bytes. The copy's cap is on its long side, so whether it covers the canvas is
    measured rather than assumed: the very same file is enough for a landscape montage and short
    for a portrait one, and a photo whose copy never got made was judged on the original anyway.
    """
    original = tmp_path / "original"
    copy = jpeg(tmp_path / "large.jpg", size="2048x1366")
    judged_on_the_copy = selection.Piece(media_id="one", kind="photo", path=copy)
    judged_on_the_original = selection.Piece(media_id="two", kind="photo", path=original)
    originals = {"one": original, "two": original}

    (landscape,) = montage._back_to_originals(
        [judged_on_the_copy], originals, "landscape", tmp_path
    )
    (portrait,) = montage._back_to_originals([judged_on_the_copy], originals, "portrait", tmp_path)
    (alone,) = montage._back_to_originals(
        [judged_on_the_original], originals, "landscape", tmp_path
    )

    assert landscape.path == copy, "ролик режется не из копии"
    assert portrait.path == original, "1366 в высоту мало для холста 1080×1920 — нужен оригинал"
    assert alone.path == original


@pytest.mark.parametrize("copy", ["no copy", "a copy made before previews stood HEIC up"])
def test_a_portrait_heic_stands_up_in_the_montage(conn, s3, memory, tmp_path, copy):
    """What an iPhone holds upright is a sensor frame with an `irot` note, which ffmpeg does not
    apply: cut from that file, or from a large copy made while previews did not either, the photo
    lies on its side in the video. `portrait-photo.heic` is 96×64 on the sensor with red in its
    top-left corner and blue in its bottom-right; seen upright it is 64×96, red top-right and blue
    bottom-left. On the 1920×1080 canvas it is scaled to the width and cropped to its middle band,
    which leaves a thin red strip at the top right and a thin blue one at the bottom left — and
    turned on its side, a big red block at the top left instead.
    """
    ids = []
    for index in range(2):
        item_id, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(jpeg(tmp_path / f"photo-{index}.jpg")), storage.client().bucket, key)
        ids.append(item_id)
    heic_id, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(FIXTURES / "portrait-photo.heic"), storage.client().bucket, key)
    if copy != "no copy":
        # What the old preview made of it: ffmpeg's own reading, on its side, and big enough for the
        # canvas to be cut from without going back to the original.
        sideways = tmp_path / "sideways.jpg"
        draw("-i", str(FIXTURES / "portrait-photo.heic"), "-vf", "scale=2048:1365", str(sideways))
        s3.upload_file(str(sideways), storage.client().bucket, f"previews/{heic_id}.jpg")
        conn.execute(
            "UPDATE media_items SET preview_key = %s WHERE id = %s",
            (f"previews/{heic_id}.jpg", heic_id),
        )
    montage_id = order(conn, memory, {"target_seconds": 30, "music_tracks": []})
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, _, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    video_file = fetch(s3, video_key, tmp_path / "montage.mp4")
    ((start, seconds),) = [
        (start, seconds)
        for media_id, start, seconds in layout_of(conn, montage_id)
        if media_id == heic_id
    ]
    middle = start + seconds / 2  # clear of the crossfades on both sides
    assert near(pixel_at(video_file, middle, 1560, 20), (255, 0, 0)), (
        "красной полосы справа вверху нет"
    )
    assert near(pixel_at(video_file, middle, 360, 1050), (0, 0, 255)), (
        "синей полосы слева внизу нет"
    )
    assert not near(pixel_at(video_file, middle, 300, 100), (255, 0, 0)), "кадр лежит на боку"
