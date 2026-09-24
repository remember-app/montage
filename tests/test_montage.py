"""Ролик целиком: от заказа до готового файла в хранилище — см. `docs/jobs/montage.md`.

Всё живое: настоящий Postgres, настоящий бакет, настоящий ffmpeg, а режиссёр-нейросеть — записанным
ответом из `fixtures/` (`recorded.py`): в сеть здесь не ходит ничто.
"""

import json
import uuid

import pytest

from montage import ffmpeg, storage, worker
from montage.media import preview
from montage.video import assembly, frames, layout, library, montage, selection

from checks import about, loudness
from material import dark_opening_clip, hush, jpeg, long_clip, probe, tone
from recorded import planning_over
from rows import add_item, fetch, layout_of, order, read_montage, ready_material

BLACK = 16  # the value `frames._exposed` already counts as black


def poster_of(conn, montage_id):
    """The montage's cover key, `None` until a video has actually been assembled. Read on its own
    rather than added to `read_montage`: only the two tests below have anything to say about it."""
    return conn.execute("SELECT poster_key FROM montages WHERE id = %s", (montage_id,)).fetchone()[
        0
    ]


def dimmest_edge(path):
    """The dimmest edge in the whole file: the brightest pixel along each of a frame's four edges,
    and the smallest of those numbers over every second of the video.

    This is how a black bar is caught on the finished file instead of in the filter string that was
    supposed to prevent it. A letterboxed montage has a whole edge row of nothing but black, so the
    brightest pixel along it is 0; a canvas the picture fills has something in every edge. The
    frames come from `selection`'s own 96×54 grey sampling — coarse, but a bar is a third of the
    canvas, and the material here is a test pattern with its corners lit.
    """
    width = frames.SAMPLE_WIDTH
    dimmest = 255
    for frame in frames.sampled(path, per_second=frames.SAMPLE_FPS):
        rows = [frame[at : at + width] for at in range(0, len(frame), width)]
        edges = (rows[0], rows[-1], [row[0] for row in rows], [row[-1] for row in rows])
        dimmest = min(dimmest, *(max(edge) for edge in edges))
    return dimmest


def test_a_memory_of_photos_and_a_clip_becomes_one_playable_video(conn, s3, memory, tmp_path):
    montage_id = order(conn, memory, {"target_seconds": 30, "music_track_id": "warm"})
    ready_material(conn, s3, memory, tmp_path)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, duration, error_text = read_montage(conn, montage_id)
    assert (status, video_key, error_text) == ("ready", f"montages/{montage_id}.mp4", None)
    # Three stills of two and a half seconds and a four-second clip cut to the three it has — ten
    # and a half seconds of material, less half a second for each of the three crossfades. Pieces
    # butted together instead of overlapped would come to 10.5, which is outside the window even
    # with the slack. Thirty seconds were ordered and there is no more material to reach them with:
    # the clip is already whole and a photograph is never stretched (`selection.stretch`).
    assert about(duration, 9.0)

    assembled_file = fetch(s3, video_key, tmp_path / "montage.mp4")
    assembled = probe(assembled_file)
    assert about(float(assembled["format"]["duration"]), 9.0)
    tracks = {stream["codec_type"]: stream for stream in assembled["streams"]}
    assert set(tracks) == {"video", "audio"}, "picture and music both landed"
    assert tracks["video"]["codec_name"] == "h264"
    assert tracks["audio"]["codec_name"] == "aac"
    assert (tracks["video"]["width"], tracks["video"]["height"]) == (1920, 1080), (
        "an order that says nothing about orientation is the horizontal canvas, as it always was"
    )
    # And the vertical clip in that material (1280×1440) fills the horizontal canvas instead of
    # standing between two black bars: it is cropped to the canvas, not padded into it.
    assert dimmest_edge(assembled_file) > BLACK, "чёрные поля по краям горизонтального ролика"

    # And the cover beside it, on the same row — the list of videos shows a still, not a black
    # rectangle, before anyone presses play.
    poster_key = poster_of(conn, montage_id)
    assert poster_key == f"montages/{montage_id}.jpg"
    poster_file = fetch(s3, poster_key, tmp_path / "poster.jpg")
    poster = probe(poster_file)["streams"][0]
    assert (poster["codec_name"], poster["width"]) == ("mjpeg", preview.THUMBNAIL_WIDTH)
    # Not just a jpeg of the right size — a frame worth showing. The same judgement the worker
    # applies to source material, turned on its own cover: an all-black or blown-out still fails
    # it, which is exactly what a poster cut from frame zero of a clip that opens dark would be.
    assert frames._exposed(frames.sampled(poster_file)[0])


@pytest.mark.parametrize(
    "music",
    [
        "a track of the shelf, the way it was ordered before lists",
        "an uploaded track and a shelf track in one list",
    ],
)
def test_a_montage_reaches_the_length_that_was_ordered_by_holding_its_clips_longer(
    conn, s3, memory, tmp_path, music
):
    """The ordered length as a promise, on a real file: two twelve-second clips and two stills come
    to eleven and a half seconds at the lengths the отбор cuts them at, and twenty seconds were
    ordered. The clips are held longer — equally, and well inside their own files — until the video
    is as long as it was asked to be. Before this, the same order came back at those eleven and a
    half seconds without a word (TASK-193).

    Under a list of the organizer's own track and one of the shelf's, with a note to the director
    that no director is there to read, the file is the same: picture, and music that plays to the
    very end. The uploaded track is three seconds of silence, so a shelf track that was not joined
    on behind it would leave the rest of the video quiet."""
    if music.startswith("a track"):
        params = {"target_seconds": 20, "music_track_id": "warm"}
    else:
        own = f"music/{uuid.uuid4()}.wav"
        s3.upload_file(str(hush(tmp_path / "own.wav")), storage.bucket(), own)
        params = {
            "target_seconds": 20,
            "music_tracks": [own, "warm"],
            "music_note": "первый трек — на вступление",
        }
    montage_id = order(conn, memory, params)
    for index in range(2):
        _, key = add_item(conn, memory, "video", status="ready")
        s3.upload_file(str(long_clip(tmp_path / f"clip-{index}.mp4")), storage.bucket(), key)
    for index in range(2):
        _, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(jpeg(tmp_path / f"photo-{index}.jpg")), storage.bucket(), key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    # 4 + 4 + 2.5 + 2.5 less three joins is 11.5, so 8.5 seconds are missing and the two clips take
    # 4.25 each — 8.25 seconds of a twelve-second file apiece.
    assert about(duration, 20.0), "ролик вышел не той длины, что заказали"
    assembled_file = fetch(s3, video_key, tmp_path / "m.mp4")
    assembled = probe(assembled_file)
    assert about(float(assembled["format"]["duration"]), 20.0)
    tracks = {stream["codec_type"]: stream for stream in assembled["streams"]}
    assert (tracks["video"]["codec_name"], tracks["audio"]["codec_name"]) == ("h264", "aac")
    assert float(tracks["audio"]["duration"]) == pytest.approx(
        float(tracks["video"]["duration"]), abs=0.3
    ), "дорожка кончилась раньше картинки или пережила её"
    # The clips are on the screen at the start and the stills at the end, and the clips' own sound
    # is left out of a montage with music: a loud end can only be the music.
    assert loudness(assembled_file, at=duration - 3.2, seconds=1.0) > -45, (
        "в конце ролика нет музыки"
    )


@pytest.mark.parametrize("music", [["night"], []], ids=["under «Ночь»", "with no music"])
def test_a_montage_under_music_comes_out_the_length_it_was_ordered(
    conn, s3, memory, tmp_path, music
):
    """The ordered length under music, on a real file at a length a test can afford: a long clip
    and twelve stills ordered for thirty seconds. Under «Ночь», a hundred beats a minute, joins each
    pulled from the one before would all fall the same way and leave the video three seconds short;
    with no music there is nothing to pull. Either way the file ffprobe reads runs the thirty
    seconds to within one."""
    montage_id = order(conn, memory, {"target_seconds": 30, "music_tracks": music})
    _, key = add_item(conn, memory, "video", status="ready")
    s3.upload_file(str(long_clip(tmp_path / "clip.mp4")), storage.bucket(), key)
    for index in range(12):
        _, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(jpeg(tmp_path / f"photo-{index}.jpg")), storage.bucket(), key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, _, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    seconds = float(probe(fetch(s3, video_key, tmp_path / "m.mp4"))["format"]["duration"])
    assert abs(seconds - 30.0) <= 1.0, f"тридцать секунд заказали, а ролик — {seconds:.2f} с"


def test_the_finished_montage_says_where_in_it_each_frame_it_took_stands(
    conn, s3, memory, tmp_path
):
    """Every piece of the assembled video has its second and its length written down, and the one
    the organizer struck out has nothing written down at all — «Сведения о кадре» shows a frame the
    videos it is in and where, and a frame that is in none of them shows none (doc-02).

    Checked against the video that was actually made rather than against numbers named here: the
    pieces are pulled to the beat of the track, so what any of them is held for is the rhythm's to
    decide — but they must add up, join by join, to the file that came out of it.
    """
    ids = ready_material(conn, s3, memory, tmp_path)
    montage_id = order(conn, memory, {"target_seconds": 30, "exclude_media_ids": [ids[0]]})
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    where = layout_of(conn, montage_id)
    assert [media_id for media_id, _, _ in where] == ids[1:], (
        "не те кадры или не в том порядке: у вошедших запись есть, у исключённого — нет"
    )
    assert where[0][1] == 0.0, "the video opens on its first piece"
    # Each piece begins where the one before it ends, less the half-second the two of them share.
    for (_, start, seconds), (_, next_start, _) in zip(where, where[1:]):
        assert next_start == pytest.approx(start + seconds - assembly.CROSSFADE_SECONDS)
    # And the last of them ends with the video itself: the layout is of this file, not of a plan
    # that was made before it was encoded.
    assert where[-1][1] + where[-1][2] == pytest.approx(duration, abs=0.2)


def test_a_montage_assembled_again_replaces_its_layout_instead_of_doubling_it(conn, memory):
    """Re-ordering the same video writes its layout a second time, and the first one has to go with
    it: a frame that was dropped from the remake would otherwise still say it stands in it, and one
    that stayed would be listed twice.

    Straight at the writing, with pieces made up here: what a remake is, as far as this table is
    concerned, is one more run of it under the same montage id — and assembling two whole videos to
    see that costs minutes of ffmpeg for one `delete`.
    """
    montage_id = order(conn, memory, {"target_seconds": 30})
    dropped, _ = add_item(conn, memory, "photo", status="ready")
    kept, _ = add_item(conn, memory, "photo", status="ready")

    layout.write(conn, montage_id, [selection.Piece(media_id=dropped, seconds=3.0)])
    layout.write(conn, montage_id, [selection.Piece(media_id=kept, seconds=4.0)])

    assert layout_of(conn, montage_id) == [(kept, 0.0, 4.0)]


def test_a_layout_that_cannot_be_written_does_not_cost_the_montage(conn, memory):
    """The video is the thing; its layout is a line in a sheet about one frame. Minutes of encoding
    are not thrown away because an insert did not land, so a write that fails is a warning in the
    log — and the connection is still good afterwards, or the montage would die on the next
    statement instead of on this one."""
    gone = str(uuid.uuid4())  # no `montages` row, so both foreign keys refuse the rows

    layout.write(conn, gone, [selection.Piece(media_id=str(uuid.uuid4()), seconds=3.0)])

    assert layout_of(conn, gone) == []


def test_a_montage_ordered_portrait_is_assembled_the_other_way_up(conn, s3, memory, tmp_path):
    """The organizer asked for a video to watch on a phone. The same material, the same length, the
    same everything else — cut into 1080×1920 instead of 1920×1080. The numbers are written out
    here rather than read off `assembly.CANVASES`, which is the thing under test."""
    montage_id = order(conn, memory, {"target_seconds": 30, "orientation": "portrait"})
    # Stills 1200×800 and a clip 1280×1440 — material shot both ways round, and the portrait canvas
    # crops every piece of it to itself rather than padding any into it.
    ready_material(conn, s3, memory, tmp_path)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assert about(duration, 9.0), "turning the canvas does not change what goes in it"

    assembled_file = fetch(s3, video_key, tmp_path / "portrait.mp4")
    tracks = {stream["codec_type"]: stream for stream in probe(assembled_file)["streams"]}
    assert (tracks["video"]["width"], tracks["video"]["height"]) == (1080, 1920)
    # The real size is only half of it: a vertical file made of horizontal stills is exactly where
    # black bars are burnt in, and 1080×1920 is 1080×1920 with or without them.
    assert dimmest_edge(assembled_file) > BLACK, "чёрные поля по краям вертикального ролика"

    # The cover too: it is cut from the assembled file at the thumbnail width and keeps that file's
    # own shape, so a vertical montage gets a vertical cover with nobody deciding it should. The one
    # output whose orientation is a consequence rather than a choice, so it is the one that could
    # quietly go back to 16:9 while everything above stayed green.
    poster = probe(fetch(s3, poster_of(conn, montage_id), tmp_path / "poster.jpg"))["streams"][0]
    assert poster["width"] == preview.THUMBNAIL_WIDTH
    assert poster["height"] > poster["width"], "the cover stands the same way up as the video"


def test_an_orientation_nobody_has_fails_the_montage_instead_of_retrying(conn, memory):
    """The api writes one of two words here and refuses the rest, so a third one is the contract
    having moved on one side only — and it would be the same word on all three attempts."""
    montage_id = order(conn, memory, {"orientation": "diagonal"})
    job_id = worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "done"
    status, _, _, error_text = read_montage(conn, montage_id)
    assert status == "failed"
    assert "ориентация" in error_text


def test_a_montage_that_opens_on_a_video_with_a_dark_start_gets_a_bright_cover(
    conn, s3, memory, tmp_path
):
    """The regression the poster fix (ревью m-16) exists for: a phone clip that opens black. The
    piece cut from it starts inside that black stretch (centred on the best moment, then pulled back
    to the file), so a poster taken off frame zero of the assembled file — the pre-fix behaviour —
    would be black. Made to open the montage (oldest item, sorted first) so this is actually the
    first piece the cover is cut from, unlike the happy-path fixture where a photo always leads."""
    montage_id = order(conn, memory, {"target_seconds": 30})
    _, key = add_item(
        conn, memory, "video", status="ready", age_hours=5
    )  # oldest → opens the montage
    s3.upload_file(str(dark_opening_clip(tmp_path / "dark.mp4")), storage.bucket(), key)
    for index in range(3):
        _, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(jpeg(tmp_path / f"photo-{index}.jpg")), storage.bucket(), key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, _, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)

    poster_key = poster_of(conn, montage_id)
    poster_file = fetch(s3, poster_key, tmp_path / "poster.jpg")
    assert frames._exposed(frames.sampled(poster_file)[0]), (
        "the cover must be the opening clip's best moment, not the dark frame it was cut from"
    )


def test_a_montage_ordered_with_a_story_is_cut_to_the_directors_plan(
    conn, s3, memory, tmp_path, monkeypatch
):
    """The director end to end: a real order with a story in it, real material out of the bucket,
    and a video that is the plan's rather than the отбор's — a different length, out of a different
    three files. What the plan does with pieces on its own is `test_director.py`; this is here for
    what only a real video can answer, that the order actually reaches it and comes back assembled.

    Every other montage in this file is cut with no cloud model at all, which is the other half of
    the same proof: none of them changed.
    """
    ids = ready_material(conn, s3, memory, tmp_path)
    for media_id, description in zip(ids, ("накрывают на стол", "вносят торт", "все за столом")):
        conn.execute(
            """INSERT INTO media_understandings (media_id, memory_id, description, cast_labels)
               VALUES (%s, %s, %s, %s)""",
            (media_id, str(memory[0]), description, ["человек-1"]),
        )
    net = planning_over(monkeypatch, ids[:3])
    own = f"music/{uuid.uuid4()}.wav"
    s3.upload_file(str(tone(tmp_path / "own.wav", seconds=5.0)), storage.bucket(), own)
    montage_id = order(
        conn,
        memory,
        {
            "target_seconds": 30,
            "story_text": "день рождения, всё вокруг торта",
            "music_tracks": [own, "warm", own],
            "music_note": "  торт — под припев второго трека  ",
        },
    )
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    # The plan's own three, four and three seconds, less half a second for each of two joins, and
    # room for rhythm's pull towards the beat (`about`, `RHYTHM_SLACK`) — not the отбор's ten to
    # eleven, which is what these same four files come to on their own. The clip, which the plan
    # leaves out and the отбор takes, is the visible half of the difference.
    assert about(duration, 9.0), "the video is as long as the plan made it"
    assert about(float(probe(fetch(s3, video_key, tmp_path / "m.mp4"))["format"]["duration"]), 9.0)

    question = json.loads(net.sent.data)["input"][0]["text"]
    assert "всё вокруг торта" in question, "the story the organizer typed reached the director"
    assert "вносят торт" in question, "and so did what the pipeline already knew about the material"
    ((request,),) = conn.execute(
        "SELECT request FROM ai_requests WHERE operation = 'plan_montage' AND ref = %s",
        (montage_id,),
    ).fetchall()
    said = json.loads(request["prompt"])
    warm = round(ffmpeg.duration(library.track_file("warm")), 1)
    assert said["music"] == [
        {"track": "свой трек", "seconds": 5.0},
        {"track": "warm", "seconds": warm},
        {"track": "свой трек", "seconds": 5.0},
    ], "the soundtrack goes to the director in the order it plays, an own track by no bucket key"
    assert said["music_note"] == "торт — под припев второго трека", (
        "the wish is the organizer's words"
    )


def test_the_directors_film_is_held_to_the_ordered_length_the_same_way(
    conn, s3, memory, tmp_path, monkeypatch
):
    """The other half of the promise: the length is filled after the film is planned, not inside
    the planning, so a plan that comes to nine seconds against a sixteen-second order is held to it
    exactly as the отбор's own cut is. The plan opens on the long clip, and that is what the
    missing seconds go into — the model is not asked again and nothing is added to the film."""
    clip_id, key = add_item(conn, memory, "video", status="ready")
    s3.upload_file(str(long_clip(tmp_path / "clip.mp4")), storage.bucket(), key)
    ids = [clip_id]
    for index in range(2):
        photo_id, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(jpeg(tmp_path / f"photo-{index}.jpg")), storage.bucket(), key)
        ids.append(photo_id)
    for media_id, description in zip(ids, ("вносят торт", "все за столом", "задувают свечи")):
        conn.execute(
            """INSERT INTO media_understandings (media_id, memory_id, description, cast_labels)
               VALUES (%s, %s, %s, %s)""",
            (media_id, str(memory[0]), description, ["человек-1"]),
        )
    planning_over(monkeypatch, ids)
    montage_id = order(conn, memory, {"target_seconds": 16, "story_text": "день рождения"})
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    # The plan's own three, four and three seconds come to nine; the seven that are missing all go
    # into the clip, which the plan put first — ten seconds of its twelve.
    assert about(duration, 16.0), "план не дотянули до заказанной длины"


def test_a_montage_of_hundreds_of_pieces_is_assembled_without_ffmpeg_running_out_of_threads(
    tmp_path, monkeypatch
):
    """Ten minutes of montage is about three hundred pieces, and all of them are open in one ffmpeg
    at once: a decoder with a thread pool of its own on every input took the process past its thread
    limit at about two hundred (ревью m-26). The pieces here are short and on a canvas the size of a
    stamp — what is being asked is only whether that many inputs can be open at the same time, and
    the picture is not the point. Five hundred, not three: the limit is threads, cores times inputs,
    and on this ten-core machine three hundred inputs of the same clips still fit under it with the
    fix taken out."""
    monkeypatch.setitem(assembly.CANVASES, "landscape", (128, 72))
    still = jpeg(tmp_path / "still.jpg")
    pieces = [selection.Piece(kind="photo", seconds=0.6, path=still) for _ in range(500)]
    soundtrack = tone(tmp_path / "track.wav", seconds=40)
    out = tmp_path / "long.mp4"

    clips = assembly.assemble(pieces, "landscape", soundtrack, tmp_path, out)

    assert len(clips) == len(pieces)
    assembled = probe(out)
    assert float(assembled["format"]["duration"]) == pytest.approx(
        assembly.length_of(clips), abs=0.5
    )
    assert {stream["codec_type"] for stream in assembled["streams"]} == {"video", "audio"}


def test_too_little_material_fails_the_montage_instead_of_retrying(conn, memory):
    """Not a transient failure: the library would be just as small on the third attempt, and the
    organizer needs the reason under the video rather than three useless retries."""
    montage_id = order(conn, memory, {})
    for _ in range(montage.MIN_ITEMS - 1):
        add_item(conn, memory, "photo", status="ready")
    job_id = worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "done"
    status, _, _, error_text = read_montage(conn, montage_id)
    assert status == "failed"
    assert "материал" in error_text
    # No video was assembled, so there is no frame to stand for one: the api serves `poster_url`
    # null and the organizer sees «повторить» rather than a cover for a video that does not exist.
    assert poster_of(conn, montage_id) is None


def test_a_montage_that_keeps_failing_fails_the_row(conn, memory):
    # Rows whose originals never made it into the bucket: the download fails every time.
    montage_id = order(conn, memory, {})
    for _ in range(montage.MIN_ITEMS):
        add_item(conn, memory, "photo", status="ready")
    job_id = worker.enqueue(conn, "montage", montage_id)

    for _ in range(worker.MAX_ATTEMPTS):
        assert worker.run_once(conn) is True
        conn.execute(
            "UPDATE jobs SET run_after = now() WHERE id = %s", (job_id,)
        )  # skip the backoff

    assert (
        conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "failed"
    )
    # And the montage with it, so the organizer gets «повторить» and the reason it fell over.
    status, _, _, error_text = read_montage(conn, montage_id)
    assert status == "failed"
    # A fixed, human sentence — never the exception's own text. `error_text` goes out to every
    # member of the memory, and the exception here is a botocore message naming the bucket host.
    assert error_text == "не удалось собрать ролик"


def test_an_api_that_does_not_answer_does_not_fail_a_finished_montage(monkeypatch):
    """The video exists and the row already says so — a failure here would rebuild the whole thing
    on the next attempt to deliver one push."""
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # discard: nothing is listening
    monkeypatch.setenv("INTERNAL_WORKER_TOKEN", "secret")

    montage._tell_the_api("a3b1f2e4-0000-4000-8000-000000000000")  # no exception is the assertion
