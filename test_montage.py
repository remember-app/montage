"""Whole montages, ordered and assembled end to end — see README.md.

Real everything, like `test_media.py`: a real Postgres, a real bucket, real ffmpeg, and fixtures
ffmpeg draws on the spot. What the choosing does on its own, with no file behind it, is
`test_selection.py`; these are here for what only a real video can answer — that the material
actually comes out of the bucket, through both modules and into one playable file.

Warning: these wipe the `jobs` table of whatever DATABASE_URL points at, and write into the bucket
R2_ENDPOINT points at.
"""

import json
import uuid

import pytest

import ai
import assembly
import media
import montage
import rhythm
import selection
import worker

# The recorded answer and the transport that serves it are `test_ai.py`'s, like everywhere else the
# cloud model is involved: there is one way to answer without a network in this suite.
from test_ai import Recorded

# The fixtures and the ffmpeg-drawn material are `test_media.py`'s — a second copy of a Postgres
# fixture is a second thing to keep true. pytest resolves fixtures by name in this module, so they
# have to be imported by name and not through the module.
from test_media import (  # noqa: F401
    add_item, conn, ffmpeg, fetch, memory, photo, probe, s3, video,
)


# --- the whole thing --------------------------------------------------------


def order(conn, memory, params):
    """A `montages` row as the api leaves it when a video is ordered."""
    if conn.execute("SELECT to_regclass('montages')").fetchone()[0] is None:
        pytest.skip("montages is missing — run the api's migrations first (see README.md)")
    memory_id, user_id = memory
    return str(conn.execute(
        """INSERT INTO montages (memory_id, ordered_by, params)
           VALUES (%s, %s, %s::jsonb) RETURNING id""",
        (memory_id, user_id, json.dumps(params)),
    ).fetchone()[0])


def read_montage(conn, montage_id):
    return conn.execute(
        "SELECT status, video_key, duration_seconds, error_text FROM montages WHERE id = %s",
        (montage_id,),
    ).fetchone()


def poster_of(conn, montage_id):
    """The montage's cover key, `None` until a video has actually been assembled. Read on its own
    rather than added to `read_montage`: only the two tests below have anything to say about it."""
    return conn.execute(
        "SELECT poster_key FROM montages WHERE id = %s", (montage_id,)
    ).fetchone()[0]


# A montage is cut to the beat of its track: every piece is pulled to the nearest one, so the
# finished video comes out around the sum of its pieces rather than exactly on it — by up to half a
# beat either way, and the beats of real music are not evenly spaced (`rhythm.py`). A second of
# slack covers that and still tells a montage of four pieces from one of three or five: there are
# three seconds between those. What the windows below are for is how many pieces went in, not the
# music — the rhythm itself is `test_rhythm.py`'s, and a cut to an uploaded track is below.
RHYTHM_SLACK = 1.0


def about(duration, expected):
    """The montage came out about as long as it should, with room for the pull towards the beat."""
    return abs(duration - expected) <= RHYTHM_SLACK


BLACK = 16  # the value `selection._exposed` already counts as black


def dimmest_edge(path):
    """The dimmest edge in the whole file: the brightest pixel along each of a frame's four edges,
    and the smallest of those numbers over every second of the video.

    This is how a black bar is caught on the finished file instead of in the filter string that was
    supposed to prevent it. A letterboxed montage has a whole edge row of nothing but black, so the
    brightest pixel along it is 0; a canvas the picture fills has something in every edge. The
    frames come from `selection`'s own 96×54 grey sampling — coarse, but a bar is a third of the
    canvas, and the material here is a test pattern with its corners lit.
    """
    width = selection.SAMPLE_WIDTH
    dimmest = 255
    for frame in selection._grey_frames(path, per_second=selection.SAMPLE_FPS):
        rows = [frame[at:at + width] for at in range(0, len(frame), width)]
        edges = (rows[0], rows[-1], [row[0] for row in rows], [row[-1] for row in rows])
        dimmest = min(dimmest, *(max(edge) for edge in edges))
    return dimmest


def material(conn, s3, memory, tmp_path):
    """Three photographs and a clip in the bucket, `ready`, as the previews leave them. Their ids
    come back in the order the montage reads them, for the tests that name one in the order."""
    ids = []
    for index in range(3):
        item_id, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(photo(tmp_path / f"photo-{index}.jpg")), media.bucket(), key)
        ids.append(item_id)
    item_id, key = add_item(conn, memory, "video", status="ready")
    s3.upload_file(str(video(tmp_path / "clip.mp4")), media.bucket(), key)
    return [*ids, item_id]


def test_a_memory_of_photos_and_a_clip_becomes_one_playable_video(conn, s3, memory, tmp_path):
    montage_id = order(conn, memory, {"target_seconds": 30, "music_track_id": "warm"})
    material(conn, s3, memory, tmp_path)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, duration, error_text = read_montage(conn, montage_id)
    assert (status, video_key, error_text) == ("ready", f"montages/{montage_id}.mp4", None)
    # Three stills of three seconds and a four-second clip cut to the three it has — twelve seconds
    # of material, less half a second for each of the three crossfades. Pieces butted together
    # instead of overlapped would come to twelve, which is outside the window even with the slack.
    assert about(duration, 10.5)

    assembled_file = fetch(s3, video_key, tmp_path / "montage.mp4")
    assembled = probe(assembled_file)
    assert about(float(assembled["format"]["duration"]), 10.5)
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
    assert (poster["codec_name"], poster["width"]) == ("mjpeg", media.THUMBNAIL_WIDTH)
    # Not just a jpeg of the right size — a frame worth showing. The same judgement the worker
    # applies to source material, turned on its own cover: an all-black or blown-out still fails
    # it, which is exactly what a poster cut from frame zero of a clip that opens dark would be.
    assert selection._exposed(selection._grey_frames(poster_file)[0])


def click_track(path, gap=1.6):
    """Thirty seconds of metronome: one click every 1.6 seconds. The tempo is known because it was
    put there, and it is deliberately slow — under a track like this the pieces stretch visibly, and
    the same material can no longer come to its even 10.5 seconds."""
    ffmpeg("-f", "lavfi",
           "-i", f"aevalsrc='0.8*sin(2*PI*1200*t)*exp(-40*mod(t,{gap}))':d=30", str(path))
    return path


def test_a_montage_is_cut_to_the_beat_of_the_track_the_organizer_uploaded(
    conn, s3, memory, tmp_path
):
    """The organizer's own track instead of the library's: the order names a key rather than an id
    (the api's `music_uploads`), and the joins fall into its rhythm just the same.

    The rhythm is checked at the last join and only there: behind it stands the last piece, which
    the fitting never touches, and its length is known — three seconds. So the last join sits at
    the end of the montage less those three seconds plus the half-crossfade that piece began with,
    and that moment has to be one of the track's beats. The montage itself, under a metronome this
    slow, has to come out well past the even 10.5 seconds — otherwise nothing was pulled anywhere.
    """
    track_key = f"music/{uuid.uuid4()}.wav"
    s3.upload_file(str(click_track(tmp_path / "click.wav")), media.bucket(), track_key)
    montage_id = order(conn, memory, {"target_seconds": 30, "music_track_key": track_key})
    material(conn, s3, memory, tmp_path)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assert duration > 11.5, "the pieces reached for the beats — even three-second cuts do not"

    # Three seconds: that is how long both a photograph and this material's clip are held (its file
    # is three seconds long, and «3–5 секунд вокруг лучшего момента» is cut from what there is), so
    # whichever of the four ends up last, the tail of the montage is the same.
    last_join = duration - 3.0 + assembly.CROSSFADE_SECONDS / 2
    beats = rhythm.beats(tmp_path / "click.wav")
    assert beats, "a metronome is a tempo, or there is nothing here to check"
    # A frame and a half: for the encoding, which rounds the montage's length to one, and for
    # nothing else — a join that missed its beat would miss it by hundreds of milliseconds.
    assert min(abs(last_join - beat) for beat in beats) < 0.1, (
        f"the last join, at {last_join:.2f}, is not on a beat"
    )


def test_a_music_file_that_will_not_open_fails_the_montage_instead_of_retrying(
    conn, s3, memory, tmp_path
):
    """An uploaded track is somebody else's bytes, and ffmpeg reads them. Not music at all (or a
    playlist, which would send ffmpeg walking this worker's disks) — refused on the spot: the second
    attempt would meet the same file."""
    track_key = f"music/{uuid.uuid4()}.wav"
    s3.upload_file(str(unopenable(tmp_path / "notmusic")), media.bucket(), track_key)
    montage_id = order(conn, memory, {"music_track_key": track_key})
    material(conn, s3, memory, tmp_path)
    job_id = worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "done"
    status, _, _, error_text = read_montage(conn, montage_id)
    assert status == "failed"
    assert "трек" in error_text


def test_a_montage_ordered_portrait_is_assembled_the_other_way_up(conn, s3, memory, tmp_path):
    """The organizer asked for a video to watch on a phone. The same material, the same length, the
    same everything else — cut into 1080×1920 instead of 1920×1080. The numbers are written out
    here rather than read off `assembly.CANVASES`, which is the thing under test."""
    montage_id = order(conn, memory, {"target_seconds": 30, "orientation": "portrait"})
    # Stills 1200×800 and a clip 1280×1440 — material shot both ways round, and the portrait canvas
    # crops every piece of it to itself rather than padding any into it.
    material(conn, s3, memory, tmp_path)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assert about(duration, 10.5), "turning the canvas does not change what goes in it"

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
    assert poster["width"] == media.THUMBNAIL_WIDTH
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


def dark_opening_clip(path):
    """A clip that opens black for its first 1.5s, then three and a half seconds of a sharp,
    properly exposed pattern — the shape of a real phone clip that takes a moment to find its
    exposure. `_best_moment` finds its best frame only after the black stretch, but the piece is cut
    *centred* on that moment and pulled back to fit the file, so the cut can still start inside the
    black part. The happy-path fixture (`material`) always puts a photo first, so its montage never
    opens on a video and this shape of bug never gets a chance to show up."""
    ffmpeg(
        "-f", "lavfi", "-i", "color=c=black:size=1280x720:rate=24:duration=1.5",
        "-f", "lavfi", "-i", "testsrc=size=1280x720:rate=24:duration=3.5",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map", "[v]", "-map", "2:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    )
    return path


def test_a_montage_that_opens_on_a_video_with_a_dark_start_gets_a_bright_cover(
    conn, s3, memory, tmp_path
):
    """The regression the poster fix (ревью m-16) exists for: a phone clip that opens black. The
    piece cut from it starts inside that black stretch (centred on the best moment, then pulled back
    to the file), so a poster taken off frame zero of the assembled file — the pre-fix behaviour —
    would be black. Made to open the montage (oldest item, sorted first) so this is actually the
    first piece the cover is cut from, unlike the happy-path fixture where a photo always leads."""
    montage_id = order(conn, memory, {"target_seconds": 30})
    _, key = add_item(conn, memory, "video", status="ready", age_hours=5)  # oldest → opens the montage
    s3.upload_file(str(dark_opening_clip(tmp_path / "dark.mp4")), media.bucket(), key)
    for index in range(3):
        _, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(photo(tmp_path / f"photo-{index}.jpg")), media.bucket(), key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, _, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)

    poster_key = poster_of(conn, montage_id)
    poster_file = fetch(s3, poster_key, tmp_path / "poster.jpg")
    assert selection._exposed(selection._grey_frames(poster_file)[0]), (
        "the cover must be the opening clip's best moment, not the dark frame it was cut from"
    )


def planning_over(monkeypatch, media_ids):
    """The recorded plan (`fixtures/gemini_plan.json`) as the only cloud model this worker can find.

    The ids in it are this run's own: a montage is planned over material whose ids Postgres makes up
    at insert time, so the three the answer was written about are rewritten into the three actually
    ordered. Everything else — what a plan is, what a step of it carries — is the recorded file's,
    and nothing here goes anywhere near the network.
    """
    recorded = Recorded("gemini_plan.json")
    for index, media_id in enumerate(media_ids, start=1):
        recorded.answer = recorded.answer.replace(f"m-{index}".encode(), media_id.encode())
    monkeypatch.setattr(ai, "from_env", lambda *_, **__: ai.Gemini("test-key", transport=recorded))
    return recorded


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
    ids = material(conn, s3, memory, tmp_path)
    for media_id, description in zip(ids, ("накрывают на стол", "вносят торт", "все за столом")):
        conn.execute(
            """INSERT INTO media_understandings (media_id, memory_id, description, cast_labels)
               VALUES (%s, %s, %s, %s)""",
            (media_id, str(memory[0]), description, ["человек-1"]),
        )
    net = planning_over(monkeypatch, ids[:3])
    montage_id = order(conn, memory, {
        "target_seconds": 30, "story_text": "день рождения, всё вокруг торта",
    })
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


def unopenable(path):
    """Bytes in the bucket that ffmpeg will not open — a truncated upload, a file the phone
    mislabelled. Not a playlist: that is the security case and `test_media.py` has it."""
    path.write_bytes(b"this is not a photo and never was")
    return path


def crushed(path):
    """A photo with the same edges `photo()` has — sharp, easily past `MIN_SHARPNESS` — pushed
    almost to black. ffmpeg opens it without complaint, unlike `unopenable`; only the exposure
    check has anything to say about it, which is the point: a flat black frame would be rejected on
    blur alone and prove nothing about the histogram half of `_best_moment`."""
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=640x480:duration=1", "-vf", "eq=brightness=-0.85",
        "-frames:v", "1", str(path),
    )
    return path


def lopsided(path):
    """Two seconds of picture under three and a half of sound, so the container calls it 3.5 —
    `format=duration` is the longest stream, and `-t` cuts against the picture."""
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=640x480:rate=30:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3.5",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path),
    )
    return path


def test_a_file_that_will_not_open_is_left_out_rather_than_failing_the_video(
    conn, s3, memory, tmp_path
):
    """One corrupt upload is that file's problem, not the memory's. It used to raise straight out
    of the handler: three retries, then a montage failed for good — and «повторить» met the same
    file, so any one member could deny the organizer a video permanently."""
    montage_id = order(conn, memory, {"target_seconds": 30})
    material(conn, s3, memory, tmp_path)
    _, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(unopenable(tmp_path / "broken")), media.bucket(), key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assert about(duration, 10.5), "the four that opened went in, the broken one was left out"


def test_a_badly_exposed_photo_is_left_out_rather_than_failing_the_video(
    conn, s3, memory, tmp_path
):
    """A different code path from a file ffmpeg refuses to open: this one decodes fine and the
    отсев turns it down on exposure, the one frame in it being crushed almost to black. The unit
    tests on `choose` never exercise this — they score every piece by hand."""
    montage_id = order(conn, memory, {"target_seconds": 30})
    material(conn, s3, memory, tmp_path)
    _, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(crushed(tmp_path / "crushed.jpg")), media.bucket(), key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assert about(duration, 10.5), "the crushed photo had no usable frame and was left out uncounted"


def test_a_photo_the_organizer_marked_include_is_in_the_video_the_algorithm_notwithstanding(
    conn, s3, memory, tmp_path
):
    """The same crushed photo as above, and the same verdict from the algorithm — except this time
    the order names it. «Включить» is the organizer overruling the automatic отсев, so the piece
    the test above proves is dropped is here proved to be in the video."""
    material(conn, s3, memory, tmp_path)
    crushed_id, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(crushed(tmp_path / "crushed.jpg")), media.bucket(), key)
    montage_id = order(
        conn, memory, {"target_seconds": 30, "include_media_ids": [crushed_id]}
    )
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    # Five pieces of three seconds instead of four, less half a second for each of four joins.
    assert about(duration, 13.0), "the crushed photo went in because the order asked for it"


def test_the_floor_counts_the_pieces_the_organizer_asked_for_too(conn, s3, memory, tmp_path):
    """A memory the отсев turns down almost entirely, and an organizer who overrules it. The floor
    is «сколько кусков реально войдёт», so the two crushed photos named in «включить» count towards
    it — refusing the order here would be the automatic отсев beating the person, which is the one
    thing «включить» exists to prevent (doc-03, «Кадры, отмеченные включить, есть в ролике»)."""
    _, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(photo(tmp_path / "good.jpg")), media.bucket(), key)
    wanted = []
    for index in range(2):
        crushed_id, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(crushed(tmp_path / f"crushed-{index}.jpg")), media.bucket(), key)
        wanted.append(crushed_id)
    montage_id = order(conn, memory, {"target_seconds": 30, "include_media_ids": wanted})
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assert about(duration, 8.0), "all three went in: three stills, less half a second for each join"


def test_a_photo_the_organizer_marked_exclude_stays_out_of_the_video(conn, s3, memory, tmp_path):
    """The other half of the same word from the organizer, on material the algorithm is perfectly
    happy with: a photo named in «исключить» is not in the video, and the rest of the material
    fills the time it would have taken."""
    photo_id = material(conn, s3, memory, tmp_path)[0]
    montage_id = order(conn, memory, {"target_seconds": 30, "exclude_media_ids": [photo_id]})
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    # Two stills and the clip — nine seconds of material, less half a second for each of two joins.
    assert about(duration, 8.0), "one of the four pieces was left out on the order's own word"


def test_the_floor_does_not_count_the_pieces_the_organizer_struck_out(conn, s3, memory, tmp_path):
    """Four files the отсев is happy with, two of them named in «исключить». The floor is what can
    actually end up on screen, and `selection.choose` throws the struck-out two away whatever the
    floor thought — so a minute-long order backed by two pieces is refused here rather than
    assembled into a seven-second «ролик»."""
    struck = material(conn, s3, memory, tmp_path)[:2]
    montage_id = order(conn, memory, {"target_seconds": 60, "exclude_media_ids": struck})
    job_id = worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "done"
    status, video_key, _, error_text = read_montage(conn, montage_id)
    assert (status, video_key) == ("failed", None), "nothing was assembled out of the two left"
    assert "не годится" in error_text


def test_the_floor_treats_a_piece_named_on_both_lists_as_struck_out(conn, s3, memory, tmp_path):
    """The order contradicting itself is settled the same way here as in `selection.choose`:
    «исключить» is the one that cannot be undone once a video is made, so it is the one the floor
    counts too. Three good photos, one of them on both lists, are really only two — and a floor that
    let «включить» save it here would pass an order `selection.choose` then cuts down to two anyway."""
    ids = []
    for index in range(3):
        item_id, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(photo(tmp_path / f"photo-{index}.jpg")), media.bucket(), key)
        ids.append(item_id)
    montage_id = order(conn, memory, {
        "target_seconds": 60, "include_media_ids": [ids[0]], "exclude_media_ids": [ids[0]],
    })
    job_id = worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "done"
    status, video_key, _, error_text = read_montage(conn, montage_id)
    assert (status, video_key) == ("failed", None), "the contradiction did not save it from the floor"
    assert "не годится" in error_text


def test_too_little_material_that_actually_opens_fails_without_retrying(conn, s3, memory, tmp_path):
    """The floor is about material that can go in the video, not rows in the table: three rows of
    which two are unreadable would otherwise have been assembled into one three-second still."""
    montage_id = order(conn, memory, {})
    _, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(photo(tmp_path / "good.jpg")), media.bucket(), key)
    for index in range(montage.MIN_ITEMS - 1):
        _, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(unopenable(tmp_path / f"broken-{index}")), media.bucket(), key)
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
    s3.upload_file(str(lopsided(tmp_path / "lopsided.mp4")), media.bucket(), key)
    for index in range(3):
        _, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(photo(tmp_path / f"photo-{index}.jpg")), media.bucket(), key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, duration, _ = read_montage(conn, montage_id)
    assert status == "ready"
    # Two seconds of picture and three stills of three, less half a second for each of three
    # joins: 9.5. Booked at the container's 3.5 instead, ffmpeg truncates the chain and the whole
    # montage comes out at three seconds — nowhere near this window, slack for the rhythm or not.
    assert about(duration, 9.5), "the clip went in at its real two seconds"
    assert about(float(probe(fetch(s3, video_key, tmp_path / "m.mp4"))["format"]["duration"]), 9.5)


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


def test_a_track_nobody_has_fails_the_montage_instead_of_retrying(conn, memory):
    montage_id = order(conn, memory, {"music_track_id": "polka"})
    job_id = worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "done"
    status, _, _, error_text = read_montage(conn, montage_id)
    assert status == "failed"
    assert "трек" in error_text


def test_a_montage_that_keeps_failing_fails_the_row(conn, memory):
    # Rows whose originals never made it into the bucket: the download fails every time.
    montage_id = order(conn, memory, {})
    for _ in range(montage.MIN_ITEMS):
        add_item(conn, memory, "photo", status="ready")
    job_id = worker.enqueue(conn, "montage", montage_id)

    for _ in range(worker.MAX_ATTEMPTS):
        assert worker.run_once(conn) is True
        conn.execute("UPDATE jobs SET run_after = now() WHERE id = %s", (job_id,))  # skip the backoff

    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "failed"
    # And the montage with it, so the organizer gets «повторить» and the reason it fell over.
    status, _, _, error_text = read_montage(conn, montage_id)
    assert status == "failed"
    # A fixed, human sentence — never the exception's own text. `error_text` goes out to every
    # member of the memory, and the exception here is a botocore message naming the bucket host.
    assert error_text == "не удалось собрать ролик"


@pytest.mark.parametrize("track_id", assembly.TRACKS)
def test_every_track_of_the_library_is_music_that_plays(track_id):
    """Five files on disk, and a montage ordered with one of them falls over only when somebody
    happens to order *that* one — so all five are opened here, not just the default. A track
    renamed, moved or never committed fails this test instead of somebody's video."""
    played = probe(assembly.track_file(track_id))

    assert "audio" in [stream["codec_type"] for stream in played["streams"]]
    assert float(played["format"]["duration"]) > 30, "music, not a second and a half of signal"


def library_in_the_bucket(s3):
    """What the bucket holds under each track's key: its size and type, or None for one that is not
    there at all."""
    heads = {}
    for track_id in assembly.TRACKS:
        try:
            head = s3.head_object(Bucket=media.bucket(), Key=assembly.track_key(track_id))
        except s3.exceptions.ClientError:
            heads[track_id] = None
        else:
            heads[track_id] = (head["ContentLength"], head["ContentType"], head["LastModified"])
    return heads


def test_the_library_is_published_to_the_bucket_for_the_app_to_play(s3):
    """The five files are this worker's own disk, and «Послушать» is the api's and the phone's — so
    the library has to be in the bucket too, under the keys the api builds from the same ids."""
    for track_id in assembly.TRACKS:
        s3.delete_object(Bucket=media.bucket(), Key=assembly.track_key(track_id))

    assert sorted(assembly.publish_tracks(s3)) == sorted(assembly.TRACKS)

    landed = library_in_the_bucket(s3)
    assert [track_id for track_id, head in landed.items() if head is None] == []
    assert {track_id: head[:2] for track_id, head in landed.items()} == {
        track_id: (assembly.track_file(track_id).stat().st_size, "audio/mpeg")
        for track_id in assembly.TRACKS
    }, "all five, whole, and served as music rather than as bytes of nothing in particular"

    # Run on every deploy: five mp3s must not be written over every time.
    assert assembly.publish_tracks(s3) == [], "what is already there is not uploaded again"
    assert library_in_the_bucket(s3) == landed, "and the objects themselves were not touched"


def test_an_api_that_does_not_answer_does_not_fail_a_finished_montage(monkeypatch):
    """The video exists and the row already says so — a failure here would rebuild the whole thing
    on the next attempt to deliver one push."""
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # discard: nothing is listening
    monkeypatch.setenv("INTERNAL_WORKER_TOKEN", "secret")

    montage._tell_the_api("a3b1f2e4-0000-4000-8000-000000000000")  # no exception is the assertion
