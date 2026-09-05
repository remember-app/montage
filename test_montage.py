"""The choosing on its own, and one live montage end to end — see README.md.

The first half needs nothing at all: `choose` is a function over scored pieces, so what it does with
a burst of shots or a length it cannot fit is checked without a frame being decoded. The second half
is real everything, like `test_media.py`: a real Postgres, a real bucket, real ffmpeg, and fixtures
ffmpeg draws on the spot.

Warning: these wipe the `jobs` table of whatever DATABASE_URL points at, and write into the bucket
R2_ENDPOINT points at.
"""

import datetime
import json

import pytest

import media
import montage
import worker

# The fixtures and the ffmpeg-drawn material are `test_media.py`'s — a second copy of a Postgres
# fixture is a second thing to keep true. pytest resolves fixtures by name in this module, so they
# have to be imported by name and not through the module.
from test_media import (  # noqa: F401
    add_item, conn, ffmpeg, fetch, memory, photo, probe, s3, video,
)


# --- choosing ---------------------------------------------------------------
# No database, no bucket, no ffmpeg: scored pieces in, the chosen ones out.


def at(second, sharpness, *, dated=True):
    """A piece shot `second` seconds past midnight, as sharp as it is told to be."""
    when = datetime.datetime(2026, 9, 5) + datetime.timedelta(seconds=second)
    return montage.Piece(sharpness=sharpness, when=when, captured_at=when if dated else None)


def test_a_burst_of_shots_leaves_only_its_sharpest():
    burst = [at(0, 10), at(1, 90), at(2, 30)]  # three shots inside two seconds — one moment
    apart = at(30, 20)

    chosen = montage.choose([*burst, apart], target_seconds=60)

    assert chosen == [burst[1], apart], "the sharpest of the burst stands for all three"


def test_a_long_sequence_stays_several_moments():
    """Measured from the first of a burst and not from the shot before it: a whole evening of
    photographs a second apart is not one moment."""
    every_second = [at(second, 10 + second) for second in range(6)]

    chosen = montage.choose(every_second, target_seconds=60)

    # 0-1-2 is one moment and 3-4-5 the next, so the sharpest of each survives.
    assert chosen == [every_second[2], every_second[5]]


def test_a_file_with_no_capture_date_is_never_deduped():
    """Nothing says those two were taken together, and dropping one would drop material over a
    missing EXIF field."""
    undated = [at(0, 10, dated=False), at(1, 90, dated=False)]

    assert montage.choose(undated, target_seconds=60) == undated


def test_only_what_fits_the_asked_for_length_gets_in_and_it_gets_in_in_order():
    pieces = [at(second * 10, sharpness) for second, sharpness in enumerate([10, 90, 20, 80, 30])]

    chosen = montage.choose(pieces, target_seconds=8)

    # Three seconds each, half a second eaten by every crossfade: 3, 5.5, 8 — a fourth would be 10.5.
    assert montage.length_of(chosen) == 8
    assert chosen == [pieces[1], pieces[3], pieces[4]], "the sharpest three, back in date order"


def test_the_first_piece_is_taken_however_long_it_is():
    """A montage shorter than what was asked for is still a montage; an empty one is not a file."""
    assert montage.choose([at(0, 10, dated=False)], target_seconds=1) == [at(0, 10, dated=False)]


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


def material(conn, s3, memory, tmp_path):
    """Three photographs and a clip in the bucket, `ready`, as the previews leave them."""
    for index in range(3):
        _, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(photo(tmp_path / f"photo-{index}.jpg")), media.bucket(), key)
    _, key = add_item(conn, memory, "video", status="ready")
    s3.upload_file(str(video(tmp_path / "clip.mp4")), media.bucket(), key)


def test_a_memory_of_photos_and_a_clip_becomes_one_playable_video(conn, s3, memory, tmp_path):
    montage_id = order(conn, memory, {"target_seconds": 30, "music_track_id": "warm"})
    material(conn, s3, memory, tmp_path)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, duration, error_text = read_montage(conn, montage_id)
    assert (status, video_key, error_text) == ("ready", f"montages/{montage_id}.mp4", None)
    # Three stills of three seconds and a four-second clip cut to the three it has — twelve seconds
    # of material, less half a second for each of the three crossfades. The window is tight on
    # purpose: pieces butted together instead of overlapped would come to twelve.
    assert 10 <= duration <= 11

    assembled = probe(fetch(s3, video_key, tmp_path / "montage.mp4"))
    assert 10 <= float(assembled["format"]["duration"]) <= 11
    tracks = {stream["codec_type"]: stream for stream in assembled["streams"]}
    assert set(tracks) == {"video", "audio"}, "picture and music both landed"
    assert tracks["video"]["codec_name"] == "h264"
    assert tracks["audio"]["codec_name"] == "aac"
    assert (tracks["video"]["width"], tracks["video"]["height"]) == (
        montage.CANVAS_WIDTH, montage.CANVAS_HEIGHT
    ), "everything is padded onto the one canvas, whichever way it was shot"


def unopenable(path):
    """Bytes in the bucket that ffmpeg will not open — a truncated upload, a file the phone
    mislabelled. Not a playlist: that is the security case and `test_media.py` has it."""
    path.write_bytes(b"this is not a photo and never was")
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
    assert 10 <= duration <= 11, "the four that opened went in, the broken one was left out"


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
    # joins: 9.5. Booked at the container's 3.5 instead, the file comes out visibly shorter.
    assert 9.3 <= duration <= 9.7, "the clip went in at its real two seconds"
    assert 9.3 <= float(probe(fetch(s3, video_key, tmp_path / "m.mp4"))["format"]["duration"]) <= 9.7


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
    assert error_text, "the last error is what the organizer is shown"


@pytest.mark.parametrize("track_id", sorted(montage.TRACKS))
def test_every_track_is_something_ffmpeg_can_play(track_id, tmp_path):
    """The tracks are expressions rather than files, so a typo in one is a montage that fails only
    when somebody happens to order that track. Here all three are drawn, not just the default."""
    played = tmp_path / f"{track_id}.m4a"
    media._ffmpeg(
        "-f", "lavfi", "-i", f"aevalsrc={montage.TRACKS[track_id]}:d=1", "-c:a", "aac", str(played)
    )

    assert probe(played)["streams"][0]["codec_name"] == "aac"


def test_an_api_that_does_not_answer_does_not_fail_a_finished_montage(monkeypatch):
    """The video exists and the row already says so — a failure here would rebuild the whole thing
    on the next attempt to deliver one push."""
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")  # discard: nothing is listening
    monkeypatch.setenv("INTERNAL_WORKER_TOKEN", "secret")

    montage._tell_the_api("a3b1f2e4-0000-4000-8000-000000000000")  # no exception is the assertion
