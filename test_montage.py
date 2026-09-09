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
import random

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


# --- judging a frame ---------------------------------------------------------
# `choose`'s own tests hand it a sharpness directly and never touch the arithmetic that produces
# one. These do: tiny grey frames built by hand, the same size ffmpeg hands `_sharpness`/`_exposed`
# for real, no ffmpeg needed to check the maths itself is right.

FRAME_SIZE = montage.SAMPLE_WIDTH * montage.SAMPLE_HEIGHT


def grey(*values):
    """`FRAME_SIZE` grey bytes, `values` repeated to fill it — one value is a flat frame, two
    alternated is a checkerboard."""
    return bytes(values[i % len(values)] for i in range(FRAME_SIZE))


def test_a_flat_frame_has_no_sharpness():
    """No edges anywhere: the Laplacian is zero at every pixel, so its variance is too — this is
    what a smeared, out-of-focus frame looks like to `_sharpness`."""
    assert montage._sharpness(grey(128)) == 0.0


def test_a_checkerboard_is_sharp():
    """Every neighbouring pixel is as different as it gets — the opposite end of the scale from a
    flat frame, and well past `MIN_SHARPNESS`."""
    assert montage._sharpness(grey(0, 255)) > montage.MIN_SHARPNESS


def laplacian_variance(frame):
    """The same number `_sharpness` promises, written out the long obvious way and owing nothing to
    how `_sharpness` happens to compute it: the Laplacian at every pixel that has four neighbours,
    then the variance of those values over the whole population."""
    width, height = montage.SAMPLE_WIDTH, montage.SAMPLE_HEIGHT

    def at(x, y):
        return frame[y * width + x]

    values = [
        4 * at(x, y) - at(x - 1, y) - at(x + 1, y) - at(x, y - 1) - at(x, y + 1)
        for y in range(1, height - 1)
        for x in range(1, width - 1)
    ]
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


@pytest.mark.parametrize("frame", [
    grey(128),  # flat
    grey(0, 255),  # checkerboard
    grey(3, 200, 17, 250, 99),  # a period that does not divide the row: no two rows alike
    random.Random(59).randbytes(FRAME_SIZE),  # noise, the same noise every run
], ids=["flat", "checkerboard", "irregular", "noise"])
def test_sharpness_is_the_variance_of_the_laplacian(frame):
    """«A flat frame scores nothing, a checkerboard scores a lot» is true of half a dozen different
    formulas, so those two alone would let the number drift while the tests stayed green — and the
    number is what decides which frames end up in the video. This one pins it: the value itself,
    against the definition spelled out here rather than against `_sharpness`'s own arithmetic."""
    assert montage._sharpness(frame) == pytest.approx(laplacian_variance(frame), rel=1e-12)


def test_a_mid_grey_varied_frame_is_exposed():
    assert montage._exposed(grey(100, 150)) is True


def test_a_flat_dark_frame_is_not_exposed():
    assert montage._exposed(grey(5)) is False


def test_a_flat_bright_frame_is_not_exposed():
    assert montage._exposed(grey(250)) is False


def test_a_frame_split_between_black_and_blown_is_not_exposed():
    """Mean brightness alone would wave this one through — half black, half blown averages to a mid
    grey. The clipped-pixel count is what catches a frame that is really crushed shadow next to
    blown highlight and nothing usable in between."""
    assert montage._exposed(grey(0, 255)) is False


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
    assert (tracks["video"]["width"], tracks["video"]["height"]) == (1920, 1080), (
        "an order that says nothing about orientation is the horizontal canvas, as it always was"
    )

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
    assert montage._exposed(montage._grey_frames(poster_file)[0])


def test_a_montage_ordered_portrait_is_assembled_the_other_way_up(conn, s3, memory, tmp_path):
    """The organizer asked for a video to watch on a phone. The same material, the same length, the
    same everything else — padded into 1080×1920 instead of 1920×1080. The numbers are written out
    here rather than read off `montage.CANVASES`, which is the thing under test."""
    montage_id = order(conn, memory, {"target_seconds": 30, "orientation": "portrait"})
    # Stills 1200×800 and a clip 1280×1440 — material shot both ways round, and the portrait canvas
    # pads every piece of it rather than cropping any.
    material(conn, s3, memory, tmp_path)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, video_key, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assert 10 <= duration <= 11, "turning the canvas does not change what goes in it"

    assembled = probe(fetch(s3, video_key, tmp_path / "portrait.mp4"))
    tracks = {stream["codec_type"]: stream for stream in assembled["streams"]}
    assert (tracks["video"]["width"], tracks["video"]["height"]) == (1080, 1920)

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
    assert montage._exposed(montage._grey_frames(poster_file)[0]), (
        "the cover must be the opening clip's best moment, not the dark frame it was cut from"
    )


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
    assert 10 <= duration <= 11, "the four that opened went in, the broken one was left out"


def test_a_badly_exposed_photo_is_left_out_rather_than_failing_the_video(
    conn, s3, memory, tmp_path
):
    """A different code path from a file ffmpeg refuses to open: this one decodes fine and
    `_best_moment` itself returns none because the one frame in it fails exposure. The unit tests
    on `choose` never exercise this — they score every piece by hand."""
    montage_id = order(conn, memory, {"target_seconds": 30})
    material(conn, s3, memory, tmp_path)
    _, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(crushed(tmp_path / "crushed.jpg")), media.bucket(), key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assert 10 <= duration <= 11, "the crushed photo had no usable frame and was left out uncounted"


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
