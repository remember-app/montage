"""Live checks of the previews and the sweep — see README.md.

Real everything: a real Postgres, a real bucket, real ffmpeg. The fixtures are made by ffmpeg on the
spot rather than committed, so nothing binary lives in the repository.

Warning: these wipe the `jobs` table of whatever DATABASE_URL points at, and write into the bucket
R2_ENDPOINT points at.
"""

import json
import logging
import pathlib
import struct
import subprocess
import uuid

import pytest
from PIL import ExifTags, Image, ImageChops, ImageDraw, ImageOps, ImageStat, TiffImagePlugin

import media
import worker

# The photographs with EXIF in them are the one thing here ffmpeg cannot draw: it writes no EXIF,
# and a HEIC it does not write at all. See README.md, «Тесты».
FIXTURES = pathlib.Path(__file__).with_name("fixtures")

# Eight bytes of `exif-photo.heic` overwritten so that ffmpeg still cuts a thumbnail out of it and
# libheif segfaults inside `Image.open` — found by flipping bytes, libheif 1.23. Should a newer libheif
# stop falling over, what the tests below hold to stays true: the worker outlives the file.
CRASHING_HEIC = {153: 163, 179: 220, 228: 183, 450: 174, 484: 87, 606: 93, 616: 114, 619: 181}


def ffmpeg(*args):
    subprocess.run(("ffmpeg", "-nostdin", "-loglevel", "error", "-y", *args), check=True)


def probe(path):
    """`{"streams": [...], "format": {...}}`, as ffprobe sees the file."""
    out = subprocess.run(
        ("ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)),
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def photo(path, size="1200x800"):
    ffmpeg("-f", "lavfi", "-i", f"testsrc=size={size}:duration=1", "-frames:v", "1", str(path))
    return path


def video(path):
    """Three seconds of picture and sound, taller than 720 so the preview has to scale down."""
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=1280x1440:rate=24:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    )
    return path


def sideways_photo(path, size=(1200, 800), orientation=6):
    """A photograph off a phone held sideways: the sensor's own frame, and the EXIF note saying
    which way up it goes. Drawn by Pillow — `ffmpeg` writes no EXIF at all."""
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = orientation
    Image.new("RGB", size, "gray").save(path, exif=exif)
    return path


def upright_video(path):
    """A clip the phone was held upright for: stored on its side, with a display matrix beside it
    saying to stand it up. Written in two passes because that matrix survives only a copy —
    re-encoding turns the picture itself instead and drops it, autorotate doing its job."""
    stored = video(path.with_name("stored.mp4"))
    ffmpeg("-display_rotation", "90", "-i", str(stored), "-c", "copy", str(path))
    return path


def sound(path):
    """A second of tone and no picture at all: a file with nothing whatever to measure."""
    ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac", str(path))
    return path


def phone_video(path, place="+55.5180+037.5300+158.000/"):
    """A clip with the tags a phone puts beside the picture: which camera took it, and where.
    `use_metadata_tags` so they are written under those names and not dropped as unknown."""
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=320x240:rate=24:duration=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-metadata", "com.apple.quicktime.model=iPhone 15 Pro",
        "-metadata", f"com.apple.quicktime.location.ISO6709={place}",
        "-movflags", "use_metadata_tags", str(path),
    )
    return path


def marked_photo(path, orientation):
    """A photograph with something to tell its corners apart, and the EXIF note that says how the
    phone was held: what the sensor wrote is 200×120, and the note turns it (or flips it) as the
    person saw it. Drawn by Pillow — `ffmpeg` writes no EXIF."""
    picture = Image.new("RGB", (200, 120), (128, 128, 128))
    paint = ImageDraw.Draw(picture)
    paint.rectangle([0, 0, 59, 39], fill=(255, 0, 0))
    paint.rectangle([160, 0, 199, 19], fill=(0, 255, 0))
    paint.rectangle([0, 90, 29, 119], fill=(0, 0, 255))
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = orientation
    picture.save(path, exif=exif, quality=95)
    return path


def crashing_heic(path):
    """`exif-photo.heic` damaged so that opening it with libheif crashes the process."""
    data = bytearray((FIXTURES / "exif-photo.heic").read_bytes())
    for offset, value in CRASHING_HEIC.items():
        data[offset] = value
    path.write_bytes(data)
    return path


def photo_with_odd_numbers(path):
    """What some cameras write when they have nothing to say: an f-number over zero, a focal length
    of nothing, and a latitude of two hundred degrees."""
    def rational(number):
        return TiffImagePlugin.IFDRational(number, 1)

    exif = Image.Exif()
    optics = exif.get_ifd(ExifTags.IFD.Exif)
    optics[ExifTags.Base.FNumber] = TiffImagePlugin.IFDRational(1, 0)
    optics[ExifTags.Base.FocalLength] = rational(0)
    place = exif.get_ifd(ExifTags.IFD.GPSInfo)
    place[ExifTags.GPS.GPSLatitudeRef] = "N"
    place[ExifTags.GPS.GPSLatitude] = (rational(200), rational(0), rational(0))
    place[ExifTags.GPS.GPSLongitudeRef] = "E"
    place[ExifTags.GPS.GPSLongitude] = (rational(37), rational(0), rational(0))
    Image.new("RGB", (64, 48), "gray").save(path, exif=exif)
    return path


def gigantic_photo(path):
    """`exif-photo.jpg` whose header claims 30000×30000 — 900 megapixels in 900 bytes. Pillow
    refuses that as a decompression bomb, and ffprobe reports it as a size of nothing."""
    data = bytearray((FIXTURES / "exif-photo.jpg").read_bytes())
    sof = data.index(b"\xff\xc0")  # start of frame: marker, length, precision, height, width
    data[sof + 5:sof + 9] = struct.pack(">HH", 30000, 30000)
    path.write_bytes(data)
    return path


def near(pixel, colour):
    return all(abs(a - b) < 40 for a, b in zip(pixel, colour))


@pytest.fixture
def conn():
    with worker.connect() as connection:
        worker.init_schema(connection)
        connection.execute("DELETE FROM jobs")
        yield connection


@pytest.fixture
def s3():
    return media.storage()


@pytest.fixture
def memory(conn):
    """Somebody's memory to hang media on. `media_items` and everything it points at belong to the
    api's migrations, not to `schema.sql` — without them there is nothing to test against."""
    if conn.execute("SELECT to_regclass('media_items')").fetchone()[0] is None:
        pytest.skip("media_items is missing — run the api's migrations first (see README.md)")
    user_id = conn.execute(
        "INSERT INTO users (name) VALUES ('montage test') RETURNING id"
    ).fetchone()[0]
    memory_id = conn.execute(
        """INSERT INTO memories (organizer_id, title, event_start)
           VALUES (%s, 'montage test', now()) RETURNING id""",
        (user_id,),
    ).fetchone()[0]
    yield memory_id, user_id
    # Everything hung on the memory goes with it, the worker's own tables included: nothing has a
    # foreign key to `media_items` (`schema.sql` applies to databases the api has never touched),
    # so a description left behind here would outlive every row that explains it.
    conn.execute("DELETE FROM media_understandings WHERE memory_id = %s", (str(memory_id),))
    conn.execute("DELETE FROM media_items WHERE memory_id = %s", (memory_id,))
    conn.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
    conn.execute("DELETE FROM users WHERE id = %s", (user_id,))


def add_item(conn, memory, kind, *, size=0, status="processing", age_hours=0, upload_id=None):
    """A `media_items` row as the api leaves it, and the key its original would be under."""
    memory_id, user_id = memory
    item_id = uuid.uuid4()
    conn.execute(
        """INSERT INTO media_items
             (id, memory_id, uploaded_by, kind, status, original_key, size_bytes, upload_id,
              created_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now() - make_interval(hours => %s))""",
        (item_id, memory_id, user_id, kind, status, f"originals/{item_id}", size, upload_id,
         age_hours),
    )
    return str(item_id), f"originals/{item_id}"


def read_item(conn, item_id):
    return conn.execute(
        """SELECT status, thumbnail_key, preview_key, duration_seconds
           FROM media_items WHERE id = %s""",
        (item_id,),
    ).fetchone()


def read_frame_info(conn, item_id):
    """The camera and the place as the sheet «Сведения о кадре» reads them off the row."""
    return conn.execute(
        """SELECT camera_model, focal_length_mm, aperture, latitude, longitude
           FROM media_items WHERE id = %s""",
        (item_id,),
    ).fetchone()


def read_size(conn, item_id):
    """The sheet's «Размер 4032 × 3024», off the same row."""
    return conn.execute(
        "SELECT width, height FROM media_items WHERE id = %s", (item_id,)
    ).fetchone()


def fetch(s3, key, path):
    s3.download_file(media.bucket(), key, str(path))
    return path


def test_a_photo_gets_a_thumbnail_and_a_2048px_copy(conn, s3, memory, tmp_path):
    # Bigger than the cap on both sides, so the copy has to be scaled down to it.
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(photo(tmp_path / "original.jpg", "3000x2000")), media.bucket(), key)
    worker.enqueue(conn, "media_preview", item_id)

    assert worker.run_once(conn) is True

    status, thumbnail_key, preview_key, _ = read_item(conn, item_id)
    assert (status, thumbnail_key, preview_key) == (
        "ready", f"thumbnails/{item_id}.jpg", f"previews/{item_id}.jpg"
    )
    thumbnail = probe(fetch(s3, thumbnail_key, tmp_path / "thumbnail.jpg"))["streams"][0]
    assert thumbnail["width"] == media.THUMBNAIL_WIDTH
    assert thumbnail["height"] == 426  # 3000×2000 kept in proportion

    # What full screen actually opens: real pixels, not a stretched grid thumbnail.
    copy = probe(fetch(s3, preview_key, tmp_path / "preview.jpg"))["streams"][0]
    assert copy["codec_name"] == "mjpeg"
    # The long side is the cap exactly; the short one is 3000×2000 in proportion, give or take the
    # pixel ffmpeg rounds it by to keep the size even (which way it rounds varies by version).
    assert (copy["width"], copy["height"]) == (media.PREVIEW_LONG_SIDE, pytest.approx(1365, abs=2))


def test_a_video_gets_a_poster_and_a_720p_preview(conn, s3, memory, tmp_path):
    item_id, key = add_item(conn, memory, "video")
    s3.upload_file(str(video(tmp_path / "original.mp4")), media.bucket(), key)
    worker.enqueue(conn, "media_preview", item_id)

    assert worker.run_once(conn) is True

    status, thumbnail_key, preview_key, duration = read_item(conn, item_id)
    assert (status, preview_key) == ("ready", f"previews/{item_id}.mp4")
    assert 2.5 <= duration <= 3.5, "the measured length replaces the declared one"

    poster = probe(fetch(s3, thumbnail_key, tmp_path / "poster.jpg"))["streams"][0]
    assert (poster["codec_name"], poster["width"]) == ("mjpeg", media.THUMBNAIL_WIDTH)

    preview = probe(fetch(s3, preview_key, tmp_path / "preview.mp4"))
    assert 2.5 <= float(preview["format"]["duration"]) <= 3.5
    tracks = {stream["codec_type"]: stream for stream in preview["streams"]}
    assert set(tracks) == {"video", "audio"}, "both tracks survive the re-encode"
    assert tracks["video"]["codec_name"] == "h264"
    assert tracks["video"]["height"] == media.PREVIEW_HEIGHT
    assert tracks["video"]["width"] == 640  # 1280×1440 kept in proportion


@pytest.mark.parametrize("started_as, ends_as", [("processing", "failed"), ("ready", "ready")])
def test_a_preview_that_keeps_failing_fails_the_item(conn, memory, started_as, ends_as):
    """Except when the item is already in the gallery: the catch-up sweep re-runs this job on photos
    that have been `ready` for weeks, and one derivative it could not make must not take a picture
    that works away from everybody."""
    # A row whose original never made it into the bucket: the download fails every time.
    item_id, _ = add_item(conn, memory, "photo", status=started_as)
    job_id = worker.enqueue(conn, "media_preview", item_id)

    for _ in range(worker.MAX_ATTEMPTS):
        assert worker.run_once(conn) is True
        conn.execute("UPDATE jobs SET run_after = now() WHERE id = %s", (job_id,))  # skip the backoff

    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "failed"
    # And the item with it, so the gallery shows a placeholder instead of a spinner.
    assert read_item(conn, item_id)[0] == ends_as


def test_the_sweep_takes_an_abandoned_upload_with_its_multipart(conn, s3, memory):
    abandoned, key = add_item(conn, memory, "photo", status="pending_upload", age_hours=25)
    upload_id = s3.create_multipart_upload(Bucket=media.bucket(), Key=key)["UploadId"]
    conn.execute("UPDATE media_items SET upload_id = %s WHERE id = %s", (upload_id, abandoned))
    # Somebody who started uploading a minute ago, and is still uploading.
    fresh, _ = add_item(conn, memory, "photo", status="pending_upload")

    media.sweep_stale_uploads(conn)

    assert read_item(conn, abandoned) is None
    assert read_item(conn, fresh) is not None, "a day has to pass before a row is given up on"
    live = s3.list_multipart_uploads(Bucket=media.bucket(), Prefix=key).get("Uploads", [])
    assert upload_id not in [upload["UploadId"] for upload in live], "the parts are released"


def test_the_sweep_leaves_an_upload_that_finished_while_it_ran(conn, memory, monkeypatch):
    """The race the status guard is there for: the row is found stale, and the phone finishes it
    before the sweep gets round to freeing it. The api has counted those bytes by then, so sweeping
    the row would put the memory's counters permanently over and delete a live original."""
    item_id, _ = add_item(conn, memory, "photo", status="pending_upload", age_hours=25)

    # `storage()` is called after the stale rows are read and before any of them is claimed, which
    # is exactly the window the api's «загружено» lands in.
    real_storage = media.storage

    def finish_the_upload_first():
        conn.execute(
            "UPDATE media_items SET status = 'processing' WHERE id = %s", (item_id,)
        )
        return real_storage()

    monkeypatch.setattr(media, "storage", finish_the_upload_first)

    assert media.sweep_stale_uploads(conn) == 0
    assert read_item(conn, item_id)[0] == "processing", "the finished upload survives the sweep"


def only_jobs_for(conn, item_id, type_="media_preview"):
    """Whatever else this database holds is not this test's business — the sweep is over the whole
    table, and a dev stand's own photos would otherwise be queued into the run."""
    conn.execute("DELETE FROM jobs WHERE ref IS DISTINCT FROM %s", (item_id,))
    return conn.execute(
        "SELECT count(*) FROM jobs WHERE type = %s AND ref = %s", (type_, item_id)
    ).fetchone()[0]


def test_the_sweep_catches_up_a_photo_that_has_no_large_copy(conn, s3, memory, tmp_path):
    """A photo from before the worker made large copies: `ready`, thumbnail and all, `preview_key`
    empty. Nobody uploads anything again — the original is in the bucket and the sweep finds it."""
    item_id, key = add_item(conn, memory, "photo", status="ready")
    conn.execute(
        "UPDATE media_items SET thumbnail_key = %s WHERE id = %s",
        (f"thumbnails/{item_id}.jpg", item_id),
    )
    s3.upload_file(str(photo(tmp_path / "original.jpg")), media.bucket(), key)

    worker.sweep_photos_without_previews(conn)
    assert only_jobs_for(conn, item_id) == 1
    assert worker.run_once(conn) is True

    preview_key = read_item(conn, item_id)[2]
    assert preview_key == f"previews/{item_id}.jpg"
    copy = probe(fetch(s3, preview_key, tmp_path / "preview.jpg"))["streams"][0]
    # 1200×800 is under the cap: the copy is the original's own size, not blown up past it.
    assert (copy["width"], copy["height"]) == (1200, 800)


@pytest.mark.parametrize("already", ["queued", "running"])
def test_the_sweep_does_not_double_a_photo_that_is_already_on_its_way(conn, memory, already):
    item_id, _ = add_item(conn, memory, "photo", status="ready")
    job_id = worker.enqueue(conn, "media_preview", item_id)
    conn.execute("UPDATE jobs SET status = %s WHERE id = %s", (already, job_id))

    worker.sweep_photos_without_previews(conn)

    assert only_jobs_for(conn, item_id) == 1, "the job on its way is left to finish"


def test_the_sweep_does_not_go_on_asking_for_a_photo_it_has_given_up_on(conn, memory):
    """The catch-up is given up on like anything else, and the next hour's sweep leaves it that way.

    Three attempts is the whole retry policy of this queue. A sweep that queued a fresh job an hour
    later would make it endless: a `ready` photo stays `ready` (`media.give_up`), so nothing about it
    ever changes, and one whose original is missing from the bucket would be downloaded and encoded
    three times an hour for as long as the worker runs.
    """
    # An original that never made it into the bucket: every attempt fails, and the third gives up.
    item_id, _ = add_item(conn, memory, "photo", status="ready")
    worker.sweep_photos_without_previews(conn)
    assert only_jobs_for(conn, item_id) == 1
    for _ in range(worker.MAX_ATTEMPTS):
        conn.execute("UPDATE jobs SET run_after = now() WHERE ref = %s", (item_id,))
        assert worker.run_once(conn) is True

    worker.sweep_photos_without_previews(conn)
    assert only_jobs_for(conn, item_id) == 1, "an hour later it is not asked for all over again"

    # A day later it is, once: what failed may as easily have been an hour of R2 being unreachable,
    # and one such hour must not cost every old photo its large copy for good.
    conn.execute(
        "UPDATE jobs SET updated_at = now() - make_interval(hours => %s) WHERE ref = %s",
        (worker.CATCH_UP_AFTER_HOURS + 1, item_id),
    )
    worker.sweep_photos_without_previews(conn)
    assert only_jobs_for(conn, item_id) == 2, "the day after, it is given one more go"


def test_a_playlist_dressed_up_as_a_photo_is_refused(conn, s3, memory, tmp_path):
    """ffmpeg picks its demuxer by probing, and nothing before this worker looks inside an upload.
    A playlist naming local files would have ffmpeg read them into a preview the uploader can then
    open from the gallery."""
    item_id, key = add_item(conn, memory, "photo")
    playlist = tmp_path / "original"
    playlist.write_text(
        "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1.0,\nfile:///etc/hosts\n#EXT-X-ENDLIST\n"
    )
    s3.upload_file(str(playlist), media.bucket(), key)
    worker.enqueue(conn, "media_preview", item_id)

    assert worker.run_once(conn) is True

    # Refused, not previewed: the job carries the reason and the item never went `ready`.
    # Whose reason it is depends on the ffmpeg: `_refuse_playlists` on one that opens a playlist
    # whatever it is called, ffprobe's own on 4.3 and up, which will not touch an m3u8 that is not
    # named like one — and an upload is always the extension-less `original`. Either way nothing is
    # encoded, so the assertion is on the refusal and the test below on the half that is ours.
    job = conn.execute(
        "SELECT status, last_error FROM jobs WHERE type = 'media_preview' AND ref = %s", (item_id,)
    ).fetchone()
    assert job[0] == "queued", "the first refusal is a retry like any other failure"
    assert job[1], "and it says what went wrong"
    assert read_item(conn, item_id)[0] == "processing"


def test_the_guard_refuses_what_ffmpeg_calls_a_playlist(tmp_path):
    """The refusal that is ours rather than ffmpeg's, on the one playlist a current ffmpeg still
    opens unprompted: a concat list naming media beside it."""
    video(tmp_path / "local.mp4")
    playlist = tmp_path / "original"
    playlist.write_text("ffconcat version 1.0\nfile local.mp4\nduration 1\n")

    with pytest.raises(RuntimeError, match="playlist"):
        media._refuse_playlists(playlist)


# --- the size, the camera and the place --------------------------------------
# The lines «Размер», «Камера» и «Место» of the sheet «Сведения о кадре», read off the original.

MOSCOW_SOUTH = (pytest.approx(55.518, abs=0.0005), pytest.approx(37.53, abs=0.0005))


@pytest.mark.parametrize("kind, draw, name, seen_as", [
    ("photo", photo, "original.jpg", (1200, 800)),
    # The sideways phone: 1200×800 on the sensor, and the EXIF note stands it up.
    ("photo", sideways_photo, "sideways.jpg", (800, 1200)),
    ("video", video, "original.mp4", (1280, 1440)),
    # And the clip shot upright: stored 1280×1440, played the other way round.
    ("video", upright_video, "upright.mp4", (1440, 1280)),
])
def test_a_frame_is_measured_the_way_it_is_seen(conn, s3, memory, tmp_path, kind, draw, name,
                                                seen_as):
    """«Размер 4032 × 3024» on the sheet: the pixels a person sees, not the ones the sensor wrote.
    A file turned by its metadata is the ordinary case — every photograph taken with the phone on
    its side, and every clip shot holding it upright."""
    item_id, key = add_item(conn, memory, kind)
    s3.upload_file(str(draw(tmp_path / name)), media.bucket(), key)
    worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    assert read_size(conn, item_id) == seen_as


def test_a_file_with_no_picture_in_it_is_left_unmeasured(conn, s3, memory, tmp_path):
    """Nothing to measure is not a failure: the two columns stay empty, the job is done, and the
    sheet leaves the line out — `kind` being only the phone's word for what it sent."""
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(sound(tmp_path / "original.m4a")), media.bucket(), key)
    job_id = worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    assert read_size(conn, item_id) == (None, None)
    status, = conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert status == "done"


@pytest.mark.parametrize("fixture", ["exif-photo.jpg", "exif-photo.heic"])
def test_a_photograph_brings_its_camera_and_its_place(conn, s3, memory, fixture):
    """HEIC beside JPEG on purpose: HEIC is what an iPhone actually uploads."""
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(FIXTURES / fixture), media.bucket(), key)
    worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    camera, focal_length, aperture, latitude, longitude = read_frame_info(conn, item_id)
    # «iPhone 15 Pro · 24 мм · f/1.8» on the layout — in five columns, because those units are
    # words and the app puts them in the reader's own language.
    assert (camera, focal_length, aperture) == ("iPhone 15 Pro", 24, 1.8)
    assert (latitude, longitude) == MOSCOW_SOUTH


def test_a_clip_brings_its_camera_and_its_place_off_the_container(conn, s3, memory, tmp_path):
    item_id, key = add_item(conn, memory, "video")
    s3.upload_file(str(phone_video(tmp_path / "original.mp4")), media.bucket(), key)
    worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    camera, focal_length, aperture, latitude, longitude = read_frame_info(conn, item_id)
    assert camera == "iPhone 15 Pro"
    assert (latitude, longitude) == MOSCOW_SOUTH
    assert (focal_length, aperture) == (None, None), "a container says nothing about the optics"


@pytest.mark.parametrize("draw, name", [(photo, "original.jpg"), (video, "original.mp4")])
def test_a_file_that_carries_nothing_leaves_the_camera_empty(conn, s3, memory, tmp_path,
                                                             draw, name):
    """Which is most material: a screenshot, a frame re-saved by an editor, anything that came
    through a messenger. Reading it is not a failure and the job says so — nor is a file that is
    no photograph at all, `kind` being only the phone's word for what it sent."""
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(draw(tmp_path / name)), media.bucket(), key)
    job_id = worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    assert read_frame_info(conn, item_id) == (None, None, None, None, None)
    status, = conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert status == "done"


def test_a_finished_preview_asks_for_the_camera_to_be_read(conn, s3, memory, tmp_path):
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(photo(tmp_path / "original.jpg")), media.bucket(), key)
    worker.enqueue(conn, "media_preview", item_id)

    assert worker.run_once(conn) is True

    assert only_jobs_for(conn, item_id, "media_frame_info") == 1


def test_the_sweep_reads_the_camera_off_material_from_before(conn, s3, memory):
    """A frame that went `ready` before the worker learned to read one. Nobody uploads anything
    again — the original is in the bucket and the sweep finds it."""
    item_id, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(FIXTURES / "exif-photo.jpg"), media.bucket(), key)

    worker.sweep_media_without_frame_info(conn)
    assert only_jobs_for(conn, item_id, "media_frame_info") == 1
    assert worker.run_once(conn) is True
    assert read_frame_info(conn, item_id)[0] == "iPhone 15 Pro"

    worker.sweep_media_without_frame_info(conn)
    assert only_jobs_for(conn, item_id, "media_frame_info") == 1, "read once, then left alone"


def test_the_sweep_stops_asking_about_a_file_that_carries_nothing(conn, s3, memory, tmp_path):
    """Empty camera columns are an answer, not a gap — and they look exactly like a file nobody has
    opened yet. What the sweep goes by is the size, which this one has like every other picture;
    going by the camera it would fetch and probe most of a library every hour forever."""
    item_id, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(photo(tmp_path / "original.jpg")), media.bucket(), key)

    worker.sweep_media_without_frame_info(conn)
    assert worker.run_once(conn) is True
    assert read_frame_info(conn, item_id) == (None, None, None, None, None)

    worker.sweep_media_without_frame_info(conn)
    assert only_jobs_for(conn, item_id, "media_frame_info") == 1, "an hour later, not asked again"


def test_the_sweep_measures_a_frame_that_was_read_before_sizes(conn, s3, memory, tmp_path):
    """A frame the worker had already read back when it took the camera and nothing else: the job
    stands `done` on it — days ago, as it must, or it would be a read that has just failed to find
    a size (below) — and the size is missing all the same. It is measured on the next pass, and
    nobody uploads anything again."""
    item_id, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(photo(tmp_path / "original.jpg")), media.bucket(), key)
    already_read = worker.enqueue(conn, "media_frame_info", item_id)
    conn.execute("UPDATE jobs SET status = 'done', updated_at = now() - interval '2 days' "
                 "WHERE id = %s", (already_read,))

    worker.sweep_media_without_frame_info(conn)
    assert only_jobs_for(conn, item_id, "media_frame_info") == 2, "it is asked for once more"
    assert worker.run_once(conn) is True

    assert read_size(conn, item_id) == (1200, 800)


def test_the_sweep_does_not_go_on_asking_about_a_file_it_cannot_measure(conn, s3, memory, tmp_path):
    """A file with no picture in it ends `done` with the width still null, which is what the sweep
    looks for. Left at that it would queue the job again every hour, and fetch and probe the same
    file until the worker was stopped; a read that finished within the day is a read that has
    been made."""
    item_id, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(sound(tmp_path / "original.m4a")), media.bucket(), key)

    worker.sweep_media_without_frame_info(conn)
    assert worker.run_once(conn) is True
    assert read_size(conn, item_id) == (None, None)

    assert worker.sweep_media_without_frame_info(conn) == 0
    assert only_jobs_for(conn, item_id, "media_frame_info") == 1, "an hour later, not asked again"

    # A day on, the same file gets its one more go — the sweep is not blind to it for good.
    conn.execute("UPDATE jobs SET updated_at = now() - interval '25 hours'")
    assert worker.sweep_media_without_frame_info(conn) == 1


def test_the_log_says_what_was_read_and_not_where_it_was(conn, s3, memory, caplog):
    """The place is where somebody was. The sheet shows it to the members of one memory; the log
    is read by whoever runs the worker and kept by whatever collects it."""
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(FIXTURES / "exif-photo.jpg"), media.bucket(), key)
    worker.enqueue(conn, "media_frame_info", item_id)

    with caplog.at_level(logging.INFO):
        assert worker.run_once(conn) is True

    said = caplog.text
    assert item_id in said and "latitude" in said and "longitude" in said
    assert "55.5" not in said and "37.5" not in said


@pytest.mark.parametrize("place", [
    "+5531.08+03731.80/",  # degrees and minutes, which reads as decimal degrees far off the globe
    "+" + "9" * 400 + "+037.5300/",  # a run of digits that is infinity as a float
    "+00.0000+000.0000/",  # what a camera with no satellite fix writes; the Gulf of Guinea, not a place
], ids=["degrees-and-minutes", "infinity", "no-fix"])
def test_a_place_no_globe_can_hold_is_no_place(conn, s3, memory, tmp_path, place):
    """A clip's place tag comes off somebody's phone. What is not a place on the globe — or is the
    point (0, 0) that a camera without a fix writes in its stead — is left out like any other thing
    the file does not say, and the camera beside it is still read."""
    item_id, key = add_item(conn, memory, "video")
    s3.upload_file(str(phone_video(tmp_path / "original.mp4", place)), media.bucket(), key)
    worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    camera, _, _, latitude, longitude = read_frame_info(conn, item_id)
    assert (camera, latitude, longitude) == ("iPhone 15 Pro", None, None)


def test_numbers_a_camera_wrote_that_nothing_can_use_are_left_out(conn, s3, memory, tmp_path):
    """An f-number over zero is NaN, and NaN goes into a double column and out to the sheet as if
    it were a reading. The same file's latitude is two hundred degrees."""
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(photo_with_odd_numbers(tmp_path / "original.jpg")), media.bucket(), key)
    job_id = worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    assert read_frame_info(conn, item_id) == (None, None, None, None, None)
    assert read_size(conn, item_id) == (64, 48)
    status, = conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert status == "done"


def test_south_and_west_are_minus(tmp_path):
    """The hemisphere is a tag of its own next to the degrees, and the one place a sign is lost —
    the fixtures are all north and east."""
    def degrees(whole, minutes):
        return tuple(TiffImagePlugin.IFDRational(number, 1) for number in (whole, minutes, 0))

    exif = Image.Exif()
    place = exif.get_ifd(ExifTags.IFD.GPSInfo)
    place[ExifTags.GPS.GPSLatitudeRef], place[ExifTags.GPS.GPSLatitude] = "S", degrees(33, 52)
    place[ExifTags.GPS.GPSLongitudeRef], place[ExifTags.GPS.GPSLongitude] = "W", degrees(70, 40)
    Image.new("RGB", (32, 32), "gray").save(tmp_path / "original.jpg", exif=exif)

    found = media.photo.read(tmp_path / "original.jpg")

    assert found["latitude"] == pytest.approx(-33.8667, abs=1e-4)
    assert found["longitude"] == pytest.approx(-70.6667, abs=1e-4)


def test_a_photo_that_claims_to_be_a_gigapixel_is_left_unmeasured(conn, s3, memory, tmp_path):
    """The header of a decompression bomb says what nobody will ever decode. Nothing is decoded
    for it — and the size that comes back is none, not a size of zero."""
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(gigantic_photo(tmp_path / "original.jpg")), media.bucket(), key)
    job_id = worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    assert read_size(conn, item_id) == (None, None)
    assert read_frame_info(conn, item_id) == (None, None, None, None, None)
    status, = conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert status == "done"


@pytest.mark.parametrize("model, kept", [
    ("abc\x00def", "abcdef"),  # a NUL, which the `text` column refuses
    ("iPhone\n15\x07 Pro", "iPhone15 Pro"),  # a line break and a bell
    ("M" * 60000, "M" * media.MODEL_LENGTH),  # a paragraph, where a camera's name goes
])
def test_a_camera_name_is_made_a_short_line_of_text(conn, s3, memory, tmp_path, model, kept):
    """The model is the file's own text and every member of the memory is shown it. A NUL in it
    would fail the job three times over and be tried again every day."""
    exif = Image.Exif()
    exif[ExifTags.Base.Model] = model
    Image.new("RGB", (64, 48), "gray").save(tmp_path / "original.jpg", exif=exif)
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(tmp_path / "original.jpg"), media.bucket(), key)
    job_id = worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    status, = conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert status == "done"
    assert read_frame_info(conn, item_id)[0] == kept


def test_a_png_is_measured_by_ffprobe_and_carries_no_camera(conn, s3, memory, tmp_path):
    """PNG is left to ffprobe: Pillow decodes the whole picture to answer `getexif()` on one, and a
    camera does not write PNG in any case."""
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(photo(tmp_path / "original.png")), media.bucket(), key)
    worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    assert read_size(conn, item_id) == (1200, 800)
    assert read_frame_info(conn, item_id) == (None, None, None, None, None)


def test_a_photo_that_crashes_libheif_does_not_take_the_worker_with_it(conn, s3, memory, tmp_path):
    """Opening this HEIC segfaults the library. It is read in a process of its own, so the job ends
    — `done`, with nothing read — instead of the worker ending with it."""
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(crashing_heic(tmp_path / "original.heic")), media.bucket(), key)
    job_id = worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    status, = conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert status == "done"


@pytest.fixture
def reader_script(tmp_path, monkeypatch):
    """Stands in for `photo.py` as the child: whatever the test writes is what runs."""
    script = tmp_path / "reader.py"
    monkeypatch.setattr(media.photo, "__file__", str(script))
    return script


def test_a_reader_that_dies_is_a_photo_with_nothing_to_read(reader_script):
    reader_script.write_text("import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)\n")

    assert media.photo.read(FIXTURES / "exif-photo.jpg") is None


def test_a_reader_that_hangs_is_given_up_on(reader_script, monkeypatch):
    reader_script.write_text("import time\ntime.sleep(60)\n")
    monkeypatch.setattr(media.photo, "TIMEOUT_SECONDS", 1)

    assert media.photo.read(FIXTURES / "exif-photo.jpg") is None


def test_a_preview_is_made_all_the_same_when_the_reader_dies(conn, s3, memory, reader_script):
    """Standing a HEIC up is a nicety of the copies, not a condition of having them: with the reader
    gone, ffmpeg gets the original as it did before there was one."""
    reader_script.write_text("import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)\n")
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(FIXTURES / "portrait-photo.heic"), media.bucket(), key)
    worker.enqueue(conn, "media_preview", item_id)

    assert worker.run_once(conn) is True

    assert read_item(conn, item_id)[0] == "ready"


def test_the_reader_leaves_alone_what_is_not_a_heic(tmp_path):
    out = tmp_path / "upright.jpg"

    assert media.photo.stand_up(FIXTURES / "exif-photo.jpg", out) is False
    assert not out.exists()


# --- a photograph turned by its own metadata, previewed ---------------------------


def previewed(conn, s3, memory, tmp_path, original):
    """The thumbnail and the large copy `media_preview` makes of `original`, as Pillow images."""
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(original), media.bucket(), key)
    worker.enqueue(conn, "media_preview", item_id)
    assert worker.run_once(conn) is True
    _, thumbnail_key, preview_key, _ = read_item(conn, item_id)
    return [Image.open(fetch(s3, k, tmp_path / f"copy-{n}.jpg")).convert("RGB")
            for n, k in enumerate((thumbnail_key, preview_key))]


@pytest.mark.parametrize("orientation", range(1, 9))
def test_a_jpeg_turned_by_its_exif_is_previewed_standing_up(conn, s3, memory, tmp_path,
                                                            orientation):
    """The gallery shows these copies, and the model that describes a frame is sent one. Pillow's
    own reading of the note is the truth — the picture as the person saw it — and both copies are
    held to it, for every one of the eight ways a phone can be held. ffmpeg does the turning itself
    (7.1.5 in the image); the day a base image brings one that does not, this is what says so."""
    original = marked_photo(tmp_path / "original.jpg", orientation)
    seen = ImageOps.exif_transpose(Image.open(original)).convert("RGB")

    for copy in previewed(conn, s3, memory, tmp_path, original):
        assert copy.size == seen.size
        assert sum(ImageStat.Stat(ImageChops.difference(copy, seen)).mean) / 3 < 8


def test_a_portrait_heic_is_previewed_standing_up(conn, s3, memory, tmp_path):
    """What an iPhone holds upright is stored as its sensor's frame with an `irot` box saying to turn
    it, and ffmpeg 7.1 does not: the thumbnail and the large copy of every portrait photo lay on
    their side. The fixture is 96×64 on the sensor — red in its top-left corner, blue in the
    bottom-right — and to be seen it is 64×96, red at the top-right and blue at the bottom-left."""
    for copy in previewed(conn, s3, memory, tmp_path, FIXTURES / "portrait-photo.heic"):
        assert copy.size == (64, 96)
        assert near(copy.getpixel((60, 4)), (255, 0, 0))
        assert near(copy.getpixel((3, 92)), (0, 0, 255))


def test_a_portrait_heic_is_measured_the_way_it_is_seen(conn, s3, memory):
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(FIXTURES / "portrait-photo.heic"), media.bucket(), key)
    worker.enqueue(conn, "media_frame_info", item_id)

    assert worker.run_once(conn) is True

    assert read_size(conn, item_id) == (64, 96)
    assert read_frame_info(conn, item_id)[0] == "iPhone 15 Pro"
