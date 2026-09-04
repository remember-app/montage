"""Live checks of the previews and the sweep — see README.md.

Real everything: a real Postgres, a real bucket, real ffmpeg. The fixtures are made by ffmpeg on the
spot rather than committed, so nothing binary lives in the repository.

Warning: these wipe the `jobs` table of whatever DATABASE_URL points at, and write into the bucket
R2_ENDPOINT points at.
"""

import json
import subprocess
import uuid

import pytest

import media
import worker


def ffmpeg(*args):
    subprocess.run(("ffmpeg", "-nostdin", "-loglevel", "error", "-y", *args), check=True)


def probe(path):
    """`{"streams": [...], "format": {...}}`, as ffprobe sees the file."""
    out = subprocess.run(
        ("ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)),
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def photo(path):
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=1200x800:duration=1", "-frames:v", "1", str(path))
    return path


def video(path):
    """Three seconds of picture and sound, taller than 720 so the preview has to scale down."""
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=1280x1440:rate=24:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    )
    return path


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


def fetch(s3, key, path):
    s3.download_file(media.bucket(), key, str(path))
    return path


def test_a_photo_gets_a_thumbnail_and_goes_ready(conn, s3, memory, tmp_path):
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(photo(tmp_path / "original.jpg")), media.bucket(), key)
    worker.enqueue(conn, "media_preview", item_id)

    assert worker.run_once(conn) is True

    status, thumbnail_key, preview_key, _ = read_item(conn, item_id)
    # A photo has nothing to preview beyond its thumbnail — the gallery opens the original.
    assert (status, thumbnail_key, preview_key) == ("ready", f"thumbnails/{item_id}.jpg", None)
    thumbnail = probe(fetch(s3, thumbnail_key, tmp_path / "thumbnail.jpg"))["streams"][0]
    assert thumbnail["width"] == media.THUMBNAIL_WIDTH
    assert thumbnail["height"] == 426  # 1200×800 kept in proportion


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


def test_a_preview_that_keeps_failing_fails_the_item(conn, memory):
    # A row whose original never made it into the bucket: the download fails every time.
    item_id, _ = add_item(conn, memory, "photo")
    job_id = worker.enqueue(conn, "media_preview", item_id)

    for _ in range(worker.MAX_ATTEMPTS):
        assert worker.run_once(conn) is True
        conn.execute("UPDATE jobs SET run_after = now() WHERE id = %s", (job_id,))  # skip the backoff

    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "failed"
    # And the item with it, so the gallery shows a placeholder instead of a spinner.
    assert read_item(conn, item_id)[0] == "failed"


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
    job = conn.execute(
        "SELECT status, last_error FROM jobs WHERE type = 'media_preview' AND ref = %s", (item_id,)
    ).fetchone()
    assert job[0] == "queued", "the first refusal is a retry like any other failure"
    assert "playlist" in job[1]
    assert read_item(conn, item_id)[0] == "processing"
