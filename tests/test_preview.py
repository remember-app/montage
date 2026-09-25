"""Превью загруженного файла: то, что показывает галерея — см. `docs/jobs/preview.md`.

Всё живое: настоящий Postgres, настоящий бакет, настоящий ffmpeg. Материал тесты рисуют себе сами
(`material.py`), так что двоичного в репозитории не лежит ничего.
"""

import pytest
from PIL import Image, ImageChops, ImageOps, ImageStat

from montage import ffmpeg, storage, worker
from montage.media import preview

from checks import near
from material import FIXTURES, jpeg, marked_photo, probe, video
from rows import add_item, fetch, read_item


def test_a_photo_gets_a_thumbnail_and_a_2048px_copy(conn, s3, memory, tmp_path):
    # Bigger than the cap on both sides, so the copy has to be scaled down to it.
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(jpeg(tmp_path / "original.jpg", "3000x2000")), storage.client().bucket, key)
    worker.enqueue(conn, "media_preview", item_id)

    assert worker.run_once(conn) is True

    status, thumbnail_key, preview_key, _ = read_item(conn, item_id)
    assert (status, thumbnail_key, preview_key) == (
        "ready",
        f"thumbnails/{item_id}.jpg",
        f"previews/{item_id}.jpg",
    )
    thumbnail = probe(fetch(s3, thumbnail_key, tmp_path / "thumbnail.jpg"))["streams"][0]
    assert thumbnail["width"] == preview.THUMBNAIL_WIDTH
    assert thumbnail["height"] == 426  # 3000×2000 kept in proportion

    # What full screen actually opens: real pixels, not a stretched grid thumbnail.
    copy = probe(fetch(s3, preview_key, tmp_path / "preview.jpg"))["streams"][0]
    assert copy["codec_name"] == "mjpeg"
    # The long side is the cap exactly; the short one is 3000×2000 in proportion, give or take the
    # pixel ffmpeg rounds it by to keep the size even (which way it rounds varies by version).
    assert (copy["width"], copy["height"]) == (
        preview.PREVIEW_LONG_SIDE,
        pytest.approx(1365, abs=2),
    )


def test_a_video_gets_a_poster_and_a_720p_preview(conn, s3, memory, tmp_path):
    item_id, key = add_item(conn, memory, "video")
    s3.upload_file(str(video(tmp_path / "original.mp4")), storage.client().bucket, key)
    worker.enqueue(conn, "media_preview", item_id)

    assert worker.run_once(conn) is True

    status, thumbnail_key, preview_key, duration = read_item(conn, item_id)
    assert (status, preview_key) == ("ready", f"previews/{item_id}.mp4")
    assert 2.5 <= duration <= 3.5, "the measured length replaces the declared one"

    poster = probe(fetch(s3, thumbnail_key, tmp_path / "poster.jpg"))["streams"][0]
    assert (poster["codec_name"], poster["width"]) == ("mjpeg", preview.THUMBNAIL_WIDTH)

    made = probe(fetch(s3, preview_key, tmp_path / "preview.mp4"))
    assert 2.5 <= float(made["format"]["duration"]) <= 3.5
    tracks = {stream["codec_type"]: stream for stream in made["streams"]}
    assert set(tracks) == {"video", "audio"}, "both tracks survive the re-encode"
    assert tracks["video"]["codec_name"] == "h264"
    assert tracks["video"]["height"] == preview.PREVIEW_HEIGHT
    assert tracks["video"]["width"] == 640  # 1280×1440 kept in proportion


def test_a_preview_run_again_queues_what_comes_after_it_once(conn, s3, memory, tmp_path):
    """The job runs again whenever its worker died after the handler and before the job was marked
    done (`worker.reap`) — and what the first run queued is still waiting then."""
    item_id, key = add_item(conn, memory, "video")
    s3.upload_file(str(video(tmp_path / "original.mp4")), storage.client().bucket, key)

    preview.media_preview(conn, item_id)
    preview.media_preview(conn, item_id)

    queued = conn.execute(
        "SELECT type, count(*) FROM jobs WHERE ref = %s GROUP BY type", (item_id,)
    ).fetchall()
    assert dict(queued) == {
        "media_screening": 1,
        "media_speech": 1,
        "media_frame_info": 1,
        "media_understanding": 1,
    }
    assert read_item(conn, item_id)[0] == "ready"


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
        conn.execute(
            "UPDATE jobs SET run_after = now() WHERE id = %s", (job_id,)
        )  # skip the backoff

    assert (
        conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "failed"
    )
    # And the item with it, so the gallery shows a placeholder instead of a spinner.
    assert read_item(conn, item_id)[0] == ends_as


def test_a_playlist_dressed_up_as_a_photo_is_refused(conn, s3, memory, tmp_path):
    """ffmpeg picks its demuxer by probing, and nothing before this worker looks inside an upload.
    A playlist naming local files would have ffmpeg read them into a preview the uploader can then
    open from the gallery."""
    item_id, key = add_item(conn, memory, "photo")
    playlist = tmp_path / "original"
    playlist.write_text(
        "#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1.0,\nfile:///etc/hosts\n#EXT-X-ENDLIST\n"
    )
    s3.upload_file(str(playlist), storage.client().bucket, key)
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
        ffmpeg.refuse_playlists(playlist)


def previewed(conn, s3, memory, tmp_path, original):
    """The thumbnail and the large copy `media_preview` makes of `original`, as Pillow images."""
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(original), storage.client().bucket, key)
    worker.enqueue(conn, "media_preview", item_id)
    assert worker.run_once(conn) is True
    _, thumbnail_key, preview_key, _ = read_item(conn, item_id)
    return [
        Image.open(fetch(s3, k, tmp_path / f"copy-{n}.jpg")).convert("RGB")
        for n, k in enumerate((thumbnail_key, preview_key))
    ]


@pytest.mark.parametrize("orientation", range(1, 9))
def test_a_jpeg_turned_by_its_exif_is_previewed_standing_up(
    conn, s3, memory, tmp_path, orientation
):
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
    """What an iPhone holds upright is stored as its sensor's frame with an `irot` box saying to
    turn
    it, and ffmpeg 7.1 does not: the thumbnail and the large copy of every portrait photo lay on
    their side. The fixture is 96×64 on the sensor — red in its top-left corner, blue in the
    bottom-right — and to be seen it is 64×96, red at the top-right and blue at the bottom-left."""
    for copy in previewed(conn, s3, memory, tmp_path, FIXTURES / "portrait-photo.heic"):
        assert copy.size == (64, 96)
        assert near(copy.getpixel((60, 4)), (255, 0, 0))
        assert near(copy.getpixel((3, 92)), (0, 0, 255))
