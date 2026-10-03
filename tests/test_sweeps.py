"""Ежечасная уборка: оборванные загрузки и догонялки того, чего очередь не досчиталась.

См. `docs/queue.md`. Всё живое: настоящий Postgres и настоящий бакет.
"""

import datetime
import logging

import pytest
from botocore.exceptions import ClientError
from botocore.stub import Stubber

from montage import storage, sweeps, worker
from montage.video import selection

from material import FIXTURES, jpeg, probe, sound
from pieces import picture
from rows import add_item, fetch, only_jobs_for, read_frame_info, read_item, read_size

EVENING = datetime.datetime(2026, 9, 5, 19, tzinfo=datetime.UTC)


def test_the_sweep_takes_an_abandoned_upload_with_its_multipart(conn, s3, memory):
    abandoned, key = add_item(conn, memory, "photo", status="pending_upload", age_hours=25)
    upload_id = s3.create_multipart_upload(Bucket=storage.client().bucket, Key=key)["UploadId"]
    conn.execute("UPDATE media_items SET upload_id = %s WHERE id = %s", (upload_id, abandoned))
    # Somebody who started uploading a minute ago, and is still uploading.
    fresh, _ = add_item(conn, memory, "photo", status="pending_upload")

    sweeps.sweep_stale_uploads(conn)

    assert read_item(conn, abandoned) is None
    assert read_item(conn, fresh) is not None, "a day has to pass before a row is given up on"
    live = s3.list_multipart_uploads(Bucket=storage.client().bucket, Prefix=key).get("Uploads", [])
    assert upload_id not in [upload["UploadId"] for upload in live], "the parts are released"


def test_the_sweep_deletes_an_upload_finished_in_the_bucket_but_never_reported(
    conn, s3, memory, caplog
):
    """The phone completed its multipart and never told the api: the row is still
    `pending_upload`, the object is in the bucket, and the upload id names an upload that is over.
    The abort is refused as «no such upload» and the object has to go all the same."""
    item_id, key = add_item(conn, memory, "photo", status="pending_upload", age_hours=25)
    upload_id = s3.create_multipart_upload(Bucket=storage.client().bucket, Key=key)["UploadId"]
    part = s3.upload_part(
        Bucket=storage.client().bucket, Key=key, UploadId=upload_id, PartNumber=1, Body=b"photo"
    )
    s3.complete_multipart_upload(
        Bucket=storage.client().bucket,
        Key=key,
        UploadId=upload_id,
        MultipartUpload={"Parts": [{"PartNumber": 1, "ETag": part["ETag"]}]},
    )
    conn.execute("UPDATE media_items SET upload_id = %s WHERE id = %s", (upload_id, item_id))

    with caplog.at_level(logging.WARNING, logger="montage.sweeps"):
        assert sweeps.sweep_stale_uploads(conn) == 1

    assert read_item(conn, item_id) is None
    with pytest.raises(ClientError) as gone:
        s3.head_object(Bucket=storage.client().bucket, Key=key)
    assert storage.missing(gone.value), "the finished original is left in the bucket"
    assert "left behind" not in caplog.text


def test_the_sweep_takes_googles_nothing_there_for_an_answer(conn, memory, caplog):
    """Where S3 and MinIO say «done», Google says «not found»: deleting a key with nothing under it
    is 404 `NoSuchKey` there — and an upload abandoned half-way never put anything under its key —
    while an abort of an upload already over is 404 `NoSuchUpload`. Both are the outcome the sweep
    wants. Taken for failures they would cost a warning for every upload swept, every hour, and
    MinIO, which answers 204, would never show it — so these are Google's answers, word for word."""
    item_id, key = add_item(
        conn, memory, "photo", status="pending_upload", age_hours=25, upload_id="over"
    )
    google = Stubber(storage.client().s3)
    where = {"Bucket": storage.client().bucket, "Key": key}
    google.add_client_error(
        "abort_multipart_upload",
        "NoSuchUpload",
        http_status_code=404,
        expected_params=where | {"UploadId": "over"},
    )
    google.add_client_error(
        "delete_object", "NoSuchKey", http_status_code=404, expected_params=where
    )

    with google, caplog.at_level(logging.WARNING, logger="montage.sweeps"):
        assert sweeps.sweep_stale_uploads(conn) == 1
        google.assert_no_pending_responses()

    assert read_item(conn, item_id) is None
    assert "left behind" not in caplog.text


def test_the_sweep_in_google_frees_under_the_prefix_and_nowhere_else(conn, memory, google, caplog):
    """The same sweep over the google driver: an abandoned upload is aborted and its key deleted
    under the prefix, and a row whose key would lead out of it — nothing the api writes, but the
    row is all the sweep goes by — costs a warning and not a single request."""
    _, key = add_item(conn, memory, "photo", status="pending_upload", age_hours=25, upload_id="up")
    stray, _ = add_item(conn, memory, "photo", status="pending_upload", age_hours=25, upload_id="x")
    conn.execute("UPDATE media_items SET original_key = '../elsewhere' WHERE id = %s", (stray,))
    google.answers = [204, (404, "NoSuchKey")]

    with caplog.at_level(logging.WARNING, logger="montage.sweeps"):
        assert sweeps.sweep_stale_uploads(conn) == 2

    assert [(sent.method, sent.url) for sent in google.sent] == [
        ("DELETE", f"{google.objects}{key}?uploadId=up"),
        ("DELETE", f"{google.objects}{key}"),
    ]
    assert "../elsewhere" in caplog.text and "left behind" in caplog.text
    assert f"{key} but" not in caplog.text, "Google's «nothing there» is not a failure"


def test_the_sweep_leaves_an_upload_that_finished_while_it_ran(conn, memory, monkeypatch):
    """The race the status guard is there for: the row is found stale, and the phone finishes it
    before the sweep gets round to freeing it. The api has counted those bytes by then, so sweeping
    the row would put the memory's counters permanently over and delete a live original."""
    item_id, _ = add_item(conn, memory, "photo", status="pending_upload", age_hours=25)

    # `storage()` is called after the stale rows are read and before any of them is claimed, which
    # is exactly the window the api's «загружено» lands in.
    real_storage = storage.client

    def finish_the_upload_first():
        conn.execute("UPDATE media_items SET status = 'processing' WHERE id = %s", (item_id,))
        return real_storage()

    monkeypatch.setattr(storage, "client", finish_the_upload_first)

    assert sweeps.sweep_stale_uploads(conn) == 0
    assert read_item(conn, item_id)[0] == "processing", "the finished upload survives the sweep"


def test_the_sweep_catches_up_a_photo_that_has_no_large_copy(conn, s3, memory, tmp_path):
    """A photo from before the worker made large copies: `ready`, thumbnail and all, `preview_key`
    empty. Nobody uploads anything again — the original is in the bucket and the sweep finds it."""
    item_id, key = add_item(conn, memory, "photo", status="ready")
    conn.execute(
        "UPDATE media_items SET thumbnail_key = %s WHERE id = %s",
        (f"thumbnails/{item_id}.jpg", item_id),
    )
    s3.upload_file(str(jpeg(tmp_path / "original.jpg")), storage.client().bucket, key)

    sweeps.sweep_photos_without_previews(conn)
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

    sweeps.sweep_photos_without_previews(conn)

    assert only_jobs_for(conn, item_id) == 1, "the job on its way is left to finish"


def test_the_sweep_does_not_go_on_asking_for_a_photo_it_has_given_up_on(conn, memory):
    """The catch-up is given up on like anything else, and the next hour's sweep leaves it that way.

    Three attempts is the whole retry policy of this queue. A sweep that queued a fresh job an hour
    later would make it endless: a `ready` photo stays `ready` (`preview.give_up`), so nothing about
    it ever changes, and one whose original is missing from the bucket would be downloaded and
    encoded three times an hour for as long as the worker runs.
    """
    # An original that never made it into the bucket: every attempt fails, and the third gives up.
    item_id, _ = add_item(conn, memory, "photo", status="ready")
    sweeps.sweep_photos_without_previews(conn)
    assert only_jobs_for(conn, item_id) == 1
    for _ in range(worker.MAX_ATTEMPTS):
        conn.execute("UPDATE jobs SET run_after = now() WHERE ref = %s", (item_id,))
        assert worker.run_once(conn) is True

    sweeps.sweep_photos_without_previews(conn)
    assert only_jobs_for(conn, item_id) == 1, "an hour later it is not asked for all over again"

    # A day later it is, once: what failed may as easily have been an hour of R2 being unreachable,
    # and one such hour must not cost every old photo its large copy for good.
    conn.execute(
        "UPDATE jobs SET updated_at = now() - make_interval(hours => %s) WHERE ref = %s",
        (sweeps.CATCH_UP_AFTER_HOURS + 1, item_id),
    )
    sweeps.sweep_photos_without_previews(conn)
    assert only_jobs_for(conn, item_id) == 2, "the day after, it is given one more go"


def test_the_sweep_reads_the_camera_off_material_from_before(conn, s3, memory):
    """A frame that went `ready` before the worker learned to read one. Nobody uploads anything
    again — the original is in the bucket and the sweep finds it."""
    item_id, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(FIXTURES / "exif-photo.jpg"), storage.client().bucket, key)

    sweeps.sweep_media_without_frame_info(conn)
    assert only_jobs_for(conn, item_id, "media_frame_info") == 1
    assert worker.run_once(conn) is True
    assert read_frame_info(conn, item_id)[0] == "iPhone 15 Pro"

    sweeps.sweep_media_without_frame_info(conn)
    assert only_jobs_for(conn, item_id, "media_frame_info") == 1, "read once, then left alone"


def test_the_sweep_stops_asking_about_a_file_that_carries_nothing(conn, s3, memory, tmp_path):
    """Empty camera columns are an answer, not a gap — and they look exactly like a file nobody has
    opened yet. What the sweep goes by is the size, which this one has like every other picture;
    going by the camera it would fetch and probe most of a library every hour forever."""
    item_id, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(jpeg(tmp_path / "original.jpg")), storage.client().bucket, key)

    sweeps.sweep_media_without_frame_info(conn)
    assert worker.run_once(conn) is True
    assert read_frame_info(conn, item_id) == (None, None, None, None, None)

    sweeps.sweep_media_without_frame_info(conn)
    assert only_jobs_for(conn, item_id, "media_frame_info") == 1, "an hour later, not asked again"


def test_the_sweep_measures_a_frame_that_was_read_before_sizes(conn, s3, memory, tmp_path):
    """A frame the worker had already read back when it took the camera and nothing else: the job
    stands `done` on it — days ago, as it must, or it would be a read that has just failed to find
    a size (below) — and the size is missing all the same. It is measured on the next pass, and
    nobody uploads anything again."""
    item_id, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(jpeg(tmp_path / "original.jpg")), storage.client().bucket, key)
    already_read = worker.enqueue(conn, "media_frame_info", item_id)
    conn.execute(
        "UPDATE jobs SET status = 'done', updated_at = now() - interval '2 days' WHERE id = %s",
        (already_read,),
    )

    sweeps.sweep_media_without_frame_info(conn)
    assert only_jobs_for(conn, item_id, "media_frame_info") == 2, "it is asked for once more"
    assert worker.run_once(conn) is True

    assert read_size(conn, item_id) == (1200, 800)


def test_the_sweep_does_not_go_on_asking_about_a_file_it_cannot_measure(conn, s3, memory, tmp_path):
    """A file with no picture in it ends `done` with the width still null, which is what the sweep
    looks for. Left at that it would queue the job again every hour, and fetch and probe the same
    file until the worker was stopped; a read that finished within the day is a read that has
    been made."""
    item_id, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(sound(tmp_path / "original.m4a")), storage.client().bucket, key)

    sweeps.sweep_media_without_frame_info(conn)
    assert worker.run_once(conn) is True
    assert read_size(conn, item_id) == (None, None)

    assert sweeps.sweep_media_without_frame_info(conn) == 0
    assert only_jobs_for(conn, item_id, "media_frame_info") == 1, "an hour later, not asked again"

    # A day on, the same file gets its one more go — the sweep is not blind to it for good.
    conn.execute("UPDATE jobs SET updated_at = now() - interval '25 hours'")
    assert sweeps.sweep_media_without_frame_info(conn) == 1


def _kept(conn, memory, second, frame, sharpness, *, reason=None, hidden=False):
    """A ready photograph the screening job has already looked at: its verdict on the row and its
    frame where the job leaves it, with no file behind either."""
    item_id, _ = add_item(conn, memory, "photo", status="ready")
    conn.execute(
        """UPDATE media_items
           SET captured_at = %s, screening_rejected = %s, screening_reason = %s, hidden = %s
           WHERE id = %s""",
        (EVENING + datetime.timedelta(seconds=second), reason is not None, reason, hidden, item_id),
    )
    conn.execute(
        "INSERT INTO media_frames (media_id, frame, sharpness) VALUES (%s, %s, %s)",
        (item_id, frame, sharpness),
    )
    return item_id


def _verdict(conn, item_id):
    """The proposal as the api serves it: `(rejected, why)`."""
    return conn.execute(
        "SELECT screening_rejected, screening_reason FROM media_items WHERE id = %s", (item_id,)
    ).fetchone()


def test_the_sweep_counts_the_series_again_after_a_frame_is_hidden(conn, memory):
    """The sharp shot of the cake is hidden, and the softer one it stood for is nobody's retake any
    more — kept again, on the hour, without anybody uploading anything into this memory.

    Hiding queues nothing (the api's `frame.ts` writes the row and stops), so before this sweep the
    mark stands for good: it was true when it was counted and nothing has counted since."""
    first = _kept(conn, memory, 0, picture(1), 50)
    retake = _kept(conn, memory, 10, picture(1, moved=(3, 0)), 30)
    guests = _kept(conn, memory, 15, picture(2), 40)

    sweeps.sweep_media_duplicates(conn)
    assert _verdict(conn, retake) == (True, selection.DUPLICATE), "the mark is there to begin with"

    conn.execute("UPDATE media_items SET hidden = true WHERE id = %s", (first,))
    sweeps.sweep_media_duplicates(conn)

    assert _verdict(conn, retake) == (False, None)
    assert _verdict(conn, guests) == (False, None)


def test_the_sweep_counts_the_series_again_after_a_frame_is_deleted(conn, memory):
    """The same for a frame the organizer deleted outright, and this is the one nothing could have
    found by looking: the row is gone, and so is any trace that it was ever part of a series. Only
    counting the memory as it stands now can answer for what is left of it."""
    first = _kept(conn, memory, 0, picture(1), 50)
    retake = _kept(conn, memory, 10, picture(1, moved=(3, 0)), 30)
    guests = _kept(conn, memory, 15, picture(2), 40)

    sweeps.sweep_media_duplicates(conn)
    assert _verdict(conn, retake) == (True, selection.DUPLICATE)

    conn.execute("DELETE FROM media_items WHERE id = %s", (first,))
    sweeps.sweep_media_duplicates(conn)

    assert _verdict(conn, retake) == (False, None)
    assert _verdict(conn, guests) == (False, None)


def test_the_sweep_counts_the_series_again_after_a_shooting_date_is_corrected(conn, memory):
    """Two shots of the cake a minute apart are two moments of their own; the organizer corrects
    one date and they are one moment again, so the second becomes the retake of the first — a mark
    the recount puts on, not only takes off. The api writes `captured_at` and queues nothing."""
    first = _kept(conn, memory, 0, picture(1), 50)
    retake = _kept(conn, memory, 120, picture(1, moved=(3, 0)), 30)

    sweeps.sweep_media_duplicates(conn)
    assert _verdict(conn, retake) == (False, None), "two minutes apart — two moments of their own"

    conn.execute(
        "UPDATE media_items SET captured_at = captured_at - interval '2 minutes' WHERE id = %s",
        (retake,),
    )
    sweeps.sweep_media_duplicates(conn)

    assert _verdict(conn, retake) == (True, selection.DUPLICATE)
    assert _verdict(conn, first) == (False, None)


def test_the_sweep_writes_nothing_about_a_memory_nobody_touched(conn, memory):
    """Counted every hour, so it is worth saying that it costs nothing: the same material counted
    twice in a row rewrites not one row — `_duplicates` writes only where its answer changed, which
    is what keeps an hourly pass over the library from being a write storm."""
    _kept(conn, memory, 0, picture(1), 50)
    retake = _kept(conn, memory, 10, picture(1, moved=(3, 0)), 30)

    # The count the sweep returns is how many memories it went over, and it goes over the whole
    # library — so it says how big this shared database is by now, not what this test laid out. What
    # the sweep did about this memory is said by the mark below, which only appears if it was
    # reached: `_kept` leaves the retake with no verdict of its own.
    sweeps.sweep_media_duplicates(conn)
    written = conn.execute(
        "SELECT id, updated_at FROM media_items WHERE memory_id = %s ORDER BY id", (memory[0],)
    ).fetchall()

    sweeps.sweep_media_duplicates(conn)
    assert (
        conn.execute(
            "SELECT id, updated_at FROM media_items WHERE memory_id = %s ORDER BY id", (memory[0],)
        ).fetchall()
        == written
    ), "an hour later, not one row rewritten"
    assert _verdict(conn, retake) == (True, selection.DUPLICATE), "and the mark still stands"
