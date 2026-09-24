"""Ежечасная уборка: оборванные загрузки и догонялки того, чего очередь не досчиталась.

См. `docs/queue.md`. Всё живое: настоящий Postgres и настоящий бакет.
"""

import pytest

from montage import storage, sweeps, worker

from material import FIXTURES, jpeg, probe, sound
from rows import add_item, fetch, only_jobs_for, read_frame_info, read_item, read_size


def test_the_sweep_takes_an_abandoned_upload_with_its_multipart(conn, s3, memory):
    abandoned, key = add_item(conn, memory, "photo", status="pending_upload", age_hours=25)
    upload_id = s3.create_multipart_upload(Bucket=storage.bucket(), Key=key)["UploadId"]
    conn.execute("UPDATE media_items SET upload_id = %s WHERE id = %s", (upload_id, abandoned))
    # Somebody who started uploading a minute ago, and is still uploading.
    fresh, _ = add_item(conn, memory, "photo", status="pending_upload")

    sweeps.sweep_stale_uploads(conn)

    assert read_item(conn, abandoned) is None
    assert read_item(conn, fresh) is not None, "a day has to pass before a row is given up on"
    live = s3.list_multipart_uploads(Bucket=storage.bucket(), Prefix=key).get("Uploads", [])
    assert upload_id not in [upload["UploadId"] for upload in live], "the parts are released"


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
    s3.upload_file(str(jpeg(tmp_path / "original.jpg")), storage.bucket(), key)

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
    s3.upload_file(str(FIXTURES / "exif-photo.jpg"), storage.bucket(), key)

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
    s3.upload_file(str(jpeg(tmp_path / "original.jpg")), storage.bucket(), key)

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
    s3.upload_file(str(jpeg(tmp_path / "original.jpg")), storage.bucket(), key)
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
    s3.upload_file(str(sound(tmp_path / "original.m4a")), storage.bucket(), key)

    sweeps.sweep_media_without_frame_info(conn)
    assert worker.run_once(conn) is True
    assert read_size(conn, item_id) == (None, None)

    assert sweeps.sweep_media_without_frame_info(conn) == 0
    assert only_jobs_for(conn, item_id, "media_frame_info") == 1, "an hour later, not asked again"

    # A day on, the same file gets its one more go — the sweep is not blind to it for good.
    conn.execute("UPDATE jobs SET updated_at = now() - interval '25 hours'")
    assert sweeps.sweep_media_without_frame_info(conn) == 1
