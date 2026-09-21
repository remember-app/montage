"""Live checks of the отсев's proposal on one file — see README.md.

Real everything, like `test_media.py`: a real Postgres, a real bucket, real ffmpeg, and fixtures
ffmpeg draws on the spot. What `judge` makes of a frame with no file behind it is
`test_selection.py`; these are here for the round trip — an original out of the bucket and a verdict
onto the row the api hands to the gallery.

Warning: these wipe the `jobs` table of whatever DATABASE_URL points at, and write into the bucket
R2_ENDPOINT points at.
"""

import media
import selection
import worker

# The fixtures and the ffmpeg-drawn material are `test_media.py`'s — a second copy of a Postgres
# fixture is a second thing to keep true. pytest resolves fixtures by name in this module, so they
# have to be imported by name and not through the module.
from test_media import (  # noqa: F401
    add_item, conn, ffmpeg, memory, photo, read_item, s3, video,
)
from test_montage import material, order, read_montage


def screening_of(conn, item_id):
    """The proposal as the api serves it: `(rejected, why)`, both `null` until the job has run."""
    return conn.execute(
        "SELECT screening_rejected, screening_reason FROM media_items WHERE id = %s", (item_id,)
    ).fetchone()


def only_screening_jobs_for(conn, item_id):
    """Whatever else this database holds is not this test's business — the sweep is over the whole
    table, and a dev stand's own material would otherwise be queued into the run."""
    conn.execute("DELETE FROM jobs WHERE ref IS DISTINCT FROM %s", (item_id,))
    return conn.execute(
        "SELECT count(*) FROM jobs WHERE type = 'media_screening' AND ref = %s", (item_id,)
    ).fetchone()[0]


def test_a_photo_that_becomes_ready_gets_a_proposal_with_nobody_ordering_a_video(
    conn, s3, memory, tmp_path
):
    """The whole point of the job: the organizer opens the gallery and the analysis has already had
    its say, though no video has been ordered and none may ever be."""
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(photo(tmp_path / "original.jpg")), media.bucket(), key)
    worker.enqueue(conn, "media_preview", item_id)

    assert worker.run_once(conn) is True, "the preview, which hands the item on"
    assert screening_of(conn, item_id) == (None, None), "nothing is said before the job has run"
    assert worker.run_once(conn) is True, "and the proposal it queued"

    # Looked at and kept: `false` and not `null`, which is what tells the gallery the difference
    # between a frame the analysis approved and one it has not reached yet.
    assert screening_of(conn, item_id) == (False, None)


def test_a_frame_the_analysis_throws_out_says_why(conn, s3, memory, tmp_path):
    item_id, key = add_item(conn, memory, "photo", status="ready")
    black = tmp_path / "black.jpg"
    ffmpeg("-f", "lavfi", "-i", "color=black:size=1200x800", "-frames:v", "1", str(black))
    s3.upload_file(str(black), media.bucket(), key)
    worker.enqueue(conn, "media_screening", item_id)

    assert worker.run_once(conn) is True

    # A code and not a sentence: the app shows it to the organizer in their own words.
    assert screening_of(conn, item_id) == (True, selection.EXPOSURE)


def test_a_photograph_is_judged_on_its_large_copy_and_a_clip_on_its_original(
    conn, s3, memory, tmp_path
):
    """Which file the verdict is made on — told the only way it can be told from outside: the copy
    and the original are made to say different things about the same item.

    A photograph is judged on the 2048px copy the preview left beside it, which is the very file the
    montage cuts its piece from, so the proposal and the order are made on one frame. A clip is not:
    its preview is a 720p re-encode, smoothed exactly where sharpness is measured.
    """
    black = tmp_path / "black.jpg"
    ffmpeg("-f", "lavfi", "-i", "color=black:size=1200x800", "-frames:v", "1", str(black))

    item_id, key = add_item(conn, memory, "photo", status="ready")
    copy_key = f"previews/{item_id}.jpg"
    s3.upload_file(str(photo(tmp_path / "original.jpg")), media.bucket(), key)
    s3.upload_file(str(black), media.bucket(), copy_key)
    conn.execute("UPDATE media_items SET preview_key = %s WHERE id = %s", (copy_key, item_id))
    worker.enqueue(conn, "media_screening", item_id)

    assert worker.run_once(conn) is True
    assert screening_of(conn, item_id) == (True, selection.EXPOSURE), "снимок судят не по копии"

    clip_id, clip_key = add_item(conn, memory, "video", status="ready")
    black_clip = tmp_path / "black.mp4"
    ffmpeg("-f", "lavfi", "-i", "color=black:size=1280x720:rate=24:duration=3",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(black_clip))
    s3.upload_file(str(video(tmp_path / "clip.mp4")), media.bucket(), clip_key)
    s3.upload_file(str(black_clip), media.bucket(), f"previews/{clip_id}.mp4")
    conn.execute("UPDATE media_items SET preview_key = %s WHERE id = %s",
                 (f"previews/{clip_id}.mp4", clip_id))
    worker.enqueue(conn, "media_screening", clip_id)

    assert worker.run_once(conn) is True
    assert screening_of(conn, clip_id) == (False, None), "клип судят не по оригиналу"


def test_a_copy_the_bucket_will_not_give_back_leaves_the_original_to_judge(
    conn, s3, memory, tmp_path
):
    """The copy is what this job would rather look at, never what it needs: a key that leads
    nowhere costs the item its speed, not its proposal."""
    item_id, key = add_item(conn, memory, "photo", status="ready")
    black = tmp_path / "black.jpg"
    ffmpeg("-f", "lavfi", "-i", "color=black:size=1200x800", "-frames:v", "1", str(black))
    s3.upload_file(str(black), media.bucket(), key)
    conn.execute("UPDATE media_items SET preview_key = 'previews/never-uploaded.jpg' "
                 "WHERE id = %s", (item_id,))
    worker.enqueue(conn, "media_screening", item_id)

    assert worker.run_once(conn) is True

    assert screening_of(conn, item_id) == (True, selection.EXPOSURE)


def test_a_file_that_does_not_read_is_a_proposal_too(conn, s3, memory, tmp_path):
    """«Не посчитано» must not be forever. These bytes are in the bucket and they are not a picture,
    and fetching them again would read exactly the same — so the answer is given now."""
    item_id, key = add_item(conn, memory, "photo", status="ready")
    broken = tmp_path / "original"
    broken.write_text("это не картинка")
    s3.upload_file(str(broken), media.bucket(), key)
    job_id = worker.enqueue(conn, "media_screening", item_id)

    assert worker.run_once(conn) is True

    assert screening_of(conn, item_id) == (True, selection.UNREADABLE)
    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "done"
    # And the item itself is untouched: the analysis thinking little of a photograph does not make it
    # a broken upload. It stays in the gallery, downloadable, one «включить» away from the video.
    assert read_item(conn, item_id)[0] == "ready"


def test_the_sweep_catches_up_material_from_before_the_screening(conn, s3, memory, tmp_path):
    """Material that went `ready` before this job existed: nothing else will ever come for it, and
    the original is still in the bucket, so it costs the member nothing."""
    item_id, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(photo(tmp_path / "original.jpg")), media.bucket(), key)

    worker.sweep_media_without_screening(conn)
    assert only_screening_jobs_for(conn, item_id) == 1
    worker.sweep_media_without_screening(conn)
    assert only_screening_jobs_for(conn, item_id) == 1, "the job on its way is left to finish"

    assert worker.run_once(conn) is True
    assert screening_of(conn, item_id) == (False, None)

    worker.sweep_media_without_screening(conn)
    assert only_screening_jobs_for(conn, item_id) == 1, "and an item with a proposal is not re-asked"


def test_the_sweep_does_not_go_on_asking_for_material_it_has_given_up_on(conn, memory):
    """An original that is not in the bucket at all fails every attempt, and the third gives up. The
    columns stay `null` on purpose — saying «брак» about a file nobody could reach would be a lie,
    and the day's wait is there so an hour of unreachable storage does not cost the gallery its
    proposals for good."""
    item_id, _ = add_item(conn, memory, "photo", status="ready")
    worker.sweep_media_without_screening(conn)
    assert only_screening_jobs_for(conn, item_id) == 1
    for _ in range(worker.MAX_ATTEMPTS):
        conn.execute("UPDATE jobs SET run_after = now() WHERE ref = %s", (item_id,))
        assert worker.run_once(conn) is True

    assert screening_of(conn, item_id) == (None, None)
    worker.sweep_media_without_screening(conn)
    assert only_screening_jobs_for(conn, item_id) == 1, "an hour later it is not asked all over again"

    conn.execute(
        "UPDATE jobs SET updated_at = now() - make_interval(hours => %s) WHERE ref = %s",
        (worker.CATCH_UP_AFTER_HOURS + 1, item_id),
    )
    worker.sweep_media_without_screening(conn)
    assert only_screening_jobs_for(conn, item_id) == 2, "the day after, it is given one more go"


def test_the_proposal_does_not_decide_what_goes_into_a_video(conn, s3, memory, tmp_path):
    """The one thing these two columns may not do. They are what the organizer sees *before*
    ordering; what the video is cut from is still `selection.choose`'s answer at the order, and the
    organizer acts on the proposal through «включить» and «исключить» and nowhere else."""
    montage_id = order(conn, memory, {"target_seconds": 30, "music_track_id": "warm"})
    material(conn, s3, memory, tmp_path)
    # Every last piece of the memory's material marked as rubbish by the analysis.
    conn.execute(
        "UPDATE media_items SET screening_rejected = true, screening_reason = %s "
        "WHERE memory_id = %s",
        (selection.BLUR, memory[0]),
    )
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, _, duration, error_text = read_montage(conn, montage_id)
    assert (status, error_text) == ("ready", None)
    assert 10 <= duration <= 11, "the same video as the one nobody marked at all"
