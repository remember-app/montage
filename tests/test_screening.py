"""Live checks of the отсев's proposal on one file, and on the retakes of a memory — see README.md.

Real everything, like `test_media.py`: a real Postgres, a real bucket, real ffmpeg, and fixtures
ffmpeg draws on the spot. What `judge` makes of a frame with no file behind it is
`test_selection.py`; these are here for the round trip — an original out of the bucket and a verdict
onto the row the api hands to the gallery. Counting a memory's series again is the one thing checked
over rows alone: frames made up in the test (`pieces.picture`) where the job would have left them.

Warning: these wipe the `jobs` table of whatever DATABASE_URL points at, and write into the bucket
R2_ENDPOINT points at.
"""

import datetime

from montage import storage, sweeps, worker
from montage.media import screening
from montage.video import montage as montages
from montage.video import selection

from material import draw, jpeg, video
from pieces import picture
from rows import (
    add_item,
    api_params,
    entries_of,
    journal_ready,
    layout_of,
    order,
    read_item,
    read_montage,
    ready_material,
)

EVENING = datetime.datetime(2026, 9, 5, 19, tzinfo=datetime.UTC)


def screening_of(conn, item_id):
    """The proposal as the api serves it: `(rejected, why)`, both `null` until the job has run."""
    return conn.execute(
        "SELECT screening_rejected, screening_reason FROM media_items WHERE id = %s", (item_id,)
    ).fetchone()


def shots(tmp_path):
    """A picture, a softer retake of the same picture, and something else altogether — what a phone
    leaves after the cake is photographed twice and the guests once in between."""
    sharp, soft, other = tmp_path / "cake.jpg", tmp_path / "cake-again.jpg", tmp_path / "guests.jpg"
    jpeg(sharp)
    draw("-i", str(sharp), "-vf", "gblur=sigma=6", str(soft))
    draw("-f", "lavfi", "-i", "smptebars=size=1200x800", "-frames:v", "1", str(other))
    return sharp, soft, other


def shot(conn, s3, memory, path, second):
    """A photograph taken `second` seconds into the evening, uploaded and `ready`, not screened."""
    item_id, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(path), storage.client().bucket, key)
    conn.execute(
        "UPDATE media_items SET captured_at = %s WHERE id = %s",
        (EVENING + datetime.timedelta(seconds=second), item_id),
    )
    return item_id


def screen(conn, *item_ids):
    for item_id in item_ids:
        worker.enqueue(conn, "media_screening", item_id)
        assert worker.run_once(conn) is True


def kept(conn, memory, second, frame, sharpness, *, reason=None, hidden=False):
    """A ready photograph the job has already looked at: its verdict on the row and its frame where
    the job leaves it, with no file behind either."""
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


def all_but_the_proposal(conn, item_id):
    """Every column of the row but the two the proposal is written in and the time it was — what the
    gallery shows of the frame, and whatever anybody else may ever keep on it."""
    return conn.execute(
        """SELECT to_jsonb(item) - 'screening_rejected' - 'screening_reason' - 'updated_at'
           FROM media_items item WHERE id = %s""",
        (item_id,),
    ).fetchone()[0]


def proposals(conn, memory):
    """Every row of the memory: id → `(rejected, why, updated_at)`."""
    return {
        str(item_id): rest
        for item_id, *rest in conn.execute(
            """SELECT id, screening_rejected, screening_reason, updated_at
               FROM media_items WHERE memory_id = %s""",
            (memory[0],),
        )
    }


def test_the_analysis_job_screens_the_whole_memory_in_one_pass(conn, s3, memory, tmp_path):
    """«Analyze» is one job for the memory, not one per frame (m-43): every ready visible item is
    judged and the series are counted once over all of them. The softer shot of the cake is marked
    a duplicate by the same pass that judged it — the answer a per-file recount would also reach,
    but the recount itself happens once (the next test pins that)."""
    sharp, soft, other = shots(tmp_path)
    first = shot(conn, s3, memory, sharp, 0)
    retake = shot(conn, s3, memory, soft, 10)
    guests = shot(conn, s3, memory, other, 4)

    worker.enqueue(conn, "media_analysis", str(memory[0]))
    assert worker.run_once(conn) is True

    assert screening_of(conn, first) == (False, None)
    assert screening_of(conn, retake) == (True, selection.DUPLICATE)
    assert screening_of(conn, guests) == (False, None)


def test_the_analysis_counts_the_series_once_for_the_whole_memory(
    conn, s3, memory, tmp_path, monkeypatch
):
    """The recount is not run per file — that would be one pass over the memory per frame. Three
    files are judged and the series are counted exactly once at the end."""
    counted = []
    real = screening._duplicates

    def count(conn, memory_id):
        counted.append(str(memory_id))
        return real(conn, memory_id)

    monkeypatch.setattr(screening, "_duplicates", count)
    sharp, soft, other = shots(tmp_path)
    for path, at in ((sharp, 0), (soft, 10), (other, 4)):
        shot(conn, s3, memory, path, at)

    worker.enqueue(conn, "media_analysis", str(memory[0]))
    assert worker.run_once(conn) is True

    assert counted == [str(memory[0])], "one recount for the memory, not one per file"


def test_an_analysis_asked_for_again_reads_nothing_and_rewrites_nothing(conn, s3, memory, tmp_path):
    """Idempotent: once every file has its verdict, another «Analyze» fetches and decodes nothing.
    The originals are gone from the bucket, and it still finishes cleanly — nothing is read that
    was already read, and not one row is rewritten."""
    sharp, soft, other = shots(tmp_path)
    for path, at in ((sharp, 0), (soft, 10), (other, 4)):
        shot(conn, s3, memory, path, at)

    worker.enqueue(conn, "media_analysis", str(memory[0]))
    assert worker.run_once(conn) is True
    written = proposals(conn, memory)

    for item_id in written:
        s3.delete_object(Bucket=storage.client().bucket, Key=f"originals/{item_id}")

    job_id = worker.enqueue(conn, "media_analysis", str(memory[0]))
    assert worker.run_once(conn) is True
    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "done"
    assert proposals(conn, memory) == written, "nothing re-judged, not one row rewritten"


def test_a_frame_the_analysis_throws_out_says_why(conn, s3, memory, tmp_path):
    item_id, key = add_item(conn, memory, "photo", status="ready")
    black = tmp_path / "black.jpg"
    draw("-f", "lavfi", "-i", "color=black:size=1200x800", "-frames:v", "1", str(black))
    s3.upload_file(str(black), storage.client().bucket, key)
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
    draw("-f", "lavfi", "-i", "color=black:size=1200x800", "-frames:v", "1", str(black))

    item_id, key = add_item(conn, memory, "photo", status="ready")
    copy_key = f"previews/{item_id}.jpg"
    s3.upload_file(str(jpeg(tmp_path / "original.jpg")), storage.client().bucket, key)
    s3.upload_file(str(black), storage.client().bucket, copy_key)
    conn.execute("UPDATE media_items SET preview_key = %s WHERE id = %s", (copy_key, item_id))
    worker.enqueue(conn, "media_screening", item_id)

    assert worker.run_once(conn) is True
    assert screening_of(conn, item_id) == (True, selection.EXPOSURE), "снимок судят не по копии"

    clip_id, clip_key = add_item(conn, memory, "video", status="ready")
    black_clip = tmp_path / "black.mp4"
    draw(
        "-f",
        "lavfi",
        "-i",
        "color=black:size=1280x720:rate=24:duration=3",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(black_clip),
    )
    s3.upload_file(str(video(tmp_path / "clip.mp4")), storage.client().bucket, clip_key)
    s3.upload_file(str(black_clip), storage.client().bucket, f"previews/{clip_id}.mp4")
    conn.execute(
        "UPDATE media_items SET preview_key = %s WHERE id = %s",
        (f"previews/{clip_id}.mp4", clip_id),
    )
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
    draw("-f", "lavfi", "-i", "color=black:size=1200x800", "-frames:v", "1", str(black))
    s3.upload_file(str(black), storage.client().bucket, key)
    conn.execute(
        "UPDATE media_items SET preview_key = 'previews/never-uploaded.jpg' WHERE id = %s",
        (item_id,),
    )
    worker.enqueue(conn, "media_screening", item_id)

    assert worker.run_once(conn) is True

    assert screening_of(conn, item_id) == (True, selection.EXPOSURE)


def test_a_file_that_does_not_read_is_a_proposal_too(conn, s3, memory, tmp_path):
    """«Не посчитано» must not be forever. These bytes are in the bucket and they are not a picture,
    and fetching them again would read exactly the same — so the answer is given now."""
    item_id, key = add_item(conn, memory, "photo", status="ready")
    broken = tmp_path / "original"
    broken.write_text("это не картинка")
    s3.upload_file(str(broken), storage.client().bucket, key)
    job_id = worker.enqueue(conn, "media_screening", item_id)

    assert worker.run_once(conn) is True

    assert screening_of(conn, item_id) == (True, selection.UNREADABLE)
    assert conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()[0] == "done"
    # And the item itself is untouched: the analysis thinking little of a photograph does not
    # make it a broken upload. It stays in the gallery, downloadable, one «включить» away from
    # the video.
    assert read_item(conn, item_id)[0] == "ready"


def test_an_order_gets_the_screening_nobody_asked_for(conn, s3, memory, tmp_path):
    """An order without «Analyze» must pass: the montage's analysis stage screens the material
    itself, so the frame has a verdict although nobody pressed the button."""
    item_id, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(jpeg(tmp_path / "original.jpg")), storage.client().bucket, key)
    assert screening_of(conn, item_id) == (None, None)

    montages.analysis(conn, str(memory[0]), [selection.Piece(media_id=item_id, kind="photo")])

    assert screening_of(conn, item_id) == (False, None)
    assert (
        conn.execute(
            "SELECT count(*) FROM media_frames WHERE media_id = %s", (item_id,)
        ).fetchone()[0]
        == 1
    )


def test_a_memory_the_order_already_judged_is_not_fetched_again(conn, memory, monkeypatch):
    """The order's preparation judges every file before the analysis stage, so screening is handed
    those verdicts: the memory is not downloaded and decoded a second time for the same answer."""
    item_id, _ = add_item(conn, memory, "photo", status="ready")  # nothing in the bucket

    def refuse(*_, **__):
        raise AssertionError("the memory was fetched a second time")

    monkeypatch.setattr(storage, "download", refuse)
    monkeypatch.setattr(storage, "large_copy", refuse)
    judged = selection.Piece(
        media_id=item_id, kind="photo", rejected=selection.BLUR, frame=b"frame", sharpness=0.1
    )

    screening.screen_memory(conn, str(memory[0]), [judged])

    assert screening_of(conn, item_id) == (True, selection.BLUR)


def test_the_hourly_sweeps_do_not_start_analysis(conn, s3, memory, tmp_path):
    """An item nobody analysed is not analysed by the hour (m-43, decision-22): the отсев and
    everything a cloud model answers are a person's request — «Analyze» or an order. The preview
    and the camera catch-ups stay, and they are the only jobs the sweeps put in the queue."""
    item_id, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(jpeg(tmp_path / "original.jpg")), storage.client().bucket, key)

    for sweep in sweeps.SWEEPS:
        sweep(conn)

    types = {type_ for (type_,) in conn.execute("SELECT type FROM jobs WHERE ref = %s", (item_id,))}
    assert types <= {"media_preview", "media_frame_info"}, (
        f"a sweep queued analysis without being asked: {sorted(types)}"
    )


def test_the_proposal_does_not_decide_what_goes_into_a_video(conn, s3, memory, tmp_path):
    """The one thing these two columns may not do. They are what the organizer sees *before*
    ordering; what the video is cut from is still `selection.choose`'s answer at the order, and the
    organizer acts on the proposal through «включить» and «исключить» and nowhere else."""
    montage_id = order(conn, memory, {"target_seconds": 30, "music_track_id": "warm"})
    ready_material(conn, s3, memory, tmp_path)
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
    assert 8.5 <= duration <= 9.5, "the same video as the one nobody marked at all"


def test_a_retake_is_marked_a_duplicate_before_anybody_orders_a_video(conn, s3, memory, tmp_path):
    """The softer shot of the cake is proposed as a duplicate the moment the sharper one comes in —
    on its own row, though the job that ran was the other file's.

    Only the proposal changes: the retake stays in the gallery as it was, `ready` and not hidden,
    with every other column of the row the gallery reads left alone. And в ролик оно не идёт
    «мимо»: отсев внутри сборки больше нет, и отмеченный повтор — обычный кадр, как и сосед по
    серии (doc-03, «What it doesn't»)."""
    journal_ready(conn)
    sharp, soft, other = shots(tmp_path)
    retake = shot(conn, s3, memory, soft, 10)
    guests = shot(conn, s3, memory, other, 4)
    screen(conn, retake, guests)
    assert screening_of(conn, retake) == (False, None), "alone, it is nobody's retake"

    before = all_but_the_proposal(conn, retake)
    first = shot(conn, s3, memory, sharp, 0)
    screen(conn, first)

    assert screening_of(conn, retake) == (True, selection.DUPLICATE)
    assert screening_of(conn, first) == (False, None)
    assert screening_of(conn, guests) == (False, None), "another picture between the two"
    assert all_but_the_proposal(conn, retake) == before, "still ready, not hidden, the same copies"

    montage_id = order(conn, memory, api_params(target_seconds=10))
    worker.enqueue(conn, "montage", montage_id)
    assert worker.run_once(conn) is True

    assert not [one for one in entries_of(conn, montage_id) if one.reason == selection.DUPLICATE], (
        "в журнале ролика никакого «мимо, дубль» нет — отсева в сборке не осталось"
    )
    assert {first, retake} <= {media_id for media_id, _, _ in layout_of(conn, montage_id)}, (
        "и оба кадра серии стоят в ролике"
    )


def test_the_series_are_counted_over_the_memory_from_scratch(conn, memory):
    """Every count is the whole memory as it is now, and nothing of an earlier count survives it:
    asked twice of the same material, the answer is the same and not one row is written the second
    time; a new shot that moves the series moves the marks with it — onto a frame that was kept and
    off one that was a retake, which is simply kept again.

    What the file alone was turned down for is not the recount's to touch, and it does not take part
    in it either, however sharp: at the order `selection.choose` puts it aside before its dedup. A
    hidden frame is not material at all."""
    cake = kept(conn, memory, 0, picture(1), 50)
    retake = kept(conn, memory, 10, picture(1, moved=(3, 0)), 30)
    guests = kept(conn, memory, 15, picture(2), 40)
    blurred = kept(conn, memory, 12, picture(1), 100, reason=selection.BLUR)
    hidden = kept(conn, memory, 5, picture(1), 200, hidden=True)

    screening._duplicates(conn, memory[0])
    counted = proposals(conn, memory)
    assert {item_id: tuple(verdict[:2]) for item_id, verdict in counted.items()} == {
        cake: (False, None),
        retake: (True, selection.DUPLICATE),
        guests: (False, None),
        blurred: (True, selection.BLUR),
        hidden: (False, None),
    }

    screening._duplicates(conn, memory[0])
    assert proposals(conn, memory) == counted, "the same material, the same answer, nothing written"

    # Twenty-five seconds before the cake, the same cake, sharper: the series now starts there, and
    # the retake at ten seconds is thirty-five past its first shot — a moment of its own.
    earlier = kept(conn, memory, -25, picture(1, moved=(0, 2)), 60)
    screening._duplicates(conn, memory[0])

    recounted = proposals(conn, memory)
    assert {item_id: tuple(verdict[:2]) for item_id, verdict in recounted.items()} == {
        earlier: (False, None),
        cake: (True, selection.DUPLICATE),
        retake: (False, None),
        guests: (False, None),
        blurred: (True, selection.BLUR),
        hidden: (False, None),
    }
    assert recounted[guests] == counted[guests], "a row whose answer held is not rewritten"


def test_a_recount_after_the_order_never_reaches_the_video(conn, s3, memory, tmp_path):
    """Предложение отсева — это предложение, и пересчёт после заказа его не отменяет: кадр,
    пришедший между заказом и сборкой, пересчитывает память заново, а заказ остаётся таким, каким
    его прислали. И в ролике дедупа нет: кадры, оставленные при выборе, равны, так что и отмеченный
    повтор, и его сосед по серии стоят в ролике наравне (doc-03, «What it doesn't»)."""
    sharp, soft, other = shots(tmp_path)
    first = shot(conn, s3, memory, sharp, 0)
    guests = shot(conn, s3, memory, other, 4)
    retake = shot(conn, s3, memory, soft, 10)
    screen(conn, first, guests, retake)
    assert screening_of(conn, retake) == (True, selection.DUPLICATE)

    params = api_params(target_seconds=10)
    montage_id = order(conn, memory, params)
    late = tmp_path / "late.jpg"
    draw("-f", "lavfi", "-i", "mandelbrot=size=1200x800", "-frames:v", "1", str(late))
    screen(conn, shot(conn, s3, memory, late, 120))

    assert screening_of(conn, retake) == (True, selection.DUPLICATE), "предложение на месте"
    stored = conn.execute("SELECT params FROM montages WHERE id = %s", (montage_id,)).fetchone()[0]
    assert stored == params, "the recount does not touch the order"

    worker.enqueue(conn, "montage", montage_id)
    assert worker.run_once(conn) is True

    assert read_montage(conn, montage_id)[0] == "ready"
    in_the_video = {media_id for media_id, _, _ in layout_of(conn, montage_id)}
    assert {first, guests, retake} <= in_the_video, "отсев в ролике не выбрасывает кадры"


def test_the_analysis_looks_again_at_a_frame_kept_before_its_frame_was(conn, s3, memory, tmp_path):
    """A frame an older worker kept has a verdict and no frame to compare: until it is looked at
    once more, the memory's series are counted without it. Once is enough — a kept file leaves its
    frame behind — and a frame turned down on its own never takes part in a series, so it is not
    judged again."""
    item_id, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(jpeg(tmp_path / "original.jpg")), storage.client().bucket, key)
    conn.execute("UPDATE media_items SET screening_rejected = false WHERE id = %s", (item_id,))
    blurred, _ = add_item(conn, memory, "photo", status="ready")
    conn.execute(
        "UPDATE media_items SET screening_rejected = true, screening_reason = %s WHERE id = %s",
        (selection.BLUR, blurred),
    )

    screening.screen_memory(conn, str(memory[0]))
    assert screening_of(conn, item_id) == (False, None)
    assert (
        conn.execute(
            "SELECT count(*) FROM media_frames WHERE media_id = %s", (item_id,)
        ).fetchone()[0]
        == 1
    )
    assert (
        conn.execute(
            "SELECT count(*) FROM media_frames WHERE media_id = %s", (blurred,)
        ).fetchone()[0]
        == 0
    ), "a frame turned down on its own takes no part in a series"

    s3.delete_object(Bucket=storage.client().bucket, Key=f"originals/{item_id}")
    screening.screen_memory(conn, str(memory[0]))
    assert screening_of(conn, item_id) == (False, None), "and it is not judged again"
