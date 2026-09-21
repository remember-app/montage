"""Montage worker: takes jobs from the shared `jobs` table in Postgres, one handler per job.

    python worker.py init                  # create the worker's own tables (idempotent)
    python worker.py tracks                # the music library into the bucket (idempotent)
    python worker.py enqueue <type> [ref]  # put one job in the queue
    python worker.py run                   # the loop, Ctrl-C to stop

Needs DATABASE_URL — except `tracks`, which needs the four storage variables. See README.md.
"""

import logging
import os
import pathlib
import sys
import time

import psycopg

import ai
import assembly
import media
import montage as montages  # the module and its handler share the job's name; the import cannot
import screening
import speech
import understanding

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 30  # attempt n is retried after BACKOFF_SECONDS * 2 ** (n - 1)
IDLE_POLL_SECONDS = 30  # fallback only: a queued job normally wakes the worker by NOTIFY
SWEEP_INTERVAL_SECONDS = 3600  # «раз в час» (architecture.md, «Обработка ошибок»)
CATCH_UP_AFTER_HOURS = 24  # how long a catch-up that failed is left alone before it is tried again
SCHEMA = pathlib.Path(__file__).with_name("schema.sql")

log = logging.getLogger("montage")


def connect():
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit("DATABASE_URL is not set — see README.md")
    return psycopg.connect(url, autocommit=True)


def init_schema(conn):
    conn.execute(SCHEMA.read_text())


def enqueue(conn, type_, ref=None):
    return conn.execute(
        "INSERT INTO jobs (type, ref) VALUES (%s, %s) RETURNING id", (type_, ref)
    ).fetchone()[0]


# --- handlers ---------------------------------------------------------------
# Every handler takes the worker's connection and the job's `ref`.


def dummy(conn, ref):
    """Does nothing and succeeds — unless ref is 'fail', which exercises the retry path."""
    if ref == "fail":
        raise RuntimeError("dummy job was asked to fail")
    log.info("dummy job did nothing, successfully")


HANDLERS = {
    "dummy": dummy,
    "media_preview": media.media_preview,
    "media_frame_info": media.media_frame_info,
    "media_screening": screening.media_screening,
    "media_speech": speech.transcribe,
    "media_understanding": understanding.media_understanding,
    "montage": montages.montage,
}


# --- the sweeps -------------------------------------------------------------
# Housekeeping the loop runs at start-up and then once an interval, beside `media`'s own sweep.


def _catch_up(conn, type_, missing):
    """Queue `type_` for every ready item that `missing` picks out — and only where nothing is
    already doing it.

    `missing` is the condition over `media_items item` saying what this item has not got; it is the
    one thing that differs between the catch-ups below, and it is a constant of this module, never
    anything a request carries. Everything around it is the same for all of them, which is why it
    is written once:

    an item already `queued` or `running` is left to finish, so an hourly sweep does not pile jobs
    on one that is simply still working — and one whose catch-up has just been given up on is left
    alone for a day. Three attempts is the whole retry policy (`run_once`), and a sweep that queued
    a fresh job the next hour would make it endless: an original that is not in the bucket at all
    would be downloaded and encoded three times an hour for as long as the worker runs, and a frame
    the model chokes on would be *paid for* three times an hour. A day later it is tried once more,
    because what failed may as easily have been an hour of R2 being unreachable or a day's limit on
    the account, and one such hour must not cost every old item its large copy — or its description
    — for good. Returns how many were queued.
    """
    return conn.execute(
        f"""INSERT INTO jobs (type, ref)
            SELECT %s, item.id::text FROM media_items item
            WHERE item.status = 'ready' AND {missing}
              AND NOT EXISTS (
                SELECT 1 FROM jobs job
                WHERE job.type = %s AND job.ref = item.id::text
                  AND (job.status IN ('queued', 'running')
                       OR (job.status = 'failed'
                           AND job.updated_at > now() - make_interval(hours => %s))))""",
        (type_, type_, CATCH_UP_AFTER_HOURS),
    ).rowcount


def sweep_photos_without_previews(conn):
    """Photos that are `ready` but carry no large copy, given the job that makes one.

    Every photo taken in before the worker learned to make one is like this, and nothing else will
    ever come for them: the api queues `media_preview` at upload and the upload is long over. The
    original is still in the bucket, so this costs the member nothing — no re-upload, no prompt.
    """
    queued = _catch_up(conn, "media_preview", "item.kind = 'photo' AND item.preview_key IS NULL")
    if queued:
        log.info("queued a large copy for %s photo(s) that had none", queued)
    return queued


def sweep_media_without_frame_info(conn):
    """Ready material nobody has read the camera and the place off yet, given the job that reads.

    Everything taken in before the worker learned to read them is like this: `media_preview` queues
    the job as an item goes `ready`, and for old material that moment is long past. The original is
    still in the bucket, so this costs the member nothing.

    What says «this file has been read» is the job and not the columns: a frame whose camera wrote
    nothing down keeps five nulls and is indistinguishable from an unread one, so a sweep that went
    by the columns would fetch and probe most of the library once an hour forever. Hence the
    condition below — anything but a job that failed, which `_catch_up` gives one more go a day
    later like every other catch-up here.
    """
    queued = _catch_up(
        conn, "media_frame_info",
        """NOT EXISTS (
             SELECT 1 FROM jobs looked
             WHERE looked.type = 'media_frame_info' AND looked.ref = item.id::text
               AND looked.status <> 'failed')""",
    )
    if queued:
        log.info("queued a camera and a place for %s ready item(s) that had none", queued)
    return queued


def sweep_media_without_screening(conn):
    """Ready material the отсев has not looked at yet, given the job that looks.

    Everything taken in before the worker learned to screen is like this: the proposal is queued the
    moment an item goes `ready` (`media.media_preview`), and for old material that moment is long
    past. Until the job has run both columns are `null` — «не посчитано», which the gallery shows as
    no opinion at all rather than as a frame the analysis kept.
    """
    queued = _catch_up(conn, "media_screening", "item.screening_rejected IS NULL")
    if queued:
        log.info("queued a screening proposal for %s ready item(s) that had none", queued)
    return queued


def sweep_media_without_ai_answers(conn):
    """Ready media the cloud model was never asked about: frames with no description, clips with no
    transcript.

    Both jobs are queued at the end of `media.media_preview` and both are lost the same way. One:
    the item became ready while there was no key at all — the job then stored nothing and left the
    item as it was. Two, and this is the one that bites: the model refused. An account over its free
    limit answers «429» to every call, three attempts go in a minute and the job is `failed` for
    good — and nothing ever comes back for that frame, because ordering a video does not describe
    anything, it only reads what is already in the tables (doc-03). A memory then holds a hundred
    failed descriptions and an AI montage that will never turn on, however often it is re-ordered —
    which is a thing that has happened here, on a free tier. This sweep is what comes back.

    Hence the first line — with no model configured there is simply nothing to do, and without it
    the whole library would be queued every hour for every job of it to answer «nobody to ask». The
    two are asked in one sweep so that a worker without a key says so once an hour and not twice.

    ponytail: the whole library at once, in one insert each. A cap per sweep is the knob to add when
    a real key makes a first sweep over a year of memories a bill rather than a queue.
    """
    if ai.from_env() is None:
        return 0
    # A description is made out of the copies `media_preview` left, never the original — a frame
    # whose large copy is not made yet is `sweep_photos_without_previews`'s first.
    described = _catch_up(
        conn, "media_understanding",
        """item.preview_key IS NOT NULL AND NOT EXISTS (
             SELECT 1 FROM media_understandings seen WHERE seen.media_id = item.id::text)""",
    )
    if described:
        log.info("queued a description for %s media item(s) that had none", described)
    # Speech is a clip's alone, and an empty row is an answer (`speech.py`): a silent clip has one
    # and is not asked about again.
    transcribed = _catch_up(
        conn, "media_speech",
        """item.kind = 'video' AND NOT EXISTS (
             SELECT 1 FROM media_speech said WHERE said.media_id = item.id::text)""",
    )
    if transcribed:
        log.info("queued a transcript for %s clip(s) that had none", transcribed)
    return described + transcribed


SWEEPS = (media.sweep_stale_uploads, sweep_photos_without_previews,
          sweep_media_without_frame_info, sweep_media_without_screening,
          sweep_media_without_ai_answers)


# --- the loop ---------------------------------------------------------------


def run_once(conn):
    """Claim one due job and run it. Returns False when nothing is due."""
    # One statement: the row is locked, marked running and returned, so a second
    # worker skips it. ponytail: a worker killed mid-job leaves the row in
    # 'running' forever — add a reaper (running + updated_at old -> queued) when
    # there is more than one worker or a real deploy to be killed by.
    job = conn.execute(
        """
        UPDATE jobs SET status = 'running', updated_at = now()
        WHERE id = (
            SELECT id FROM jobs
            WHERE status = 'queued' AND run_after <= now()
            ORDER BY run_after
            LIMIT 1
            FOR UPDATE SKIP LOCKED
        )
        RETURNING id, type, ref, attempts
        """
    ).fetchone()
    if job is None:
        return False

    job_id, type_, ref, attempts = job
    try:
        handler = HANDLERS.get(type_)
        if handler is None:
            raise LookupError(f"no handler for job type {type_!r}")
        handler(conn, ref)
    except Exception as exc:
        attempts += 1
        delay = BACKOFF_SECONDS * 2 ** (attempts - 1)
        status = "failed" if attempts >= MAX_ATTEMPTS else "queued"
        conn.execute(
            """UPDATE jobs SET status = %s, attempts = %s, last_error = %s,
                               run_after = now() + make_interval(secs => %s),
                               updated_at = now()
               WHERE id = %s""",
            (status, attempts, str(exc), delay, job_id),
        )
        log.warning(
            "job %s (%s) failed on attempt %s/%s: %s — %s",
            job_id, type_, attempts, MAX_ATTEMPTS, exc,
            "given up" if status == "failed" else f"retrying in {delay}s",
        )
        if status == "failed" and type_ == "media_preview":
            # The job is done retrying, and the item it was about has to say so too — the gallery
            # would otherwise wait on `processing` forever.
            media.give_up(conn, ref)
        elif status == "failed" and type_ == "montage":
            # Same for the video that was ordered: the organizer's «повторить» has to appear, and
            # it only appears under a montage that says it failed. Not `str(exc)` though — the api
            # serves `error_text` to every member of the memory, and this one is an ffmpeg stderr
            # tail or a botocore message: temp paths, and the storage endpoint, which in production
            # carries the Cloudflare account id. The full text stays in `last_error` above, which
            # is the operator's to read.
            montages.give_up(conn, ref, "не удалось собрать ролик")
    else:
        conn.execute(
            "UPDATE jobs SET status = 'done', updated_at = now() WHERE id = %s", (job_id,)
        )
        log.info("job %s (%s) done", job_id, type_)
    return True


def wait_for_job(conn):
    """Block until the queue says there is something new — or until the fallback poll is due.

    A notification sent while this worker was busy (or lost with the connection) is a normal case,
    not a stuck queue: the timeout makes the loop look again anyway.
    """
    next(conn.notifies(timeout=IDLE_POLL_SECONDS), None)


def run(conn):
    log.info("waiting for jobs, handlers: %s", ", ".join(HANDLERS))
    # Says which cloud model this worker has, or says out loud that it has none — the job that
    # then quietly cuts a montage without one is otherwise indistinguishable from one that had it.
    log.info("cloud model: %s", ai.from_env() or "none, every call to it will be skipped")
    conn.execute("LISTEN jobs")  # every insert into `jobs` notifies this channel — see schema.sql
    next_sweep = 0.0  # once at start-up, then hourly
    while True:
        if time.monotonic() >= next_sweep:
            next_sweep = time.monotonic() + SWEEP_INTERVAL_SECONDS
            for sweep in SWEEPS:
                try:
                    sweep(conn)
                except Exception:
                    # Housekeeping rides along with the queue rather than in a process of its own,
                    # so a bucket that is unreachable must not take the queue down with it — nor
                    # one sweep take the other with it.
                    log.exception("%s failed, trying again in an hour", sweep.__name__)
        if not run_once(conn):
            wait_for_job(conn)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    command, *args = sys.argv[1:] or ["run"]
    # Outside the connection below on purpose: the library is files and a bucket, and there is no
    # reason for the one command with no database in it to refuse to run without DATABASE_URL.
    if command == "tracks":
        if args:
            sys.exit(__doc__)  # here too, or a typo would ask for a database to say «usage»
        try:
            published = assembly.publish_tracks()
        except KeyError as unset:
            # A storage variable missed on the deploy — `media.storage()` reads the four of them
            # straight out of the environment. One line saying which, the same answer `connect`
            # gives for DATABASE_URL, rather than a `KeyError` out of the middle of boto3.
            sys.exit(f"{unset.args[0]} is not set — see README.md")
        except Exception as exc:
            # Storage that cannot be reached, the wrong bucket, a key that may not write: what it
            # said, in one line rather than in a traceback. Whatever landed before the failure
            # stays there and the command is idempotent, so the answer to any of these is to fix
            # the cause and run it again.
            sys.exit(f"the music library was not published: {exc}")
        print(f"the music library is in the bucket: {len(published)} of {len(assembly.TRACKS)} "
              f"uploaded, the rest were already there")
        sys.exit()
    with connect() as conn:
        if command == "init" and not args:
            init_schema(conn)
            print("the worker's tables are ready")
        elif command == "enqueue" and 1 <= len(args) <= 2:
            print("queued job", enqueue(conn, *args))
        elif command == "run" and not args:
            try:
                run(conn)
            except KeyboardInterrupt:
                log.info("stopped")
        else:
            sys.exit(__doc__)
