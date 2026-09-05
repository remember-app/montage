"""Montage worker: polls the shared `jobs` table in Postgres and runs one handler per job.

    python worker.py init                  # create the jobs table (idempotent)
    python worker.py enqueue <type> [ref]  # put one job in the queue
    python worker.py run                   # poll loop, Ctrl-C to stop

Needs DATABASE_URL. See README.md.
"""

import logging
import os
import pathlib
import sys
import time

import psycopg

import media
import montage as montages  # the module and its handler share the job's name; the import cannot

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 30  # attempt n is retried after BACKOFF_SECONDS * 2 ** (n - 1)
IDLE_SLEEP_SECONDS = 5
SWEEP_INTERVAL_SECONDS = 3600  # «раз в час» (architecture.md, «Обработка ошибок»)
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
    "montage": montages.montage,
}


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


def run(conn):
    log.info("polling for jobs, handlers: %s", ", ".join(HANDLERS))
    next_sweep = 0.0  # once at start-up, then hourly
    while True:
        if time.monotonic() >= next_sweep:
            next_sweep = time.monotonic() + SWEEP_INTERVAL_SECONDS
            try:
                media.sweep_stale_uploads(conn)
            except Exception:
                # Housekeeping rides along with the queue rather than in a process of its own, so a
                # bucket that is unreachable must not take the queue down with it.
                log.exception("the sweep of abandoned uploads failed, trying again in an hour")
        if not run_once(conn):
            time.sleep(IDLE_SLEEP_SECONDS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    command, *args = sys.argv[1:] or ["run"]
    with connect() as conn:
        if command == "init" and not args:
            init_schema(conn)
            print("jobs table ready")
        elif command == "enqueue" and 1 <= len(args) <= 2:
            print("queued job", enqueue(conn, *args))
        elif command == "run" and not args:
            try:
                run(conn)
            except KeyboardInterrupt:
                log.info("stopped")
        else:
            sys.exit(__doc__)
