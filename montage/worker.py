"""Очередь: одна задача за раз из общей таблицы `jobs` в Postgres, по обработчику на тип.

The loop claims a due job, runs the handler `HANDLERS` names for its type, and marks the row done
or retries it — three attempts, then `failed`. Between jobs it sleeps on a notification from the
table itself, and once an hour it lets `sweeps` catch up on what nobody queued.

Commands to run all this by hand are `__main__.py`: `python -m montage run`.
"""

import logging
import os
import pathlib
import sys
import time

import psycopg

from montage import ai, sweeps
from montage.media import frame_info, preview, screening, speech, understanding

# Модуль и задача, которую он выполняет, зовутся одинаково — отсюда второе имя.
from montage.video import montage as montages

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 30  # attempt n is retried after BACKOFF_SECONDS * 2 ** (n - 1)
IDLE_POLL_SECONDS = 30  # fallback only: a queued job normally wakes the worker by NOTIFY
SWEEP_INTERVAL_SECONDS = 3600  # «раз в час» (architecture.md, «Обработка ошибок»)
# Рядом с pyproject.toml, а не в пакете: этот же файл по этому же пути читает миграция api
# (`projects/api/scripts/migrate.ts`) — таблица задач общая, и заводит её тот, кто раньше.
SCHEMA = pathlib.Path(__file__).parents[1] / "schema.sql"

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
    "media_preview": preview.media_preview,
    "media_frame_info": frame_info.media_frame_info,
    "media_screening": screening.media_screening,
    "media_speech": speech.transcribe,
    "media_understanding": understanding.media_understanding,
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
            job_id,
            type_,
            attempts,
            MAX_ATTEMPTS,
            exc,
            "given up" if status == "failed" else f"retrying in {delay}s",
        )
        if status == "failed" and type_ == "media_preview":
            # The job is done retrying, and the item it was about has to say so too — the gallery
            # would otherwise wait on `processing` forever.
            preview.give_up(conn, ref)
        elif status == "failed" and type_ == "montage":
            # Same for the video that was ordered: the organizer's «повторить» has to appear, and
            # it only appears under a montage that says it failed. Not `str(exc)` though — the api
            # serves `error_text` to every member of the memory, and this one is an ffmpeg stderr
            # tail or a botocore message: temp paths, and the storage endpoint, which in production
            # carries the Cloudflare account id. The full text stays in `last_error` above, which
            # is the operator's to read.
            montages.give_up(conn, ref, "не удалось собрать ролик")
    else:
        conn.execute("UPDATE jobs SET status = 'done', updated_at = now() WHERE id = %s", (job_id,))
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
            for sweep in sweeps.SWEEPS:
                try:
                    sweep(conn)
                except Exception:
                    # Housekeeping rides along with the queue rather than in a process of its own,
                    # so a bucket that is unreachable must not take the queue down with it — nor
                    # one sweep take the other with it.
                    log.exception("%s failed, trying again in an hour", sweep.__name__)
        if not run_once(conn):
            wait_for_job(conn)
