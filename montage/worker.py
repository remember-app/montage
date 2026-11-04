import contextlib
import logging
import os
import pathlib
import signal
import sys
import threading
import time

import psycopg

from montage import ai, storage, sweeps
from montage.media import frame_info, preview, screening, speech, understanding
from montage.video import montage as montages

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 30
IDLE_POLL_SECONDS = 30
SWEEP_INTERVAL_SECONDS = 3600
HEARTBEAT_SECONDS = 30
STALE_SECONDS = 300
GRACE_SECONDS = 20
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


def dummy(conn, ref):
    if ref == "fail":
        raise RuntimeError("dummy job was asked to fail")
    log.info("dummy job did nothing, successfully")


HANDLERS = {
    "dummy": dummy,
    "media_preview": preview.media_preview,
    "media_frame_info": frame_info.media_frame_info,
    "media_analysis": screening.media_analysis,
    "media_screening": screening.media_screening,
    "media_speech": speech.transcribe,
    "media_understanding": understanding.media_understanding,
    "montage": montages.montage,
}


def run_once(conn):
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
        try:
            handler = HANDLERS.get(type_)
            if handler is None:
                raise LookupError(f"no handler for job type {type_!r}")
            with _heartbeat(job_id):
                handler(conn, ref)
        except Exception as exc:
            _failed(conn, job_id, type_, ref, attempts + 1, exc)
        else:
            conn.execute(
                "UPDATE jobs SET status = 'done', updated_at = now() WHERE id = %s", (job_id,)
            )
            log.info("job %s (%s) done", job_id, type_)
    except KeyboardInterrupt:
        _hand_back(conn, job_id, type_)
        raise
    return True


def _failed(conn, job_id, type_, ref, attempts, exc):
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
    if status == "failed":
        _give_up(conn, type_, ref)


def _give_up(conn, type_, ref):
    if type_ == "media_preview":
        preview.give_up(conn, ref)
    elif type_ == "montage":
        montages.give_up(conn, ref, "не удалось собрать ролик", "internal")


def _hand_back(conn, job_id, type_):
    handed = conn.execute(
        """UPDATE jobs SET status = 'queued', updated_at = now()
           WHERE id = %s AND status = 'running'""",
        (job_id,),
    ).rowcount
    if handed:
        log.info("job %s (%s) handed back to the queue: the worker is stopping", job_id, type_)


@contextlib.contextmanager
def _heartbeat(job_id):
    stop = threading.Event()
    threading.Thread(
        target=_beat, args=(job_id, stop), name=f"heartbeat-{job_id}", daemon=True
    ).start()
    try:
        yield
    finally:
        stop.set()


def _beat(job_id, stop):
    beat = None
    try:
        while not stop.wait(HEARTBEAT_SECONDS):
            try:
                beat = beat if beat and not beat.closed else connect()
                beat.execute(
                    "UPDATE jobs SET updated_at = now() WHERE id = %s AND status = 'running'",
                    (job_id,),
                )
            except Exception as exc:
                log.warning(
                    "job %s: could not say it is alive, trying again in %ss: %s",
                    job_id,
                    HEARTBEAT_SECONDS,
                    exc,
                )
    finally:
        if beat:
            beat.close()


def reap(conn):
    reaped = conn.execute(
        """UPDATE jobs
           SET status = CASE WHEN attempts + 1 >= %(max)s THEN 'failed' ELSE 'queued' END,
               attempts = attempts + 1,
               last_error = 'the worker running it died: silent since ' || updated_at,
               updated_at = now()
           WHERE status = 'running' AND updated_at < now() - make_interval(secs => %(stale)s)
           RETURNING id, type, ref, attempts, status""",
        {"max": MAX_ATTEMPTS, "stale": STALE_SECONDS},
    ).fetchall()
    for job_id, type_, ref, attempts, status in reaped:
        log.warning(
            "job %s (%s) was left running by a worker that died, attempt %s/%s — %s",
            job_id,
            type_,
            attempts,
            MAX_ATTEMPTS,
            "given up" if status == "failed" else "back in the queue",
        )
        if status == "failed":
            _give_up(conn, type_, ref)
    return len(reaped)


def wait_for_job(conn):
    next(conn.notifies(timeout=IDLE_POLL_SECONDS), None)


class _Shutdown:
    def __init__(self):
        self.asked = False
        self.waiting = False

    def __enter__(self):
        self._before = (
            signal.signal(signal.SIGTERM, self._terminate),
            signal.signal(signal.SIGALRM, _interrupt),
        )
        return self

    def __exit__(self, *_):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGTERM, self._before[0])
        signal.signal(signal.SIGALRM, self._before[1])

    def _terminate(self, signum, frame):
        if self.waiting:
            raise KeyboardInterrupt
        if not self.asked:
            self.asked = True
            log.info(
                "SIGTERM: taking no new job, the one in hand gets %ss to finish", GRACE_SECONDS
            )
            signal.setitimer(signal.ITIMER_REAL, GRACE_SECONDS)


def _interrupt(signum, frame):
    raise KeyboardInterrupt


def run(conn):
    log.info("waiting for jobs, handlers: %s", ", ".join(HANDLERS))
    log.info("cloud model: %s", ai.from_env() or "none, every call to it will be skipped")
    log.info("storage: %s", storage.client())
    conn.execute("LISTEN jobs")
    next_sweep = 0.0
    with _Shutdown() as shutdown:
        while not shutdown.asked:
            if time.monotonic() >= next_sweep:
                next_sweep = time.monotonic() + SWEEP_INTERVAL_SECONDS
                for sweep in sweeps.SWEEPS:
                    try:
                        sweep(conn)
                    except Exception:
                        log.exception("%s failed, trying again in an hour", sweep.__name__)
            reap(conn)
            if not shutdown.asked and not run_once(conn):
                shutdown.waiting = True
                wait_for_job(conn)
                shutdown.waiting = False
