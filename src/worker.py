import contextlib
import logging
import os
import signal
import threading

import psycopg

from src.integrations.db import connect
from src.jobs import derive, montage

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 30
IDLE_POLL_SECONDS = 30
HEARTBEAT_SECONDS = 30
STALE_SECONDS = 300
GRACE_SECONDS = 20

HANDLERS = {"media_derive": derive.run, "montage": montage.run}

log = logging.getLogger("worker")


def run_once(conn: psycopg.Connection) -> bool:
    job = conn.execute(
        """
        update jobs set status = 'running', updated_at = now()
        where id = (
            select id from jobs
            where status = 'queued' and run_after <= now()
            order by run_after
            limit 1
            for update skip locked
        )
        returning id, kind, ref, attempts
        """
    ).fetchone()
    if job is None:
        return False

    job_id, kind, ref, attempts = job
    try:
        handler = HANDLERS.get(kind)
        if handler is None:
            raise LookupError(f"no handler for job kind {kind!r}")
        with heartbeat(job_id):
            handler(conn, ref)
    except KeyboardInterrupt:
        hand_back(conn, job_id, kind)
        raise
    except Exception as exc:
        failed(conn, job_id, kind, ref, attempts + 1, exc)
    else:
        conn.execute("update jobs set status = 'done', updated_at = now() where id = %s", (job_id,))
        log.info("job %s (%s) done", job_id, kind)
    return True


def failed(conn: psycopg.Connection, job_id: int, kind: str, ref, attempts: int, exc) -> None:
    delay = BACKOFF_SECONDS * 2 ** (attempts - 1)
    status = "failed" if attempts >= MAX_ATTEMPTS else "queued"
    conn.execute(
        """
        update jobs
        set status = %s,
            attempts = %s,
            error = %s,
            run_after = now() + make_interval(secs => %s),
            updated_at = now()
        where id = %s
        """,
        (status, attempts, str(exc)[:2000], delay, job_id),
    )
    log.warning(
        "job %s (%s) attempt %s/%s failed: %s%s",
        job_id,
        kind,
        attempts,
        MAX_ATTEMPTS,
        exc,
        "" if status == "queued" else " — giving up",
    )
    if status == "failed":
        give_up(conn, kind, ref)


def give_up(conn: psycopg.Connection, kind: str, ref) -> None:
    if kind == "media_derive":
        derive.give_up(conn, ref)
    elif kind == "montage":
        montage.give_up(conn, ref)


def hand_back(conn: psycopg.Connection, job_id: int, kind: str) -> None:
    handed = conn.execute(
        """
        update jobs set status = 'queued', updated_at = now()
        where id = %s and status = 'running'
        """,
        (job_id,),
    ).rowcount
    if handed:
        log.info("job %s (%s) handed back to the queue", job_id, kind)


@contextlib.contextmanager
def heartbeat(job_id: int):
    stop = threading.Event()
    thread = threading.Thread(target=beat, args=(job_id, stop), daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()


def beat(job_id: int, stop: threading.Event) -> None:
    beat_conn = None
    try:
        while not stop.wait(HEARTBEAT_SECONDS):
            try:
                if beat_conn is None or beat_conn.closed:
                    beat_conn = connect()
                beat_conn.execute(
                    "update jobs set updated_at = now() where id = %s and status = 'running'",
                    (job_id,),
                )
            except Exception as exc:
                log.warning("job %s: heartbeat failed: %s", job_id, exc)
    finally:
        if beat_conn is not None:
            beat_conn.close()


def reap(conn: psycopg.Connection) -> int:
    reaped = conn.execute(
        """
        update jobs
        set status = case when attempts + 1 >= %(max)s then 'failed' else 'queued' end,
            attempts = attempts + 1,
            error = 'the worker running it died',
            updated_at = now()
        where status = 'running' and updated_at < now() - make_interval(secs => %(stale)s)
        returning id, kind, ref, attempts, status
        """,
        {"max": MAX_ATTEMPTS, "stale": STALE_SECONDS},
    ).fetchall()
    for job_id, kind, ref, attempts, status in reaped:
        log.warning(
            "job %s (%s) was left running, attempt %s/%s%s",
            job_id,
            kind,
            attempts,
            MAX_ATTEMPTS,
            "" if status == "queued" else " — giving up",
        )
        if status == "failed":
            give_up(conn, kind, ref)
    return len(reaped)


def wait_for_job(conn: psycopg.Connection) -> None:
    next(conn.notifies(timeout=IDLE_POLL_SECONDS), None)


class Shutdown:
    def __init__(self) -> None:
        self.asked = False
        self.waiting = False

    def __enter__(self):
        self.before = (
            signal.signal(signal.SIGTERM, self.terminate),
            signal.signal(signal.SIGALRM, interrupt),
        )
        return self

    def __exit__(self, *_):
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGTERM, self.before[0])
        signal.signal(signal.SIGALRM, self.before[1])

    def terminate(self, signum, frame):
        if self.waiting:
            raise KeyboardInterrupt
        if not self.asked:
            self.asked = True
            log.info("SIGTERM: taking no new job, the one in hand gets %ss", GRACE_SECONDS)
            signal.setitimer(signal.ITIMER_REAL, GRACE_SECONDS)


def interrupt(signum, frame):
    raise KeyboardInterrupt


def run() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    conn = connect()
    conn.execute("listen jobs")
    log.info("waiting for jobs: %s", ", ".join(HANDLERS))
    with Shutdown() as shutdown:
        while not shutdown.asked:
            reap(conn)
            if not shutdown.asked and not run_once(conn):
                shutdown.waiting = True
                wait_for_job(conn)
                shutdown.waiting = False
