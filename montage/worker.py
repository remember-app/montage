"""Очередь: одна задача за раз из общей таблицы `jobs` в Postgres, по обработчику на тип.

The loop claims a due job, runs the handler `HANDLERS` names for its type, and marks the row done
or retries it — three attempts, then `failed`. Between jobs it sleeps on a notification from the
table itself, and once an hour it lets `sweeps` catch up on what nobody queued.

A job whose worker died holding it goes back to the queue (`reap`), and a worker told to stop hands
back the job it cannot finish in time (`_Shutdown`) — see docs/queue.md.

Commands to run all this by hand are `__main__.py`: `python -m montage run`.
"""

import contextlib
import logging
import os
import pathlib
import signal
import sys
import threading
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
# The job in hand says it is alive this often, and one silent for STALE_SECONDS belongs to a worker
# that is gone. Ten beats and not two: a beat can be late behind a database hiccup or a stretch of C
# code holding the GIL, and taking a live montage off its worker would have two of them cut it.
HEARTBEAT_SECONDS = 30
STALE_SECONDS = 300
# Of the 30 s Kubernetes allows between SIGTERM and SIGKILL (`terminationGracePeriodSeconds`): the
# job in hand gets this much to finish, the rest is for handing it back and closing up.
GRACE_SECONDS = 20
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
    """Claim one due job and run it. Returns False when nothing is due.

    Interrupted — Ctrl-C, or the grace after SIGTERM run out (`_Shutdown`) — it hands the job back
    to the queue on the way out, as it was: the worker stopped, the job did not fail, and no attempt
    is counted against it.
    """
    # One statement: the row is locked, marked running and returned, so a second worker skips it.
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
    """One more attempt gone: the job is retried after a backoff, or given up on after the last."""
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
    """The job is out of attempts, and the object it was about has to say so too."""
    if type_ == "media_preview":
        # The gallery would otherwise wait on `processing` forever.
        preview.give_up(conn, ref)
    elif type_ == "montage":
        # Same for the video that was ordered: the organizer's «повторить» has to appear, and it
        # only appears under a montage that says it failed. Not the error itself though — the api
        # serves `error_text` to every member of the memory, and this one is an ffmpeg stderr tail
        # or a botocore message: temp paths, and the storage endpoint, which in production carries
        # the Cloudflare account id. The full text stays in `jobs.last_error`, which is the
        # operator's to read.
        montages.give_up(conn, ref, "не удалось собрать ролик")


def _hand_back(conn, job_id, type_):
    """The job in hand back in the queue as it was — first in line, since it was due first."""
    handed = conn.execute(
        """UPDATE jobs SET status = 'queued', updated_at = now()
           WHERE id = %s AND status = 'running'""",
        (job_id,),
    ).rowcount
    if handed:
        log.info("job %s (%s) handed back to the queue: the worker is stopping", job_id, type_)


@contextlib.contextmanager
def _heartbeat(job_id):
    """The job's `updated_at` kept fresh for as long as the handler holds it — the sign of life
    `reap` goes by. A montage runs for many minutes, and nothing else about it says it is alive."""
    stop = threading.Event()
    threading.Thread(
        target=_beat, args=(job_id, stop), name=f"heartbeat-{job_id}", daemon=True
    ).start()
    try:
        yield
    finally:
        stop.set()


def _beat(job_id, stop):
    # A connection of its own, opened at the first beat: the handler's may be inside a transaction
    # (`layout.write`), which would hold a beat back until it commits, and most jobs are over before
    # the first beat is due.
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
    """Jobs left `running` by a worker that died holding them, back in the queue — or failed, when
    that was their last attempt. Returns how many.

    A live worker keeps the job it holds fresh (`_heartbeat`), so one silent for STALE_SECONDS
    belongs to a worker that is gone: killed on a deploy past its grace, taken down with its node,
    or by running out of memory. That counts as an attempt — a file that takes the worker down with
    it would otherwise take it down forever — but costs no backoff: it was due long ago, and is
    first in line again.
    """
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
    """Block until the queue says there is something new — or until the fallback poll is due.

    A notification sent while this worker was busy (or lost with the connection) is a normal case,
    not a stuck queue: the timeout makes the loop look again anyway.
    """
    next(conn.notifies(timeout=IDLE_POLL_SECONDS), None)


class _Shutdown:
    """SIGTERM — how Kubernetes stops a pod, SIGKILL following 30 seconds later.

    Waiting for work, the worker just goes. Holding a job, it takes no new one and gives the one in
    hand GRACE_SECONDS: a preview is done by then, a montage that runs for minutes is not, and is
    interrupted the way Ctrl-C interrupts it — `run_once` hands it back to the queue on the way out.
    """

    def __init__(self):
        self.asked = False
        # For a job, with none in hand: nothing to finish, nothing to hand back.
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
    """The loop, until SIGTERM ends it (`_Shutdown`) or Ctrl-C raises out of it."""
    log.info("waiting for jobs, handlers: %s", ", ".join(HANDLERS))
    # Says which cloud model this worker has, or says out loud that it has none — the job that
    # then quietly cuts a montage without one is otherwise indistinguishable from one that had it.
    log.info("cloud model: %s", ai.from_env() or "none, every call to it will be skipped")
    conn.execute("LISTEN jobs")  # every insert into `jobs` notifies this channel — see schema.sql
    next_sweep = 0.0  # once at start-up, then hourly
    with _Shutdown() as shutdown:
        while not shutdown.asked:
            if time.monotonic() >= next_sweep:
                next_sweep = time.monotonic() + SWEEP_INTERVAL_SECONDS
                for sweep in sweeps.SWEEPS:
                    try:
                        sweep(conn)
                    except Exception:
                        # Housekeeping rides along with the queue rather than in a process of its
                        # own, so a bucket that is unreachable must not take the queue down with it
                        # — nor one sweep take the other with it.
                        log.exception("%s failed, trying again in an hour", sweep.__name__)
            reap(conn)
            if not run_once(conn):
                shutdown.waiting = True
                wait_for_job(conn)
                shutdown.waiting = False
