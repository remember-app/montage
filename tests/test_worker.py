"""Live checks of the queue cycle against a real Postgres — see README.md.

Warning: these wipe the `jobs` table of whatever DATABASE_URL points at.
"""

import collections
import logging
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time

import pytest

from montage import sweeps, worker

from rows import add_item, read_item


@pytest.fixture
def conn():
    with worker.connect() as connection:
        worker.init_schema(connection)
        connection.execute("DELETE FROM jobs")
        yield connection


def read(conn, job_id):
    return conn.execute(
        "SELECT status, attempts, last_error, run_after > now() FROM jobs WHERE id = %s", (job_id,)
    ).fetchone()


def test_failing_job_retries_then_fails(conn):
    job_id = worker.enqueue(conn, "dummy", "fail")

    for attempt in (1, 2):
        assert worker.run_once(conn) is True
        status, attempts, last_error, deferred = read(conn, job_id)
        assert (status, attempts, deferred) == ("queued", attempt, True)
        assert "asked to fail" in last_error
        assert worker.run_once(conn) is False, "a deferred job must not be claimed yet"
        conn.execute(
            "UPDATE jobs SET run_after = now() WHERE id = %s", (job_id,)
        )  # skip the backoff

    assert worker.run_once(conn) is True
    status, attempts, _, _ = read(conn, job_id)
    assert (status, attempts) == ("failed", worker.MAX_ATTEMPTS)
    assert worker.run_once(conn) is False, "a failed job is never claimed again"


def test_successful_job_is_done(conn):
    job_id = worker.enqueue(conn, "dummy")
    assert worker.run_once(conn) is True
    assert read(conn, job_id)[:2] == ("done", 0)


def test_a_queued_job_wakes_an_idle_worker(conn):
    """A job put in the queue must not wait for the fallback poll — the table notifies the loop."""
    with worker.connect() as idle_worker:
        idle_worker.execute("LISTEN jobs")
        worker.enqueue(conn, "dummy")

        started = time.monotonic()
        worker.wait_for_job(idle_worker)
        assert time.monotonic() - started < 1


def test_two_workers_on_one_queue_never_take_the_same_job(conn, monkeypatch):
    """A rolling deploy or a second replica puts two copies of the worker on the queue: every job
    still runs exactly once, or a montage would be cut twice and a preview pushed twice."""
    ran = collections.Counter()
    monkeypatch.setitem(
        worker.HANDLERS, "counted", lambda conn, ref: (ran.update([ref]), time.sleep(0.01))
    )
    refs = [str(number) for number in range(60)]
    for ref in refs:
        worker.enqueue(conn, "counted", ref)
    start = threading.Barrier(2)

    def drain():
        with worker.connect() as mine:
            start.wait()
            while worker.run_once(mine):
                pass

    copies = [threading.Thread(target=drain) for _ in range(2)]
    for copy in copies:
        copy.start()
    for copy in copies:
        copy.join(timeout=60)

    assert sorted(ran.items()) == sorted((ref, 1) for ref in refs)
    assert conn.execute("SELECT status, attempts, count(*) FROM jobs GROUP BY 1, 2").fetchall() == [
        ("done", 0, 60)
    ]


def test_init_brings_an_older_queue_up_to_date_and_keeps_its_jobs(conn):
    """A database the worker ran on before dead workers' jobs were taken back: `init` adds the
    index that search goes by and the trigger that wakes an idle worker, no job changes, and a
    second `init` is a no-op."""
    conn.execute("DROP INDEX jobs_running_idx")
    conn.execute("DROP TRIGGER jobs_notify ON jobs")
    for status, attempts in (("queued", 0), ("running", 1), ("done", 0), ("failed", 3)):
        conn.execute(
            """INSERT INTO jobs (type, ref, status, attempts, last_error)
               VALUES ('dummy', %s, %s, %s, 'e')""",
            (status, status, attempts),
        )
    before = conn.execute("SELECT * FROM jobs ORDER BY id").fetchall()

    worker.init_schema(conn)
    worker.init_schema(conn)

    assert conn.execute("SELECT * FROM jobs ORDER BY id").fetchall() == before
    assert conn.execute(
        "SELECT count(*) FROM pg_indexes WHERE indexname = 'jobs_running_idx'"
    ).fetchone() == (1,)
    with worker.connect() as idle_worker:
        idle_worker.execute("LISTEN jobs")
        worker.enqueue(conn, "dummy")
        assert next(idle_worker.notifies(timeout=2), None) is not None, "the trigger is back"


# --- the one command with no database in it ---------------------------------
# `python -m montage tracks`, run by hand on a deploy. No fixture here: the point is what the
# command says when the storage it needs is not there, and neither Postgres nor the bucket is
# involved.


def tracks_command(**environment):
    """`python -m montage tracks` with the storage variables replaced by these, if any."""
    return subprocess.run(
        [sys.executable, "-m", "montage", "tracks"],
        cwd=pathlib.Path(worker.__file__).parents[1],
        capture_output=True,
        text=True,
        env={name: value for name, value in os.environ.items() if not name.startswith("R2_")}
        | environment,
    )


def test_publishing_the_library_without_storage_names_the_variable_that_is_missing():
    """The command is run on a deploy, by a person at a terminal: a variable they forgot has to
    read as one line naming it, the way a missing DATABASE_URL does, and not as a `KeyError` out
    of the middle of boto3."""
    done = tracks_command()

    assert done.returncode == 1
    assert done.stderr.strip() == "R2_ENDPOINT is not set — see README.md"


def test_publishing_the_library_into_storage_that_does_not_answer_says_so_in_one_line():
    """Nor may an unreachable bucket come out as a traceback — the first thing `publish_tracks`
    does to every key is a `head_object`, and that one does not fail as a `ClientError`."""
    done = tracks_command(
        R2_ENDPOINT="http://127.0.0.1:9",  # discard: nothing is listening
        R2_BUCKET="remember",
        R2_ACCESS_KEY_ID="whoever",
        R2_SECRET_ACCESS_KEY="whatever",
    )

    assert done.returncode == 1
    assert "Traceback" not in done.stderr
    assert done.stderr.startswith("the music library was not published:")


def test_a_thread_count_that_is_no_count_stops_the_process_before_it_does_anything():
    """Not at the first montage, minutes into a job: the setting is read when the worker starts."""
    done = tracks_command(FFMPEG_THREADS="two")

    assert done.returncode != 0
    assert "FFMPEG_THREADS must be a whole number" in done.stderr


# --- a worker that died holding a job ---------------------------------------


def abandon(conn, job_id):
    """The row as a worker killed mid-job leaves it: `running`, and silent for longer than a live
    worker ever is."""
    conn.execute(
        """UPDATE jobs SET status = 'running', updated_at = now() - make_interval(secs => %s)
           WHERE id = %s""",
        (worker.STALE_SECONDS + 60, job_id),
    )


def test_a_job_its_worker_died_holding_goes_back_to_the_queue_and_runs_again(conn, caplog):
    job_id = worker.enqueue(conn, "dummy")
    abandon(conn, job_id)

    assert worker.reap(conn) == 1
    status, attempts, last_error, deferred = read(conn, job_id)
    assert (status, attempts, deferred) == ("queued", 1, False), "an attempt, and no backoff"
    assert "the worker running it died" in last_error
    assert any(
        record.levelname == "WARNING" and f"job {job_id} (dummy)" in record.getMessage()
        for record in caplog.records
    )

    assert worker.run_once(conn) is True
    assert read(conn, job_id)[:2] == ("done", 1)


def test_a_job_whose_worker_is_alive_is_not_reaped(conn, monkeypatch):
    """A montage runs for many minutes: however long a job takes, a worker still holding it is not
    taken for dead."""
    monkeypatch.setattr(worker, "HEARTBEAT_SECONDS", 0.1)
    monkeypatch.setattr(worker, "STALE_SECONDS", 0.5)
    reaped = []

    def long_job(conn, ref):
        time.sleep(1.5)  # three times as long as a dead worker's job stays unnoticed
        with worker.connect() as another_worker:
            reaped.append(worker.reap(another_worker))

    monkeypatch.setitem(worker.HANDLERS, "long", long_job)
    job_id = worker.enqueue(conn, "long")

    assert worker.run_once(conn) is True
    assert reaped == [0]
    assert read(conn, job_id)[:2] == ("done", 0)


def test_a_job_that_takes_its_worker_down_every_time_ends_failed(conn):
    job_id = worker.enqueue(conn, "dummy")

    for attempt in range(1, worker.MAX_ATTEMPTS + 1):
        abandon(conn, job_id)
        assert worker.reap(conn) == 1
        status, attempts, last_error, _ = read(conn, job_id)
        assert attempts == attempt
    assert status == "failed"
    assert "the worker running it died" in last_error
    assert worker.run_once(conn) is False, "a failed job is never claimed again"


def test_a_preview_that_takes_its_worker_down_every_time_fails_its_item_too(conn, memory):
    """Or the gallery waits on it forever — the same as after a preview that failed on its own."""
    item_id, _ = add_item(conn, memory, "photo")
    job_id = worker.enqueue(conn, "media_preview", item_id)
    conn.execute("UPDATE jobs SET attempts = %s WHERE id = %s", (worker.MAX_ATTEMPTS - 1, job_id))
    abandon(conn, job_id)

    assert worker.reap(conn) == 1
    assert read(conn, job_id)[0] == "failed"
    assert read_item(conn, item_id)[0] == "failed"


# --- SIGTERM, which is how Kubernetes stops the pod -------------------------
# The loop itself, in this very process: `run` takes the signal and gives its handlers back when it
# ends. Sweeps are left out — they are not what is being stopped here.


@pytest.fixture
def sigterm_in():
    """SIGTERM to this process a moment from now, the way kubelet sends it to a pod it stops — and
    never after the test, whose `run` would no longer be there to take it."""
    timers = []

    def send(seconds):
        timers.append(threading.Timer(seconds, os.kill, (os.getpid(), signal.SIGTERM)))
        timers[-1].start()

    yield send
    for timer in timers:
        timer.cancel()


@pytest.fixture
def loop(conn, monkeypatch):
    monkeypatch.setattr(sweeps, "SWEEPS", ())
    return conn


def test_sigterm_hands_back_a_job_that_would_not_finish_in_time_and_stops(
    loop, sigterm_in, monkeypatch, caplog
):
    caplog.set_level(logging.INFO, logger="montage")
    monkeypatch.setattr(worker, "GRACE_SECONDS", 0.5)
    monkeypatch.setitem(worker.HANDLERS, "long", lambda conn, ref: time.sleep(60))
    job_id = worker.enqueue(loop, "long")
    next_job = worker.enqueue(loop, "dummy")

    sigterm_in(0.5)
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        worker.run(loop)

    assert time.monotonic() - started < 5, "stopped in its grace, not at SIGKILL"
    assert read(loop, job_id)[:2] == ("queued", 0), "handed back, and no attempt counted"
    assert read(loop, next_job)[0] == "queued", "and no new job taken"
    assert f"job {job_id} (long) handed back to the queue" in caplog.text

    monkeypatch.setitem(worker.HANDLERS, "long", lambda conn, ref: None)
    assert worker.run_once(loop) is True
    assert read(loop, job_id)[0] == "done", "the next worker takes it first"
    assert read(loop, next_job)[0] == "queued"


def test_sigterm_lets_a_job_finish_in_its_grace_and_takes_no_new_one(loop, sigterm_in, monkeypatch):
    monkeypatch.setattr(worker, "GRACE_SECONDS", 5)
    monkeypatch.setitem(worker.HANDLERS, "short", lambda conn, ref: time.sleep(1))
    job_id = worker.enqueue(loop, "short")
    next_job = worker.enqueue(loop, "dummy")

    sigterm_in(0.3)
    worker.run(loop)  # returns, rather than raises

    assert read(loop, job_id)[:2] == ("done", 0)
    assert read(loop, next_job)[0] == "queued"


def test_sigterm_during_the_sweeps_takes_no_job_after_them(loop, monkeypatch):
    monkeypatch.setattr(sweeps, "SWEEPS", (lambda conn: os.kill(os.getpid(), signal.SIGTERM),))
    job_id = worker.enqueue(loop, "dummy")

    worker.run(loop)

    assert read(loop, job_id)[0] == "queued"


def test_sigterm_stops_a_waiting_worker_at_once(loop, sigterm_in):
    sigterm_in(0.5)
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        worker.run(loop)
    assert time.monotonic() - started < 3, "neither the idle poll nor the grace is waited out"


def test_the_worker_process_exits_on_sigterm(conn):
    """The process itself, the way the pod runs it: SIGTERM ends it cleanly and at once."""
    process = subprocess.Popen(
        [sys.executable, "-m", "montage", "run"],
        cwd=pathlib.Path(worker.__file__).parents[1],
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        for line in process.stderr:
            if "cloud model:" in line:
                break
        time.sleep(1)  # past the start-up sweeps, into the wait
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 0
        assert "stopped" in process.stderr.read()
    finally:
        process.kill()
        process.wait()


def test_the_wait_gives_up_without_a_notification(conn, monkeypatch):
    """A missed notification must not hang the queue: the wait ends and the loop looks again."""
    monkeypatch.setattr(worker, "IDLE_POLL_SECONDS", 0.5)
    conn.execute("LISTEN jobs")

    started = time.monotonic()
    worker.wait_for_job(conn)  # nothing is queued, so nothing notifies
    assert 0.5 <= time.monotonic() - started < 5
