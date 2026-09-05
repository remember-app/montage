"""Live checks of the queue cycle against a real Postgres — see README.md.

Warning: these wipe the `jobs` table of whatever DATABASE_URL points at.
"""

import time

import pytest

import worker


@pytest.fixture
def conn():
    with worker.connect() as connection:
        worker.init_schema(connection)
        connection.execute("DELETE FROM jobs")
        yield connection


def read(conn, job_id):
    return conn.execute(
        "SELECT status, attempts, last_error, run_after > now() FROM jobs WHERE id = %s",
        (job_id,),
    ).fetchone()


def test_failing_job_retries_then_fails(conn):
    job_id = worker.enqueue(conn, "dummy", "fail")

    for attempt in (1, 2):
        assert worker.run_once(conn) is True
        status, attempts, last_error, deferred = read(conn, job_id)
        assert (status, attempts, deferred) == ("queued", attempt, True)
        assert "asked to fail" in last_error
        assert worker.run_once(conn) is False, "a deferred job must not be claimed yet"
        conn.execute("UPDATE jobs SET run_after = now() WHERE id = %s", (job_id,))  # skip the backoff

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


def test_the_wait_gives_up_without_a_notification(conn, monkeypatch):
    """A missed notification must not hang the queue: the wait ends and the loop looks again."""
    monkeypatch.setattr(worker, "IDLE_POLL_SECONDS", 0.5)
    conn.execute("LISTEN jobs")

    started = time.monotonic()
    worker.wait_for_job(conn)  # nothing is queued, so nothing notifies
    assert 0.5 <= time.monotonic() - started < 5
