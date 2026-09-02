"""Live checks of the queue cycle against a real Postgres — see README.md.

Warning: these wipe the `jobs` table of whatever DATABASE_URL points at.
"""

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
