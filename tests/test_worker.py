"""Live checks of the queue cycle against a real Postgres — see README.md.

Warning: these wipe the `jobs` table of whatever DATABASE_URL points at.
"""

import os
import pathlib
import subprocess
import sys
import time

import pytest

from montage import worker


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


def test_the_wait_gives_up_without_a_notification(conn, monkeypatch):
    """A missed notification must not hang the queue: the wait ends and the loop looks again."""
    monkeypatch.setattr(worker, "IDLE_POLL_SECONDS", 0.5)
    conn.execute("LISTEN jobs")

    started = time.monotonic()
    worker.wait_for_job(conn)  # nothing is queued, so nothing notifies
    assert 0.5 <= time.monotonic() - started < 5
