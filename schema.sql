-- The shared job queue. The montage worker owns this table; the API only inserts rows.
-- Idempotent on purpose: `python worker.py init` applies it on every start-up of a new database.

CREATE TABLE IF NOT EXISTS jobs (
    id          bigserial   PRIMARY KEY,
    type        text        NOT NULL,
    ref         text,                                   -- id of the object the job is about
    status      text        NOT NULL DEFAULT 'queued'
                            CHECK (status IN ('queued', 'running', 'done', 'failed')),
    attempts    integer     NOT NULL DEFAULT 0,
    run_after   timestamptz NOT NULL DEFAULT now(),     -- not due before this moment
    last_error  text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- The only query the worker runs against this table.
CREATE INDEX IF NOT EXISTS jobs_due_idx ON jobs (run_after) WHERE status = 'queued';

-- Postgres wakes the idle worker itself instead of it guessing when to look again. On the table,
-- so it covers everyone who inserts — the API and the worker's own `enqueue` alike. Per statement,
-- not per row: the payload is empty anyway, and one nudge is enough for a loop that then drains the
-- queue. The notification is delivered on commit, when the row is actually visible.
CREATE OR REPLACE FUNCTION jobs_notify() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    PERFORM pg_notify('jobs', '');
    RETURN NULL;
END;
$$;

DROP TRIGGER IF EXISTS jobs_notify ON jobs;
CREATE TRIGGER jobs_notify AFTER INSERT ON jobs
    FOR EACH STATEMENT EXECUTE FUNCTION jobs_notify();
