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
