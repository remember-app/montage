-- The montage worker's own tables: the shared job queue, and the journal of what was asked of the
-- cloud model. The API only inserts rows into `jobs` and never touches the journal.
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

-- Every question the worker put to a cloud model, and what came back (doc-03: «все запросы,
-- которые делаем в нейросети, обязательно сохраняются — чтобы их можно было анализировать»).
-- Written by `ai.py` and by nothing else; read by hand in psql, there is no endpoint on it.
--
-- `request` holds the prompt and a line about each attached file — never the file's bytes, which
-- are in the bucket already, and never the API key. A row that says 'skipped' is a call that was
-- never made because no model was configured: it has no `response` for exactly that reason.
CREATE TABLE IF NOT EXISTS ai_requests (
    id          bigserial   PRIMARY KEY,
    provider    text        NOT NULL,                   -- gemini, openai — or which one was asked for
    model       text,                                   -- and its model, as the setting named it
    operation   text        NOT NULL,                   -- describe_media / transcribe_speech / plan_montage
    ref         text,                                   -- id of the media item or montage it is about
    status      text        NOT NULL DEFAULT 'sent'
                            CHECK (status IN ('sent', 'answered', 'failed', 'skipped')),
    request     jsonb       NOT NULL,
    response    jsonb,
    error       text,
    took_ms     integer,
    created_at  timestamptz NOT NULL DEFAULT now(),
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- «Что уже спрашивали про этот кадр» — the one lookup that is not just the tail of the table.
CREATE INDEX IF NOT EXISTS ai_requests_ref_idx ON ai_requests (ref) WHERE ref IS NOT NULL;

-- What is said in one clip, word for word: one row per media item, written by the `media_speech`
-- job (`speech.py`) once a video is ready, and read from then on by everything that needs the words
-- instead of asking for them again (doc-03, «выходы шагов пайплайна переиспользуются»).
--
-- There is no column for who said it, and that is the design: speech is transcribed, speakers are
-- not identified (decision-11). An empty `transcript` is an answer and not a gap — the clip has no
-- sound, or nobody speaks in it, and nobody will be asked about it a second time.
--
-- `media_id` is the `media_items` row as text, the way `jobs.ref` names it. No foreign key: that
-- table belongs to the api's migrations, and this file has to apply to a database they have never
-- touched.
CREATE TABLE IF NOT EXISTS media_speech (
    media_id    text        PRIMARY KEY,
    transcript  text        NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now()
);
