-- The montage worker's own tables: the shared job queue, the journal of what was asked of the cloud
-- model, what the model said about each frame, and the frame the отсев compares retakes by. The API
-- only inserts rows into `jobs` and never touches the others.
-- Idempotent on purpose: `python -m montage init` applies it on every start-up of a new database.

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

-- The two queries the worker runs against this table on every turn of its loop: the claim, and the
-- search for jobs a dead worker left `running` (`reap`). `jobs` keeps a row for every job ever run,
-- so neither may scan all of it.
CREATE INDEX IF NOT EXISTS jobs_due_idx ON jobs (run_after) WHERE status = 'queued';
CREATE INDEX IF NOT EXISTS jobs_running_idx ON jobs (updated_at) WHERE status = 'running';

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

-- What the provider counted a call at, off its own answer — the bill is paid in tokens. Added to a
-- table that was already in use, so `init` brings an older journal up to date and leaves its rows as
-- they were, with these empty. Empty too on every row whose provider did not say: never made up.
--
-- `usage` is the provider's count whole — Gemini splits the input by modality, and audio is priced
-- apart; the three numbers out of it make a bill one query for either provider. `output_tokens` is
-- all that is billed at the output rate: Gemini's thinking is counted apart from its answer and is
-- added in here, OpenAI's reasoning is inside its output already.
ALTER TABLE ai_requests
    ADD COLUMN IF NOT EXISTS usage         jsonb,
    ADD COLUMN IF NOT EXISTS input_tokens  integer,
    ADD COLUMN IF NOT EXISTS output_tokens integer,
    ADD COLUMN IF NOT EXISTS total_tokens  integer;

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

-- What the model saw in one photo or clip, kept so that nobody pays to see it twice (doc-03:
-- «Описание того, что происходит на фото и видео, обязательно сохраняется и переиспользуется»).
-- Written by `understanding.py`, read by it, by the director and by the montage's analysis stage;
-- one row per media item, written once and never updated — a frame does not change after it was
-- described.
--
-- `cast_labels` are this memory's own «человек-1», «животное-2»: stable inside one memory and
-- meaningless outside it, which is why the memory is on the row. No foreign key to `media_items`
-- for the same reason `ai_requests.ref` has none — that table belongs to the api's migrations and
-- this file has to apply to a database where they have not run yet.
CREATE TABLE IF NOT EXISTS media_understandings (
    media_id    text        PRIMARY KEY,           -- media_items.id, as text like jobs.ref
    memory_id   text        NOT NULL,              -- whose cast the labels below belong to
    description text        NOT NULL,              -- what happens in the frame, in a sentence
    cast_labels text[]      NOT NULL DEFAULT '{}', -- who is in it, by this memory's labels
    created_at  timestamptz NOT NULL DEFAULT now()
);

-- The same answer by fields, as the journal's card of the frame shows it (contract of the api,
-- `details` у анализа): place, light, mood, shot, and at a clip — camera and its best moments,
-- with the second each of them stands at. Free-form jsonb rather than a column per field, because
-- the card is the same free `jsonb` the journal keeps and the fields are the model's, not ours —
-- one `description` column would have thrown away everything but the first sentence.
--
-- Added to a table that was already in use, so `init` brings an older one up to date and leaves
-- its rows as they were, with this empty on every one of them.
ALTER TABLE media_understandings
    ADD COLUMN IF NOT EXISTS details jsonb NOT NULL DEFAULT '{}';

-- «Кого это воспоминание уже назвало» — asked before every description, to be given to the model.
CREATE INDEX IF NOT EXISTS media_understandings_memory_idx ON media_understandings (memory_id);

-- What the отсев tells two shots of one moment apart by: the sampled frame of a file's best moment
-- (96×54 grey bytes, `selection.Piece.frame`) and how sharp it is. Written by `media_screening`
-- beside its verdict, and read by it back for the whole memory every time a file is screened, to
-- say which frames are retakes of another (`screening.py`). Kept and not read off the files again,
-- because every new frame asks that of all of them: a memory re-downloaded and re-decoded on each
-- upload, its clips whole, for five kilobytes a file. One row per media item, replaced whenever
-- the file is judged again; none for a file with no frame in it. No foreign key, for the reason
-- `media_speech` has none.
CREATE TABLE IF NOT EXISTS media_frames (
    media_id    text             PRIMARY KEY,
    frame       bytea            NOT NULL,
    sharpness   double precision NOT NULL
);
