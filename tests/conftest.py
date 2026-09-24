"""Общие фикстуры набора: соединение с базой, бакет и чьё-то воспоминание, на которое всё вешают.

Всё живое: настоящий Postgres по DATABASE_URL и настоящий бакет по R2_ENDPOINT.

Внимание: они чистят таблицу `jobs` той базы, на которую указывает DATABASE_URL, и пишут в тот
бакет, на который указывает R2_ENDPOINT.
"""

import pytest

from montage import storage, worker


@pytest.fixture
def conn():
    with worker.connect() as connection:
        worker.init_schema(connection)
        connection.execute("DELETE FROM jobs")
        yield connection


@pytest.fixture
def s3():
    return storage.client()


@pytest.fixture
def memory(conn):
    """Somebody's memory to hang media on. `media_items` and everything it points at belong to the
    api's migrations, not to `schema.sql` — without them there is nothing to test against."""
    if conn.execute("SELECT to_regclass('media_items')").fetchone()[0] is None:
        pytest.skip("media_items is missing — run the api's migrations first (see README.md)")
    user_id = conn.execute(
        "INSERT INTO users (name) VALUES ('montage test') RETURNING id"
    ).fetchone()[0]
    memory_id = conn.execute(
        """INSERT INTO memories (organizer_id, title, event_start)
           VALUES (%s, 'montage test', now()) RETURNING id""",
        (user_id,),
    ).fetchone()[0]
    yield memory_id, user_id
    # Everything hung on the memory goes with it, the worker's own tables included: nothing has a
    # foreign key to `media_items` (`schema.sql` applies to databases the api has never touched),
    # so a description left behind here would outlive every row that explains it.
    conn.execute("DELETE FROM media_understandings WHERE memory_id = %s", (str(memory_id),))
    conn.execute("DELETE FROM media_items WHERE memory_id = %s", (memory_id,))
    conn.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
    conn.execute("DELETE FROM users WHERE id = %s", (user_id,))
