"""Строки, которые тестам нужны в базе, и ключи, которые нужны в бакете, — и как их прочитать.

Всё здесь — таблицы api (`media_items`, `montages`, `montage_frames`) и собственная `jobs` воркера:
тесты кладут в них ровно то, что кладёт api, и читают ровно то, что читает он.
"""

import collections
import json
import uuid

import pytest

from montage import storage

from material import jpeg, video


def add_item(conn, memory, kind, *, size=0, status="processing", age_hours=0, upload_id=None):
    """A `media_items` row as the api leaves it, and the key its original would be under."""
    memory_id, user_id = memory
    item_id = uuid.uuid4()
    conn.execute(
        """INSERT INTO media_items
             (id, memory_id, uploaded_by, kind, status, original_key, size_bytes, upload_id,
              created_at)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now() - make_interval(hours => %s))""",
        (
            item_id,
            memory_id,
            user_id,
            kind,
            status,
            f"originals/{item_id}",
            size,
            upload_id,
            age_hours,
        ),
    )
    return str(item_id), f"originals/{item_id}"


def read_item(conn, item_id):
    return conn.execute(
        """SELECT status, thumbnail_key, preview_key, duration_seconds
           FROM media_items WHERE id = %s""",
        (item_id,),
    ).fetchone()


def read_frame_info(conn, item_id):
    """The camera and the place as the sheet «Сведения о кадре» reads them off the row."""
    return conn.execute(
        """SELECT camera_model, focal_length_mm, aperture, latitude, longitude
           FROM media_items WHERE id = %s""",
        (item_id,),
    ).fetchone()


def read_size(conn, item_id):
    """The sheet's «Размер 4032 × 3024», off the same row."""
    return conn.execute(
        "SELECT width, height FROM media_items WHERE id = %s", (item_id,)
    ).fetchone()


def fetch(s3, key, path):
    s3.download_file(storage.bucket(), key, str(path))
    return path


def only_jobs_for(conn, item_id, type_="media_preview"):
    """Whatever else this database holds is not this test's business — the sweep is over the whole
    table, and a dev stand's own photos would otherwise be queued into the run."""
    conn.execute("DELETE FROM jobs WHERE ref IS DISTINCT FROM %s", (item_id,))
    return conn.execute(
        "SELECT count(*) FROM jobs WHERE type = %s AND ref = %s", (type_, item_id)
    ).fetchone()[0]


def order(conn, memory, params):
    """A `montages` row as the api leaves it when a video is ordered."""
    if conn.execute("SELECT to_regclass('montages')").fetchone()[0] is None:
        pytest.skip("montages is missing — run the api's migrations first (see README.md)")
    memory_id, user_id = memory
    return str(
        conn.execute(
            """INSERT INTO montages (memory_id, ordered_by, params)
           VALUES (%s, %s, %s::jsonb) RETURNING id""",
            (memory_id, user_id, json.dumps(params)),
        ).fetchone()[0]
    )


def read_montage(conn, montage_id):
    return conn.execute(
        "SELECT status, video_key, duration_seconds, error_text FROM montages WHERE id = %s",
        (montage_id,),
    ).fetchone()


def ready_material(conn, s3, memory, tmp_path):
    """Three photographs and a clip in the bucket, `ready`, as the previews leave them. Their ids
    come back in the order the montage reads them, for the tests that name one in the order."""
    ids = []
    for index in range(3):
        item_id, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(jpeg(tmp_path / f"photo-{index}.jpg")), storage.bucket(), key)
        ids.append(item_id)
    item_id, key = add_item(conn, memory, "video", status="ready")
    s3.upload_file(str(video(tmp_path / "clip.mp4")), storage.bucket(), key)
    return [*ids, item_id]


def layout_of(conn, montage_id):
    """Which frames the montage is made of and where each of them stands in it, in the order they
    are shown — `montage_frames`, the table the frame sheet's «в каких роликах» is answered from."""
    if conn.execute("SELECT to_regclass('montage_frames')").fetchone()[0] is None:
        pytest.skip("montage_frames is missing — run the api's migrations first (see README.md)")
    return conn.execute(
        """SELECT media_id::text, start_seconds, duration_seconds
           FROM montage_frames WHERE montage_id = %s ORDER BY start_seconds""",
        (montage_id,),
    ).fetchall()


Entry = collections.namedtuple(
    "Entry",
    "stage media_id thought decision reason series_size labels "
    "source_from_seconds source_to_seconds story_part story_part_title focus",
)


def journal_ready(conn):
    if conn.execute("SELECT to_regclass('montage_stages')").fetchone()[0] is None:
        pytest.skip("montage_stages is missing — run the api's migrations first (see README.md)")


def stages_of(conn, montage_id):
    """Этапы ролика — в том порядке, в котором их прошли: у этапа своего номера нет, порядок в
    базе держится только временем начала (а в ответе api — его собственным списком)."""
    return {
        stage: {
            "state": state,
            "started_at": started_at,
            "finished_at": finished_at,
            "eta": eta,
            "done": done,
            "total": total,
            "outcome": outcome,
        }
        for stage, state, started_at, finished_at, eta, done, total, outcome in conn.execute(
            """SELECT stage, state, started_at, finished_at, eta_seconds, done_count, total_count,
                      outcome
                 FROM montage_stages WHERE montage_id = %s ORDER BY started_at NULLS LAST, stage""",
            (montage_id,),
        ).fetchall()
    }


def entries_of(conn, montage_id, stage=None):
    """Реплики журнала в том порядке, в каком они легли, — `id` и есть курсор приложения.

    Рамка «куда смотрит» читается так же, как её отдаёт api, — одним значением из четырёх долей
    или пустотой: три доли из четырёх база не приняла бы вовсе.
    """
    return [
        Entry(*row[:-4], row[-4:] if row[-4] is not None else None)
        for row in conn.execute(
            """SELECT stage, media_id::text, thought, decision, reason, series_size, labels,
                      source_from_seconds, source_to_seconds, story_part, story_part_title,
                      focus_x, focus_y, focus_w, focus_h
                 FROM montage_journal_entries
                WHERE montage_id = %s AND (%s::text IS NULL OR stage = %s) ORDER BY id""",
            (montage_id, stage, stage),
        ).fetchall()
    ]
