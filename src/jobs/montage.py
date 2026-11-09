import logging
import uuid

import psycopg

from src.movie import editor, notify

log = logging.getLogger("jobs.montage")


def run(conn: psycopg.Connection, ref: uuid.UUID) -> None:
    row = conn.execute("select status, manifest from montages where id = %s", (ref,)).fetchone()
    if row is None:
        log.info("montage %s is gone, nothing to assemble", ref)
        return
    status, manifest = row
    if status == "done":
        log.info("montage %s is ready already, nothing to assemble", ref)
        return
    commands = ((manifest or {}).get("editingParams") or {}).get("commands") or []
    if not commands:
        raise RuntimeError("montage has no editing plan")
    conn.execute("update montages set status = 'processing' where id = %s", (ref,))
    editor.render(conn, ref, commands)


def give_up(conn: psycopg.Connection, ref: uuid.UUID) -> None:
    row = conn.execute(
        "select error from jobs where kind = 'montage' and ref = %s order by id desc limit 1",
        (ref,),
    ).fetchone()
    error = (row[0] if row and row[0] else "montage failed")[:2000]
    conn.execute(
        "update montages set status = 'failed', error = %s where id = %s and status <> 'done'",
        (error, ref),
    )
    notify.failed(ref, error)
    log.error("montage %s failed for good: %s", ref, error)
