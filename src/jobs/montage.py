import logging
import uuid

import psycopg

from src.movie import directing, editor, notify, preparation, rhythm

log = logging.getLogger("jobs.montage")


def _manifest(conn: psycopg.Connection, ref: uuid.UUID) -> dict:
    row = conn.execute("select manifest from montages where id = %s", (ref,)).fetchone()
    return (row[0] if row else None) or {}


def run(conn: psycopg.Connection, ref: uuid.UUID, level: str | None = None) -> None:
    row = conn.execute("select status, manifest from montages where id = %s", (ref,)).fetchone()
    if row is None:
        log.info("montage %s is gone, nothing to assemble", ref)
        return
    status, manifest = row
    if status == "done":
        log.info("montage %s is ready already, nothing to assemble", ref)
        return
    manifest = manifest or {}
    settings = manifest.get("preparationConfig") or {}
    level = preparation.level(settings)
    if level != "low":
        raise RuntimeError(f"montage level {level!r} is not implemented yet; only low is available")
    conn.execute("update montages set status = 'processing' where id = %s", (ref,))
    log.info("montage %s level: %s", ref, level)

    if not ((manifest.get("directingInput") or {}).get("mergedMontageData")):
        preparation.build(conn, ref)
        manifest = _manifest(conn, ref)
    if not ((manifest.get("editingParams") or {}).get("commands")):
        directing.direct(conn, ref, manifest["directingInput"]["mergedMontageData"])
        manifest = _manifest(conn, ref)

    commands = (manifest.get("editingParams") or {}).get("commands") or []
    if not commands:
        raise RuntimeError("montage has no editing plan")
    if not (manifest.get("editingParams") or {}).get("rhythm"):
        commands = rhythm.apply(conn, ref, commands)

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
