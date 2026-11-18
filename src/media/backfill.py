"""Отпечатки для уже загруженных кадров: считается по готовым миниатюрам, ничего не пересобирает."""

import logging
import pathlib
import tempfile

from src.integrations import db, storage
from src.media import perceptual

log = logging.getLogger("media.backfill")

MISSING = """
select distinct m.id, t.storage_key
from media_items m
join project_items i on i.original_id = m.id
join lateral jsonb_each_text(i.previews_ids) p on p.key = 'middle'
join media_items t on t.id = p.value::uuid
where m.kind in ('photo', 'video') and m.perceptual_hash = ''
"""


def run() -> None:
    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory() as directory, db.connect() as conn:
        rows = conn.execute(MISSING).fetchall()
        log.info("media without a perceptual hash: %s", len(rows))
        for media_id, key in rows:
            path = pathlib.Path(directory) / f"{media_id}.jpg"
            try:
                storage.download(key, path)
                digest = perceptual.hash_of(path)
            except Exception:
                log.exception("media %s skipped", media_id)
                continue
            finally:
                path.unlink(missing_ok=True)
            conn.execute(
                "update media_items set perceptual_hash = %s where id = %s", (digest, media_id)
            )
    log.info("perceptual hashes filled")
