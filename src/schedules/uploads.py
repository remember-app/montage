import logging

import psycopg

from src import config
from src.integrations import storage

log = logging.getLogger("schedules.uploads")


def cleanup(conn: psycopg.Connection) -> int:
    rows = conn.execute(
        """
        select id, storage_key, multipart_id
        from uploads
        where status in ('pending', 'uploading')
          and updated_at < now() - make_interval(secs => %s)
        """,
        (config.upload_stale_seconds(),),
    ).fetchall()
    for upload_id, key, multipart_id in rows:
        if multipart_id:
            storage.abort_multipart(key, multipart_id)
        conn.execute(
            """
            update uploads
            set status = 'failed', error_code = 'expired', error = 'upload expired',
                updated_at = now()
            where id = %s and status in ('pending', 'uploading')
            """,
            (upload_id,),
        )
    if rows:
        log.info("closed %s abandoned uploads", len(rows))
    return len(rows)
