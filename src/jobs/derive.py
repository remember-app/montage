import logging
import uuid

import psycopg

from src.media import derive

log = logging.getLogger("jobs.derive")


def run(conn: psycopg.Connection, ref: uuid.UUID) -> None:
    derive.run(conn, ref)


def give_up(conn: psycopg.Connection, ref: uuid.UUID) -> None:
    log.error("media_derive for %s failed for good", ref)
