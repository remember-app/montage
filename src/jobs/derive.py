import logging
import uuid

import psycopg

log = logging.getLogger("jobs.derive")


def run(conn: psycopg.Connection, ref: uuid.UUID) -> None:
    raise NotImplementedError("media_derive is not implemented yet")


def give_up(conn: psycopg.Connection, ref: uuid.UUID) -> None:
    log.error("media_derive for %s failed for good", ref)
