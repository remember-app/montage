import logging
import uuid

import psycopg

log = logging.getLogger("jobs.montage")


def run(conn: psycopg.Connection, ref: uuid.UUID) -> None:
    raise NotImplementedError("montage is not implemented yet")


def give_up(conn: psycopg.Connection, ref: uuid.UUID) -> None:
    log.error("montage %s failed for good", ref)
