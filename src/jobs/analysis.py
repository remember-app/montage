import logging
import uuid

import psycopg

from src.media import analysis

log = logging.getLogger("jobs.analysis")


def run(conn: psycopg.Connection, ref: uuid.UUID) -> None:
    analysis.run(conn, ref)


def give_up(conn: psycopg.Connection, ref: uuid.UUID) -> None:
    log.error("media_analysis for %s failed for good", ref)
