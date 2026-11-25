import psycopg

from src import config


def connect() -> psycopg.Connection:
    return psycopg.connect(config.database_url(), autocommit=True)
