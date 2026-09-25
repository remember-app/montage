"""Общие фикстуры набора: соединение с базой, бакет и чьё-то воспоминание, на которое всё вешают.

Всё живое: настоящий Postgres по DATABASE_URL и настоящий бакет по R2_ENDPOINT. Кроме `google`:
это Google Cloud Storage, у которого ответы подменены на месте, — в сеть он не ходит.

Внимание: они чистят таблицу `jobs` той базы, на которую указывает DATABASE_URL, и пишут в тот
бакет, на который указывает R2_ENDPOINT.
"""

import io
import json
from types import SimpleNamespace

import pytest
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from google.auth.credentials import CredentialsWithRegionalAccessBoundary

from montage import storage, worker

TOKEN_URI = "https://oauth2.googleapis.com/token"


@pytest.fixture
def conn():
    with worker.connect() as connection:
        worker.init_schema(connection)
        connection.execute("DELETE FROM jobs")
        yield connection


@pytest.fixture
def s3():
    """The bucket as boto3 sees it, for a test to lay out what the worker will find in it and to
    look at what it left there. The s3 driver's — the suite runs on MinIO."""
    return storage.client().s3


@pytest.fixture
def memory(conn):
    """Somebody's memory to hang media on. `media_items` and everything it points at belong to the
    api's migrations, not to `schema.sql` — without them there is nothing to test against."""
    if conn.execute("SELECT to_regclass('media_items')").fetchone()[0] is None:
        pytest.skip("media_items is missing — run the api's migrations first (see README.md)")
    user_id = conn.execute(
        "INSERT INTO users (name) VALUES ('montage test') RETURNING id"
    ).fetchone()[0]
    memory_id = conn.execute(
        """INSERT INTO memories (organizer_id, title, event_start)
           VALUES (%s, 'montage test', now()) RETURNING id""",
        (user_id,),
    ).fetchone()[0]
    yield memory_id, user_id
    # Everything hung on the memory goes with it, the worker's own tables included: nothing has a
    # foreign key to `media_items` (`schema.sql` applies to databases the api has never touched),
    # so a description left behind here would outlive every row that explains it.
    conn.execute("DELETE FROM media_understandings WHERE memory_id = %s", (str(memory_id),))
    conn.execute("DELETE FROM media_items WHERE memory_id = %s", (memory_id,))
    conn.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
    conn.execute("DELETE FROM users WHERE id = %s", (user_id,))


@pytest.fixture(scope="session")
def service_key():
    """A service account's JSON key as Google hands it out, made up here: a fresh RSA key and
    nobody's account. What the tests sign their token requests with."""
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()
    return json.dumps(
        {
            "type": "service_account",
            "project_id": "montage-tests",
            "private_key_id": "0" * 40,
            "private_key": pem,
            "client_email": "montage-tests@montage-tests.iam.gserviceaccount.com",
            "token_uri": TOKEN_URI,
        }
    )


class FakeGoogle:
    """Both ends of Google the google driver talks to, answered on this machine: the token endpoint
    and the bucket.

    `answers` is what the bucket says next, in order, and 200 to everything after them: a status, a
    status and a body (bytes as they are, a string as the error Google sends with that code), a
    readable object to stream, or an exception the network would raise. `sent` is every request
    that reached the bucket, its body read off as it went; `tokens` how many were handed out, each
    `lifetime` seconds long. A request anywhere but under `objects` fails the test on the spot.
    """

    def __init__(self, objects):
        self.objects, self.answers, self.sent, self.tokens, self.lifetime = objects, [], [], 0, 3600

    def send(self, request, **options):
        if request.url == TOKEN_URI:
            self.tokens += 1
            token = {"access_token": f"token-{self.tokens}", "expires_in": self.lifetime}
            return _reply(request, 200, json.dumps(token).encode())
        assert request.url.startswith(self.objects), f"{request.method} {request.url}: off-prefix"
        body = request.body
        self.sent.append(
            SimpleNamespace(
                method=request.method,
                url=request.url,
                headers=request.headers,
                body=body.read() if hasattr(body, "read") else body,
                from_disk=hasattr(body, "read"),
                **options,
            )
        )
        answer = self.answers.pop(0) if self.answers else 200
        if isinstance(answer, Exception):
            raise answer
        status, body = answer if isinstance(answer, tuple) else (answer, b"")
        if isinstance(body, str):
            body = (
                f"<?xml version='1.0' encoding='UTF-8'?><Error><Code>{body}</Code>"
                f"<Message>Google says {body}.</Message></Error>"
            ).encode()
        return _reply(request, status, body)


def _reply(request, status, body):
    answer = requests.Response()
    answer.status_code, answer.request, answer.url = status, request, request.url
    answer.raw = body if hasattr(body, "read") else io.BytesIO(body)
    return answer


@pytest.fixture
def google(monkeypatch, service_key):
    """The google driver over a bucket shared with others, under `remember/dev-media`, with Google
    itself replaced by a `FakeGoogle` — nothing leaves the machine. A driver of its own for every
    test, so no token outlives the test that fetched it."""
    monkeypatch.setenv("STORAGE_DRIVER", "google")
    monkeypatch.setenv("GOOGLE_STORAGE_CREDENTIALS", service_key)
    monkeypatch.setenv("GOOGLE_STORAGE_BUCKET", "shared-bucket")
    monkeypatch.setenv("GOOGLE_STORAGE_PREFIX", "remember/dev-media")
    monkeypatch.setattr(storage, "BACKOFF_SECONDS", 0)
    fake = FakeGoogle(f"{storage.GOOGLE}/shared-bucket/remember/dev-media/")
    monkeypatch.setattr(
        requests.adapters.HTTPAdapter,
        "send",
        lambda _, request, **options: fake.send(request, **options),
    )
    # After a token comes, google-auth looks up in a thread of its own which regions the account
    # may reach. That thread would outlive the test and go out to the real network.
    monkeypatch.setattr(
        CredentialsWithRegionalAccessBoundary,
        "_is_regional_access_boundary_lookup_required",
        lambda _: False,
    )
    storage._built.cache_clear()
    yield fake
    storage._built.cache_clear()
