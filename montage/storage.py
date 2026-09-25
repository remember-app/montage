"""The bucket everything the worker touches comes out of and goes back into — through one narrow
door: `download`, `upload`, `exists`, `delete`, `abort` and `free`, plus `large_copy` and `missing`.

Two drivers stand behind it, and STORAGE_DRIVER picks one:

    s3, the default — boto3 against anything that speaks S3: MinIO locally, Google's XML API with
    an HMAC key. Settings `R2_*`, the same ones the api reads, so one `.env` serves both; they are
    still called `R2_*` after the storage this started on.

    google — Google Cloud Storage's XML API with a service account's JSON key, for a bucket nobody
    hands out an HMAC key to (decision-16). Everything goes under GOOGLE_STORAGE_PREFIX: that bucket
    is the company's, shared and public, and whatever lies next to ours is somebody else's.

A key is the logical one the api keeps in its tables — `originals/<uuid>`, `previews/<id>.jpg` —
and nothing here knows what is in an object.
"""

import contextlib
import functools
import json
import logging
import os
import re
import time
import urllib.parse

import boto3
import requests
from botocore.config import Config
from botocore.exceptions import ClientError
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account

GOOGLE = "https://storage.googleapis.com"
GOOGLE_SCOPE = "https://www.googleapis.com/auth/devstorage.read_write"
TIMEOUT = (10, 60)  # seconds: to connect, and for any one read or write on the socket after that
ATTEMPTS = 4  # a request Google answered with a 5xx or a 429, or that the network dropped
BACKOFF_SECONDS = 1  # attempt n + 1 waits BACKOFF_SECONDS * 2 ** (n - 1): 1, 2, 4 seconds
CHUNK_BYTES = 1024 * 1024  # all of a download that is ever in memory at once

log = logging.getLogger("montage.storage")


def client():
    """The driver STORAGE_DRIVER names, built once per set of settings and shared from then on.

    Once, because the google driver holds its access token and boto3 its connections — building
    either for every call would ask Google for a token per call. Per set of settings, so that
    changing them (the tests do) gets a driver of those settings. A setting that is not there is a
    `KeyError` naming it; one that is there and wrong, a `ValueError` saying why — here, before
    anything is sent anywhere.
    """
    driver = os.environ.get("STORAGE_DRIVER", "").strip().lower() or "s3"
    if driver == "s3":
        return _built(
            _S3,
            os.environ["R2_ENDPOINT"],
            os.environ["R2_BUCKET"],
            os.environ["R2_ACCESS_KEY_ID"],
            os.environ["R2_SECRET_ACCESS_KEY"],
            os.environ.get("R2_REGION", "auto"),
        )
    if driver == "google":
        return _built(
            _Google,
            os.environ["GOOGLE_STORAGE_CREDENTIALS"],
            os.environ["GOOGLE_STORAGE_BUCKET"],
            os.environ["GOOGLE_STORAGE_PREFIX"],
        )
    raise ValueError(f"STORAGE_DRIVER is {driver!r}: it is s3 or google")


@functools.cache
def _built(driver, *settings):
    return driver(*settings)


def download(key, into):
    """The object under `key` into the file `into`."""
    client().download(key, into)


def upload(path, key, content_type):
    client().upload(path, key, content_type)


def exists(key):
    try:
        client().head(key)
    except Exception as exc:
        if missing(exc):
            return False
        raise
    return True


def delete(key):
    """`key` out of the bucket. Nothing under it any more is that very outcome, not a failure."""
    _unless_missing(client().delete, key)


def abort(key, upload_id):
    """A multipart upload's parts released. One already aborted or completed is fine too."""
    _unless_missing(client().abort, key, upload_id)


def free(key, upload_id=None):
    """Whatever an upload left under `key` taken out of the bucket: its unfinished parts, if it
    was a multipart, and the object, if it was ever completed.

    Either may be gone already — the upload aborted or finished, the object never made — and that
    is the outcome wanted, not a failure. The abort goes first: a phone completing the upload after
    it is refused, one that completed before it leaves an object the delete then takes.
    """
    if upload_id:
        abort(key, upload_id)
    delete(key)


def _unless_missing(call, *args):
    try:
        call(*args)
    except Exception as exc:
        if not missing(exc):
            raise


def missing(exc):
    """Whether the bucket answered «nothing there»: no object under the key, no such upload.

    By the status and not by the code: the code is `NoSuchKey` or `NoSuchUpload`, and a HEAD, which
    has no body, carries none but `404`. Storages differ on when they say it at all — S3, MinIO and
    R2 answer a DELETE of a key that holds nothing with 204, Google with 404 `NoSuchKey`. A bucket
    that is not there is a 404 too, and that one is a misconfiguration to be loud about, not an
    object already gone.
    """
    if isinstance(exc, ClientError):
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        code = exc.response.get("Error", {}).get("Code")
    elif isinstance(exc, GoogleError):
        status, code = exc.status, exc.code
    else:
        return False
    return status == 404 and code != "NoSuchBucket"


def large_copy(preview_key, into):
    """A photograph's 2048px copy out of the bucket, or None when it will not come.

    The copy `media.preview` made, fetched by everything that would rather look at a photograph
    than at the forty megapixels it was uploaded at: the отсев judges it
    (`media.screening`, `video.montage`) and the montage cuts its pieces from it.

    None and not an exception, because nobody who asks for it has to have it: the original is the
    file this copy was made from, and every caller can fall back on it — slower, and for the отсев
    measured on a coarser sample, but never wrong. A copy that is missing, not made yet, or in an
    unreachable bucket is not worth failing an ordered video or a screening proposal over.
    """
    try:
        download(preview_key, into)
        return into
    except Exception as exc:
        log.warning(
            "could not take the large copy %s, falling back on the original: %s", preview_key, exc
        )
        return None


class _S3:
    """boto3 against anything that speaks S3: MinIO locally, Google's XML API with an HMAC key."""

    def __init__(self, endpoint, bucket, key_id, secret, region):
        self.bucket = bucket
        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint.rstrip("/"),
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
            # Google takes any region in the signature and documents `auto`; MinIO signs with
            # `us-east-1` unless told otherwise. A knob because those two disagree — same as the
            # api.
            region_name=region,
            config=Config(
                # `<endpoint>/<bucket>/<key>`, which is how the api addresses it too. The
                # alternative puts the bucket in the hostname, and `remember.localhost` resolves
                # nowhere.
                s3={"addressing_style": "path"},
                # Since 1.36 boto3 checksums every upload of its own accord — over HTTPS as an
                # `aws-chunked` body with the sum in a trailer — and asks every download for a sum
                # back. Those are S3's own headers, Google's XML API does not list them, and uploads
                # sent to it that way come back refused. So: only where an operation cannot go
                # without one, and none the worker makes is such. Here and not in a version pin, so
                # that no upgrade brings the default back; `test_storage.py` holds every request to
                # it.
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
            ),
        )

    def __str__(self):
        return f"s3 {self.s3.meta.endpoint_url}/{self.bucket}"

    def download(self, key, into):
        self.s3.download_file(self.bucket, key, str(into))

    def upload(self, path, key, content_type):
        # Large files go up in parts: boto3 cuts anything over 8 MiB by itself.
        self.s3.upload_file(str(path), self.bucket, key, ExtraArgs={"ContentType": content_type})

    def head(self, key):
        self.s3.head_object(Bucket=self.bucket, Key=key)

    def delete(self, key):
        self.s3.delete_object(Bucket=self.bucket, Key=key)

    def abort(self, key, upload_id):
        self.s3.abort_multipart_upload(Bucket=self.bucket, Key=key, UploadId=upload_id)


class _Google:
    """Google Cloud Storage's XML API with a service account's JSON key, and never outside the
    prefix.

    Every object is `https://storage.googleapis.com/<bucket>/<prefix>/<key>`, asked for with the
    account's access token as `Authorization: Bearer`; google-auth fetches the token (scope
    devstorage.read_write) and renews it before it runs out. An upload is one PUT whatever its size
    — the XML API takes an object whole, so there are no parts to cut — with the file streamed off
    disk, and a download is streamed into its file: neither is ever held in memory.
    """

    def __init__(self, credentials, bucket, prefix):
        self.prefix = prefix.strip().rstrip("/")
        if not _plain(self.prefix):
            raise ValueError(
                "GOOGLE_STORAGE_PREFIX has to name a folder of ours, like remember/dev-media: "
                "without one the worker would write into the root of a bucket that is not ours"
            )
        if not bucket.strip():
            raise ValueError("GOOGLE_STORAGE_BUCKET is empty")
        try:
            account = service_account.Credentials.from_service_account_info(
                json.loads(credentials), scopes=[GOOGLE_SCOPE]
            )
        except Exception:
            # `from None`: what the parser said about the key may quote the key, and this is read
            # in a log.
            raise ValueError(
                "GOOGLE_STORAGE_CREDENTIALS is not a service account key: it takes the key "
                "file's JSON, on one line"
            ) from None
        # No 401s for the session to handle: it would retry with the same body, and an upload's
        # file, read to its end by then, would go up empty. `_send` renews the token itself.
        self._session = AuthorizedSession(account, refresh_status_codes=())
        self._objects = f"{GOOGLE}/{bucket.strip()}/"
        # A key that already starts where the prefix does is a whole object name, ours or a
        # neighbour's, and not a key of ours: prefixed again it would land somewhere nobody looks.
        self._foreign = self.prefix.split("/")[0] + "/"

    def __str__(self):
        return f"google {self._objects}{self.prefix}/"

    def url(self, key):
        """Where `key` lives in the bucket — only ever under the prefix, the one place it is added.

        Whatever could step outside it is refused here, before a request is made: an empty key, one
        with `..` or `.` or an empty segment in it (a leading `/` is one), one with anything
        unprintable in it, and one that carries a prefix of its own. The rest is sent as it is,
        percent-encoded, so no `%2e%2e` of the key's own is ever decoded into a step up either.
        """
        if not _plain(key) or key.startswith(self._foreign):
            raise ValueError(f"{key!r} is not a key under the storage prefix: refusing to touch it")
        return self._objects + urllib.parse.quote(f"{self.prefix}/{key}")

    def download(self, key, into):
        self._send("GET", key, into=into)

    def upload(self, path, key, content_type):
        self._send("PUT", key, headers={"Content-Type": content_type}, body=path)

    def head(self, key):
        self._send("HEAD", key)

    def delete(self, key):
        self._send("DELETE", key)

    def abort(self, key, upload_id):
        # The api starts a multipart upload; the worker only ever aborts one (`sweeps`).
        self._send("DELETE", key, params={"uploadId": upload_id})

    def _send(self, method, key, *, params=None, headers=None, body=None, into=None):
        """One request, answered with a yes — or a `GoogleError`, or the network's own exception.

        A 5xx, a 429 or a dropped connection is tried again, up to ATTEMPTS in all and further
        apart each time: every request here can be repeated, a whole-object PUT and a DELETE
        included. A 401 renews the token once. Every attempt opens the body afresh and rewrites
        the download from its first byte, so none goes up or comes down half.
        """
        url, renewed = self.url(key), False
        for attempt in range(ATTEMPTS):
            if attempt:
                time.sleep(BACKOFF_SECONDS * 2 ** (attempt - 1))
            try:
                with contextlib.ExitStack() as held:
                    data = held.enter_context(open(body, "rb")) if body else None
                    answer = held.enter_context(
                        self._session.request(
                            method,
                            url,
                            params=params,
                            data=data,
                            headers=headers,
                            timeout=TIMEOUT,
                            stream=into is not None,
                        )
                    )
                    if answer.ok:
                        if into is not None:
                            with open(into, "wb") as file:
                                for chunk in answer.iter_content(CHUNK_BYTES):
                                    file.write(chunk)
                        return answer
                    failure = GoogleError(method, url, answer)
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as exc:
                failure = exc
            else:
                if answer.status_code == 401 and not renewed:
                    # A token Google no longer takes though it has not run out by our clock:
                    # forgotten, and the next attempt asks for another.
                    self._session.credentials.token = None
                    renewed = True
                elif answer.status_code != 429 and answer.status_code < 500:
                    raise failure
            log.warning(
                "storage %s %s failed on attempt %s/%s: %s",
                method,
                key,
                attempt + 1,
                ATTEMPTS,
                failure,
            )
        raise failure


class GoogleError(Exception):
    """Google's «no»: its status and the code its XML body names (`NoSuchKey`, `NoSuchUpload`,
    `AccessDenied`…) — or the status again for a HEAD, which comes without a body."""

    def __init__(self, method, url, answer):
        self.status = answer.status_code
        said = dict(re.findall(r"<(Code|Message)>(.*?)</\1>", answer.text, re.DOTALL))
        self.code = said.get("Code", str(self.status))
        told = f"{method} {url}: {self.status} {self.code} {said.get('Message', '')}"
        super().__init__(told.rstrip())


def _plain(path):
    """A path of plain names: not empty, no `.`, `..` or empty segment, nothing unprintable."""
    return (
        bool(path)
        and path.isprintable()
        and all(part not in ("", ".", "..") for part in path.split("/"))
    )
