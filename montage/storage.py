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
TIMEOUT = (10, 60)
ATTEMPTS = 4
BACKOFF_SECONDS = 1
CHUNK_BYTES = 1024 * 1024

log = logging.getLogger("montage.storage")


def client():
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
    _unless_missing(client().delete, key)


def abort(key, upload_id):
    _unless_missing(client().abort, key, upload_id)


def free(key, upload_id=None):
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
    if isinstance(exc, ClientError):
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        code = exc.response.get("Error", {}).get("Code")
    elif isinstance(exc, GoogleError):
        status, code = exc.status, exc.code
    else:
        return False
    return status == 404 and code != "NoSuchBucket"


def large_copy(preview_key, into):
    try:
        download(preview_key, into)
        return into
    except Exception as exc:
        log.warning(
            "could not take the large copy %s, falling back on the original: %s", preview_key, exc
        )
        return None


class _S3:
    def __init__(self, endpoint, bucket, key_id, secret, region):
        self.bucket = bucket
        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint.rstrip("/"),
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
            region_name=region,
            config=Config(
                s3={"addressing_style": "path"},
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
            ),
        )

    def __str__(self):
        return f"s3 {self.s3.meta.endpoint_url}/{self.bucket}"

    def download(self, key, into):
        self.s3.download_file(self.bucket, key, str(into))

    def upload(self, path, key, content_type):
        self.s3.upload_file(str(path), self.bucket, key, ExtraArgs={"ContentType": content_type})

    def head(self, key):
        self.s3.head_object(Bucket=self.bucket, Key=key)

    def delete(self, key):
        self.s3.delete_object(Bucket=self.bucket, Key=key)

    def abort(self, key, upload_id):
        self.s3.abort_multipart_upload(Bucket=self.bucket, Key=key, UploadId=upload_id)


class _Google:
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
            raise ValueError(
                "GOOGLE_STORAGE_CREDENTIALS is not a service account key: it takes the key "
                "file's JSON, on one line"
            ) from None
        self._session = AuthorizedSession(account, refresh_status_codes=())
        self._objects = f"{GOOGLE}/{bucket.strip()}/"
        self._foreign = self.prefix.split("/")[0] + "/"

    def __str__(self):
        return f"google {self._objects}{self.prefix}/"

    def url(self, key):
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
        self._send("DELETE", key, params={"uploadId": upload_id})

    def _send(self, method, key, *, params=None, headers=None, body=None, into=None):
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
    def __init__(self, method, url, answer):
        self.status = answer.status_code
        said = dict(re.findall(r"<(Code|Message)>(.*?)</\1>", answer.text, re.DOTALL))
        self.code = said.get("Code", str(self.status))
        told = f"{method} {url}: {self.status} {self.code} {said.get('Message', '')}"
        super().__init__(told.rstrip())


def _plain(path):
    return (
        bool(path)
        and path.isprintable()
        and all(part not in ("", ".", "..") for part in path.split("/"))
    )
