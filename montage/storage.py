"""The bucket everything the worker touches comes out of and goes back into — Google Cloud Storage
over its S3-compatible XML API in production, MinIO locally.

The same variable names the api reads, so one `.env` serves both; they are still called `R2_*`
after the storage this started on. Nothing here knows what is in an object — a key is a key.
"""

import logging
import os

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

log = logging.getLogger("montage.storage")


def client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"].rstrip("/"),
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        # Google takes any region in the signature and documents `auto`; MinIO signs with
        # `us-east-1` unless told otherwise. A knob because those two disagree — same as the api.
        region_name=os.environ.get("R2_REGION", "auto"),
        config=Config(
            # `<endpoint>/<bucket>/<key>`, which is how the api addresses it too. The alternative
            # puts the bucket in the hostname, and `remember.localhost` resolves nowhere.
            s3={"addressing_style": "path"},
            # Since 1.36 boto3 checksums every upload of its own accord — over HTTPS as an
            # `aws-chunked` body with the sum in a trailer — and asks every download for a sum
            # back. Those are S3's own headers, Google's XML API does not list them, and uploads
            # sent to it that way come back refused. So: only where an operation cannot go without
            # one, and none the worker makes is such. Here and not in a version pin, so that no
            # upgrade brings the default back; `test_storage.py` holds every request to it.
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def bucket():
    return os.environ["R2_BUCKET"]


def missing(exc):
    """Whether the bucket answered «nothing there»: no object under the key, no such upload.

    By the status and not by the code: the code is `NoSuchKey` or `NoSuchUpload`, and a HEAD, which
    has no body, carries none but `404`. Storages differ on when they say it at all — S3, MinIO and
    R2 answer a DELETE of a key that holds nothing with 204, Google with 404 `NoSuchKey`. A bucket
    that is not there is a 404 too, and that one is a misconfiguration to be loud about, not an
    object already gone.
    """
    return (
        isinstance(exc, ClientError)
        and exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode") == 404
        and exc.response.get("Error", {}).get("Code") != "NoSuchBucket"
    )


def free(s3, key, upload_id=None):
    """Whatever an upload left under `key` taken out of the bucket: its unfinished parts, if it
    was a multipart, and the object, if it was ever completed.

    Either may be gone already — the upload aborted or finished, the object never made — and that
    is the outcome wanted, not a failure. The abort goes first: a phone completing the upload after
    it is refused, one that completed before it leaves an object the delete then takes.
    """
    if upload_id:
        _unless_missing(s3.abort_multipart_upload, Key=key, UploadId=upload_id)
    _unless_missing(s3.delete_object, Key=key)


def _unless_missing(call, **params):
    try:
        call(Bucket=bucket(), **params)
    except ClientError as exc:
        if not missing(exc):
            raise


def upload(s3, path, key, content_type):
    s3.upload_file(str(path), bucket(), key, ExtraArgs={"ContentType": content_type})


def large_copy(s3, preview_key, into):
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
        s3.download_file(bucket(), preview_key, str(into))
        return into
    except Exception as exc:
        log.warning(
            "could not take the large copy %s, falling back on the original: %s", preview_key, exc
        )
        return None
