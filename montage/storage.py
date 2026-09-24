"""R2: the bucket everything the worker touches comes out of and goes back into.

The same four variable names the api reads, so one `.env` serves both. Nothing here knows what is
in an object — a key is a key.
"""

import logging
import os

import boto3
from botocore.config import Config

log = logging.getLogger("montage.storage")


def client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"].rstrip("/"),
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        # R2 ignores the region and wants the literal `auto`; MinIO signs with `us-east-1` unless
        # told otherwise. A knob because those two disagree — same as the api.
        region_name=os.environ.get("R2_REGION", "auto"),
        # `<endpoint>/<bucket>/<key>`, which is how the api addresses it too. The alternative puts
        # the bucket in the hostname, and `remember.localhost` resolves nowhere.
        config=Config(s3={"addressing_style": "path"}),
    )


def bucket():
    return os.environ["R2_BUCKET"]


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
