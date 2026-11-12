import functools
import pathlib

import boto3
from botocore.client import BaseClient
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from src import config


@functools.cache
def client() -> BaseClient:
    return boto3.client(
        "s3",
        endpoint_url=config.s3_endpoint(),
        region_name=config.s3_region(),
        aws_access_key_id=config.s3_access_key_id(),
        aws_secret_access_key=config.s3_secret_access_key(),
        config=BotoConfig(
            connect_timeout=10,
            read_timeout=60,
            retries={"max_attempts": 3},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def media_key(media_id: str, file_format: str) -> str:
    return f"remember/{file_format}/{media_id}"


def download(key: str, path: pathlib.Path) -> None:
    client().download_file(config.s3_bucket(), key, str(path))


def upload(path: pathlib.Path, key: str, content_type: str) -> None:
    client().upload_file(
        str(path), config.s3_bucket(), key, ExtraArgs={"ContentType": content_type}
    )


def delete(key: str) -> None:
    client().delete_object(Bucket=config.s3_bucket(), Key=key)


def abort_multipart(key: str, upload_id: str) -> None:
    try:
        client().abort_multipart_upload(Bucket=config.s3_bucket(), Key=key, UploadId=upload_id)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in ("NoSuchUpload", "404"):
            raise


def exists(key: str) -> bool:
    try:
        client().head_object(Bucket=config.s3_bucket(), Key=key)
    except ClientError:
        return False
    return True
