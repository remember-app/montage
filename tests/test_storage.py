"""Драйвер s3 так, как его видит Google: что уходит в запросах и что приходит в ответ.

Драйвер s3 (он по умолчанию) говорит по S3 с MinIO локально и с Google Cloud Storage по его
S3-совместимому XML API, если есть HMAC-ключ. Протокол один, мелочи расходятся, и здесь закреплены
те, о которые спотыкается Google (родной доступ по JSON-ключу — `test_storage_google.py`):

- ни один запрос воркера не несёт контрольных сумм S3 — ни заголовков `x-amz-checksum-*`, ни тела
  `aws-chunked` с трейлером. boto3 шлёт их сам, начиная с 1.36, а Google такую загрузку не берёт;
- «уже нет» — это 404 с любым кодом: на удаление ключа, под которым ничего нет, Google отвечает
  404 `NoSuchKey`, а S3 и MinIO — 204.

Последние два теста ходят в настоящий бакет из R2_* — в прогоне набора это MinIO, а направленные
на бакет Google с HMAC-ключом они живой контроль этого пути. Пишут только под своим префиксом и за
собой убирают.
"""

import io
import os
import urllib.parse
import uuid

import pytest
from botocore.awsrequest import AWSResponse
from botocore.exceptions import ClientError

from montage import storage

# Больше порога, с которого `upload_file` режет файл на куски (8 МиБ): так проходят многочастная
# загрузка и скачивание диапазонами. Первый кусок — 8 МиБ, не меньше минимума Google в 5 МиБ.
LARGE = 9 * 1024 * 1024


def listen(s3):
    """Every request the client sends from here on, as it goes out."""
    sent = []
    s3.meta.events.register("before-send.s3", lambda request, **_: sent.append(request))
    return sent


def unfit_for_google(sent):
    """What in these requests Google refuses: S3's checksums, in a header or in an `aws-chunked`
    body's trailer."""
    unfit = []
    for request in sent:
        for name, value in request.headers.items():
            value = value.decode() if isinstance(value, bytes) else value
            if (
                "checksum" in name.lower()
                or "trailer" in name.lower()
                or (name.lower() == "content-encoding" and "aws-chunked" in value)
            ):
                unfit.append(f"{request.method} {request.url} — {name}: {value}")
    return unfit


class _Body(io.BytesIO):
    def stream(self, **_):
        yield self.getvalue()


def google_answers(request, **_):
    """Google's half of the conversation, without the network: just enough of each answer for
    boto3 to carry on to the next request."""
    query = urllib.parse.urlsplit(request.url).query
    headers, body, status = {"ETag": '"etag"'}, b"", 200
    if request.method == "POST" and query == "uploads":
        body = (
            b"<InitiateMultipartUploadResult><UploadId>up</UploadId>"
            b"</InitiateMultipartUploadResult>"
        )
    elif request.method == "POST":
        body = b"<CompleteMultipartUploadResult><ETag>etag</ETag></CompleteMultipartUploadResult>"
    elif request.method in ("HEAD", "GET"):
        headers["Content-Length"] = "5"
        body = b"hello" if request.method == "GET" else b""
    elif request.method == "DELETE":
        status = 204
    return AWSResponse(request.url, status, headers, _Body(body))


def test_nothing_the_worker_sends_google_carries_a_checksum(monkeypatch, tmp_path):
    """Every call the worker makes to the bucket, against Google's own endpoint and without leaving
    the machine. Google's address and not MinIO's because the scheme decides where the sum would
    go: over HTTPS boto3 puts it in an `aws-chunked` trailer, over plain HTTP in a header — and
    that trailer is exactly what Google turns an upload away over."""
    monkeypatch.delenv("STORAGE_DRIVER", raising=False)
    monkeypatch.setenv("R2_ENDPOINT", "https://storage.googleapis.com")
    monkeypatch.setenv("R2_BUCKET", "remember")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "GOOG1EXAMPLE")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "not-a-secret")
    s3 = storage.client().s3
    sent = listen(s3)
    s3.meta.events.register("before-send.s3", google_answers)
    large, small = tmp_path / "large.mp4", tmp_path / "small.jpg"
    large.write_bytes(os.urandom(LARGE))
    small.write_bytes(b"hello")

    storage.upload(large, "montages/large.mp4", "video/mp4")
    storage.upload(small, "previews/small.jpg", "image/jpeg")
    assert storage.exists("previews/small.jpg")
    storage.download("previews/small.jpg", tmp_path / "back.jpg")
    storage.free("originals/abandoned", "up")

    calls = {(request.method, urllib.parse.urlsplit(request.url).query) for request in sent}
    assert ("PUT", "uploadId=up&partNumber=2") in calls, "the large file did not go up in parts"
    assert ("DELETE", "uploadId=up") in calls, "the abort was never sent"
    assert unfit_for_google(sent) == []


@pytest.mark.parametrize(
    "status, code, missing",
    [
        (404, "NoSuchKey", True),  # Google, deleting a key with nothing under it
        (404, "NoSuchUpload", True),  # an abort of an upload that is over
        (404, "404", True),  # a HEAD: no body, so the status is all the code there is
        (404, "NoSuchBucket", False),  # a misconfigured deploy, not an object already gone
        (403, "AccessDenied", False),
        (503, "SlowDown", False),
    ],
)
def test_nothing_there_is_a_404_of_any_code_but_a_missing_bucket(status, code, missing):
    answer = {"Error": {"Code": code}, "ResponseMetadata": {"HTTPStatusCode": status}}
    assert storage.missing(ClientError(answer, "DeleteObject")) is missing


@pytest.fixture
def scratch(s3):
    """A prefix of this test's own in the bucket, and every request sent to it meanwhile. Emptied
    afterwards, whatever the test left: an abandoned multipart costs a real bucket money."""
    prefix, sent = f"storage-check/{uuid.uuid4()}/", listen(s3)
    yield prefix, sent
    for upload in s3.list_multipart_uploads(Bucket=storage.client().bucket, Prefix=prefix).get(
        "Uploads", []
    ):
        storage.free(upload["Key"], upload["UploadId"])
    for listed in s3.list_objects_v2(Bucket=storage.client().bucket, Prefix=prefix).get(
        "Contents", []
    ):
        storage.free(listed["Key"])


def test_a_large_file_goes_up_in_parts_and_comes_back_whole(s3, scratch, tmp_path):
    """A montage or a clip's preview is tens of megabytes: up in parts, the size and type read off
    the bucket, down in ranges, and gone when freed — twice, the second time over a key with
    nothing under it, which Google answers with a 404 and S3 with a 204."""
    prefix, sent = scratch
    key, large, back = f"{prefix}large.mp4", tmp_path / "large.mp4", tmp_path / "back.mp4"
    large.write_bytes(os.urandom(LARGE))

    storage.upload(large, key, "video/mp4")
    assert any("partNumber=2" in request.url for request in sent), "it did not go up in parts"
    head = s3.head_object(Bucket=storage.client().bucket, Key=key)
    assert (head["ContentLength"], head["ContentType"]) == (LARGE, "video/mp4")
    storage.download(key, back)
    assert back.read_bytes() == large.read_bytes()

    storage.free(key)
    assert not storage.exists(key)
    storage.free(key)

    assert unfit_for_google(sent) == []


def test_an_upload_nobody_finished_is_aborted_and_one_that_was_finished_is_deleted(s3, scratch):
    """What the hourly sweep does to an upload a day old (`sweeps.sweep_stale_uploads`): the parts
    of one left half-way are released, an upload already aborted is no error, and one the phone
    finished without telling the api — so the abort is refused as over — loses its object."""
    prefix, sent = scratch
    bucket = storage.client().bucket

    abandoned = f"{prefix}abandoned"
    upload_id = s3.create_multipart_upload(Bucket=bucket, Key=abandoned)["UploadId"]
    s3.upload_part(Bucket=bucket, Key=abandoned, UploadId=upload_id, PartNumber=1, Body=b"half")
    storage.free(abandoned, upload_id)
    live = s3.list_multipart_uploads(Bucket=bucket, Prefix=abandoned).get("Uploads", [])
    assert upload_id not in [upload["UploadId"] for upload in live], "the parts are released"
    storage.free(abandoned, upload_id)

    finished = f"{prefix}finished"
    upload_id = s3.create_multipart_upload(Bucket=bucket, Key=finished)["UploadId"]
    part = s3.upload_part(Bucket=bucket, Key=finished, UploadId=upload_id, PartNumber=1, Body=b"a")
    s3.complete_multipart_upload(
        Bucket=bucket,
        Key=finished,
        UploadId=upload_id,
        MultipartUpload={"Parts": [{"PartNumber": 1, "ETag": part["ETag"]}]},
    )
    storage.free(finished, upload_id)
    assert not storage.exists(finished), "the finished upload's object is still there"

    assert unfit_for_google(sent) == []
