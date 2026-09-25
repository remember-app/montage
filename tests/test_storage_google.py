"""Драйвер google: Google Cloud Storage по JSON-ключу служебной учётки — токен, XML API, префикс.

Все тесты, кроме последнего, в сеть не ходят: Google подменён на месте (фикстура `google` в
conftest.py), и видно каждый запрос, который ушёл бы, и что ему ответили. Закреплено:

- адрес, метод и `Authorization: Bearer` каждой операции; загрузка — один PUT с телом прямо с
  диска, скачивание — потоком;
- «уже нет» (404) — спокойно, как у драйвера s3; 5xx, 429 и оборванная связь — повтор, несколько
  раз и не больше; остальные отказы — сразу;
- токен обновляется, когда устарел, и один раз — когда Google его не принял;
- ни одна операция не выходит за префикс: ключ, который вывел бы за него, отказан до запроса;
- без префикса и с битым ключом драйвер не создаётся, а ключ и токен не попадают ни в ошибку, ни в
  лог.

Последний тест — живой контроль (docs/running.md): настоящий бакет Google, только под префиксом
проверок `remember/dev-check/<uuid>/`, и за собой убирает. Без STORAGE_DRIVER=google в окружении
он пропускается.
"""

import io
import json
import logging
import os
import re
import traceback
import uuid

import pytest
import requests

from montage import storage

MIB = 1024 * 1024


def written(path, data=b"the bytes of a frame"):
    path.write_bytes(data)
    return path


def test_every_operation_is_one_request_for_the_object_under_the_prefix(google, tmp_path):
    photo = written(tmp_path / "preview.jpg", os.urandom(3 * MIB))
    back = tmp_path / "back"
    google.answers = [200, (200, b"the original"), 200, 204, 204]

    storage.upload(photo, "previews/a.jpg", "image/jpeg")
    storage.download("originals/a", back)
    assert storage.exists("originals/a")
    storage.delete("thumbnails/a.jpg")
    storage.abort("originals/b", "upload-1")

    objects = google.objects
    assert [(sent.method, sent.url) for sent in google.sent] == [
        ("PUT", objects + "previews/a.jpg"),
        ("GET", objects + "originals/a"),
        ("HEAD", objects + "originals/a"),
        ("DELETE", objects + "thumbnails/a.jpg"),
        ("DELETE", objects + "originals/b?uploadId=upload-1"),
    ]
    assert {sent.headers["Authorization"] for sent in google.sent} == {"Bearer token-1"}
    assert google.tokens == 1, "one token for all of them, not one per request"
    assert {sent.timeout for sent in google.sent} == {storage.TIMEOUT}

    put, get = google.sent[:2]
    assert (put.headers["Content-Type"], put.headers["Content-Length"]) == (
        "image/jpeg",
        str(3 * MIB),
    )
    assert put.from_disk, "the file went up as the open file, not read into memory first"
    assert put.body == photo.read_bytes(), "and whole, in one PUT"
    assert get.stream, "a download is streamed into its file, not held in memory"
    assert back.read_bytes() == b"the original"
    assert str(storage.client()) == f"google {objects}"


def test_a_key_is_sent_as_the_name_it_is_and_nothing_is_decoded_out_of_it(google):
    storage.delete("originals/a b?c#d%2e%2e")

    assert google.sent[0].url == google.objects + "originals/a%20b%3Fc%23d%252e%252e"


def test_a_prefix_written_with_a_slash_at_the_end_is_the_same_prefix(google, monkeypatch):
    monkeypatch.setenv("GOOGLE_STORAGE_PREFIX", "remember/dev-media/")

    storage.delete("originals/a")

    assert google.sent[0].url == google.objects + "originals/a"


def test_nothing_there_is_an_answer_and_a_missing_bucket_is_not(google, tmp_path):
    """The sweep frees what an upload left: Google answers an abort of an upload that is over, and
    a delete of a key with nothing under it, with a 404 — the outcome wanted, as in the s3 driver
    (`storage.missing`). A bucket that is not there is a 404 too, and that one is loud."""
    google.answers = [(404, "NoSuchUpload"), (404, "NoSuchKey"), 404, (404, "NoSuchKey")]

    storage.free("originals/a", "upload-1")
    assert not storage.exists("originals/a"), (
        "a HEAD's 404 has no body, and is «not there» all the same"
    )
    with pytest.raises(storage.GoogleError) as gone:
        storage.download("music/a", tmp_path / "track")
    assert storage.missing(gone.value) and gone.value.code == "NoSuchKey"

    for refusal in [(404, "NoSuchBucket"), (403, "AccessDenied")]:
        google.answers = [refusal]
        with pytest.raises(storage.GoogleError) as refused:
            storage.delete("originals/a")
        assert not storage.missing(refused.value)
        assert refusal[1] in str(refused.value)

    assert len(google.sent) == 6, "none of them was asked twice"


def test_a_5xx_a_429_or_a_dropped_connection_is_tried_again_and_the_file_goes_whole(
    google, tmp_path
):
    clip = written(tmp_path / "montage.mp4", os.urandom(MIB))
    google.answers = [503, 429, requests.ConnectionError("connection reset"), 200]

    storage.upload(clip, "montages/a.mp4", "video/mp4")

    assert [sent.body for sent in google.sent] == [clip.read_bytes()] * 4


def test_a_download_broken_off_half_way_is_fetched_again_from_its_first_byte(google, tmp_path):
    class BrokenOff(io.BytesIO):
        def read(self, *args):
            if self.tell():
                raise requests.exceptions.ChunkedEncodingError("connection broken")
            return super().read(*args)

    original = tmp_path / "original"
    google.answers = [(200, BrokenOff(b"the first half")), (200, b"the whole original")]

    storage.download("originals/a", original)

    assert original.read_bytes() == b"the whole original"


def test_google_that_stays_down_is_given_up_on_after_a_few_attempts(google, tmp_path):
    google.answers = [500] * storage.ATTEMPTS + [200]

    with pytest.raises(storage.GoogleError) as down:
        storage.download("originals/a", tmp_path / "original")

    assert down.value.status == 500
    assert len(google.sent) == storage.ATTEMPTS


@pytest.mark.parametrize("refusal", [400, 403, (412, "PreconditionFailed")])
def test_a_request_google_refused_is_not_asked_again(google, refusal):
    google.answers = [refusal]

    with pytest.raises(storage.GoogleError):
        storage.delete("originals/a")

    assert len(google.sent) == 1


def test_a_token_that_ran_out_is_renewed_before_the_request(google):
    # Shorter than the margin google-auth keeps: stale by the time the next request is made.
    google.lifetime = 60

    storage.delete("originals/a")
    storage.delete("originals/b")

    assert [sent.headers["Authorization"] for sent in google.sent] == [
        "Bearer token-1",
        "Bearer token-2",
    ]


def test_a_token_google_turns_down_is_renewed_once_and_the_file_goes_again_whole(google, tmp_path):
    clip = written(tmp_path / "montage.mp4", os.urandom(MIB))
    google.answers = [401, 200]

    storage.upload(clip, "montages/a.mp4", "video/mp4")

    assert [(sent.headers["Authorization"], sent.body) for sent in google.sent] == [
        ("Bearer token-1", clip.read_bytes()),
        ("Bearer token-2", clip.read_bytes()),
    ]

    google.sent.clear()
    google.answers = [401, 401]
    with pytest.raises(storage.GoogleError) as refused:
        storage.delete("originals/a")
    assert refused.value.status == 401
    assert len(google.sent) == 2, "once, not for as long as Google keeps saying no"


def test_neither_the_key_nor_a_token_is_in_what_a_failure_says(
    google, service_key, tmp_path, caplog
):
    """A failure is read in the log and in `jobs.last_error`: what was asked, of which object, and
    what Google said — never the credentials it was asked with."""
    google.answers = [503] * storage.ATTEMPTS

    with caplog.at_level(logging.DEBUG), pytest.raises(storage.GoogleError) as down:
        storage.upload(written(tmp_path / "poster.jpg"), "montages/a.jpg", "image/jpeg")

    told = caplog.text + "".join(traceback.format_exception(down.value))
    assert "montages/a.jpg" in told and "503" in told
    assert "token-" not in told and "Bearer" not in told
    assert json.loads(service_key)["private_key"].splitlines()[1] not in told


OUTSIDE = [
    "",
    "/originals/a",
    "originals/../../somebody-else/a",
    "..",
    "originals/./a",
    "originals//a",
    "originals/a/",
    "remember/dev-check/a",
    "remember/dev-media/originals/a",
    "originals/a\n",
    "originals/\x00a",
    "originals/\x7fa",
    None,
]

OPERATIONS = {
    "download": lambda key, tmp: storage.download(key, tmp / "file"),
    "upload": lambda key, tmp: storage.upload(written(tmp / "file"), key, "image/jpeg"),
    "exists": lambda key, tmp: storage.exists(key),
    "delete": lambda key, tmp: storage.delete(key),
    "abort": lambda key, tmp: storage.abort(key, "upload-1"),
    "free": lambda key, tmp: storage.free(key, "upload-1"),
}


@pytest.mark.parametrize("key", OUTSIDE)
@pytest.mark.parametrize("operation", OPERATIONS)
def test_a_key_that_would_step_outside_the_prefix_is_refused_before_anything_is_sent(
    google, tmp_path, operation, key
):
    """The bucket is the company's and what lies beside our prefix is somebody else's: a key that
    is empty, climbs out with `..`, starts at the root, hides a character or carries a prefix of its
    own is refused before Google is asked anything — not a token, not a request."""
    with pytest.raises(ValueError, match="not a key under the storage prefix"):
        OPERATIONS[operation](key, tmp_path)

    assert (google.sent, google.tokens) == ([], 0)


@pytest.mark.parametrize(
    "prefix", ["", "  ", "/", "/remember/dev-media", "../remember", "remember//dev-media"]
)
def test_the_driver_is_not_built_without_a_prefix_of_its_own(google, monkeypatch, prefix):
    monkeypatch.setenv("GOOGLE_STORAGE_PREFIX", prefix)

    with pytest.raises(ValueError, match="GOOGLE_STORAGE_PREFIX"):
        storage.client()


@pytest.mark.parametrize(
    "setting", ["GOOGLE_STORAGE_CREDENTIALS", "GOOGLE_STORAGE_BUCKET", "GOOGLE_STORAGE_PREFIX"]
)
def test_a_setting_that_is_not_there_is_named(google, monkeypatch, setting):
    monkeypatch.delenv(setting)

    with pytest.raises(KeyError, match=setting):
        storage.client()


def test_a_driver_nobody_has_is_named_too(monkeypatch):
    monkeypatch.setenv("STORAGE_DRIVER", "gcs")

    with pytest.raises(ValueError, match="gcs"):
        storage.client()


def _mangled(key):
    info = json.loads(key)
    lines = info["private_key"].splitlines()
    info["private_key"] = "\n".join(lines[:2] + [line[::-1] for line in lines[2:-1]] + lines[-1:])
    return json.dumps(info)


def _anonymous(key):
    info = json.loads(key)
    del info["client_email"]
    return json.dumps(info)


@pytest.mark.parametrize(
    "broken",
    [
        lambda key: "",
        lambda key: "not json",
        lambda key: "{}",
        lambda key: '"a key"',
        _mangled,
        _anonymous,
    ],
    ids=["empty", "not json", "empty object", "a string", "mangled private key", "no account"],
)
def test_a_key_that_is_not_one_stops_the_driver_and_says_nothing_of_it(
    google, monkeypatch, service_key, broken
):
    """Loud where the driver is built — at the worker's start — and not at the first upload. And
    silent about the key itself: what the parser said about it is not passed on either."""
    monkeypatch.setenv("GOOGLE_STORAGE_CREDENTIALS", broken(service_key))

    with pytest.raises(ValueError, match="GOOGLE_STORAGE_CREDENTIALS") as refused:
        storage.client()

    told = "".join(traceback.format_exception(refused.value))
    pem = json.loads(service_key)["private_key"].splitlines()
    assert not [line for line in pem[1:-1] if line in told or line[::-1] in told]
    assert (google.sent, google.tokens) == ([], 0)


# --- the live control --------------------------------------------------------------------------

LARGE = 9 * MIB  # past the 8 MiB the s3 driver cuts into parts: here it still goes up in one PUT
CHECK_PREFIX = "remember/dev-check"  # decision-16: checks live here and nowhere else


@pytest.fixture
def check_bucket(monkeypatch):
    """The real bucket from STORAGE_DRIVER=google and GOOGLE_STORAGE_*, under a prefix of this run's
    own inside `remember/dev-check` — whatever GOOGLE_STORAGE_PREFIX says, so the control can never
    write where the working material lies. Whatever it made is taken back out afterwards, even when
    it failed; every address it sent a request to is kept for the test to look at."""
    if os.environ.get("STORAGE_DRIVER") != "google":
        pytest.skip("the live control: STORAGE_DRIVER=google and GOOGLE_STORAGE_* (running.md)")
    monkeypatch.setenv("GOOGLE_STORAGE_PREFIX", f"{CHECK_PREFIX}/{uuid.uuid4()}")
    store, made, seen = storage.client(), [], []
    store._session.hooks["response"].append(lambda answer, **_: seen.append(answer.request.url))
    yield store, made, seen
    left = []
    for key, upload_id in made:
        try:
            storage.free(key, upload_id)
        except Exception as exc:
            left.append(f"{key}: {exc}")
    assert left == [], "left behind in the check bucket"


def _start(store, key):
    started = store._send("POST", key, params="uploads")
    return re.search(r"<UploadId>(.+?)</UploadId>", started.text).group(1)


def _part(store, key, upload_id, tmp_path):
    part = written(tmp_path / f"part-{uuid.uuid4()}", b"part of a phone's upload")
    sent = store._send("PUT", key, params={"partNumber": 1, "uploadId": upload_id}, body=part)
    return sent.headers["ETag"]


def test_the_live_google_bucket_under_the_check_prefix(check_bucket, tmp_path):
    """What the worker does to Google, done to Google: a 9 MiB file up in one PUT, its size and type
    read back, down again byte for byte, deleted — and deleted again over a key with nothing under
    it. An upload the api started and nobody finished aborted, aborted again, and refused a part
    afterwards; one the phone finished without telling the api deleted by the same free."""
    store, made, seen = check_bucket
    large, back = written(tmp_path / "large.mp4", os.urandom(LARGE)), tmp_path / "back.mp4"

    key = f"montages/{uuid.uuid4()}.mp4"
    made.append((key, None))
    storage.upload(large, key, "video/mp4")
    head = store._send("HEAD", key)
    assert (head.headers["Content-Length"], head.headers["Content-Type"]) == (
        str(LARGE),
        "video/mp4",
    )
    assert storage.exists(key)
    storage.download(key, back)
    assert back.read_bytes() == large.read_bytes()
    storage.delete(key)
    assert not storage.exists(key)
    storage.delete(key)

    abandoned = f"originals/{uuid.uuid4()}"
    upload_id = _start(store, abandoned)
    made.append((abandoned, upload_id))
    _part(store, abandoned, upload_id, tmp_path)
    storage.free(abandoned, upload_id)
    storage.free(abandoned, upload_id)
    with pytest.raises(storage.GoogleError) as over:
        _part(store, abandoned, upload_id, tmp_path)
    assert over.value.code == "NoSuchUpload", "the parts are released"

    finished = f"originals/{uuid.uuid4()}"
    upload_id = _start(store, finished)
    made.append((finished, upload_id))
    etag = _part(store, finished, upload_id, tmp_path)
    parts = written(
        tmp_path / "complete.xml",
        f"<CompleteMultipartUpload><Part><PartNumber>1</PartNumber><ETag>{etag}</ETag></Part>"
        "</CompleteMultipartUpload>".encode(),
    )
    store._send("POST", finished, params={"uploadId": upload_id}, body=parts)
    assert storage.exists(finished)
    storage.free(finished, upload_id)
    assert not storage.exists(finished), "the finished upload's object is still there"

    under = f"{storage.GOOGLE}/{os.environ['GOOGLE_STORAGE_BUCKET'].strip()}/{store.prefix}/"
    assert store.prefix.startswith(f"{CHECK_PREFIX}/")
    assert seen and [url for url in seen if not url.startswith(under)] == []
