"""Читалка HEIC и EXIF в отдельном процессе: что бывает, когда она падает или виснет.

`photo.py` запускает libheif отдельным процессом именно потому, что тот падает на битом файле —
здесь проверяется, что воркер это переживает. См. `docs/jobs/frame-info.md`.
"""

import pytest

from montage import photo, storage, worker

from material import FIXTURES
from rows import add_item, read_item


@pytest.fixture
def reader_script(tmp_path, monkeypatch):
    """Stands in for `photo.py` as the child: whatever the test writes is what runs."""
    script = tmp_path / "reader.py"
    monkeypatch.setattr(photo, "__file__", str(script))
    return script


def test_a_reader_that_dies_is_a_photo_with_nothing_to_read(reader_script):
    reader_script.write_text("import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)\n")

    assert photo.read(FIXTURES / "exif-photo.jpg") is None


def test_a_reader_that_hangs_is_given_up_on(reader_script, monkeypatch):
    reader_script.write_text("import time\ntime.sleep(60)\n")
    monkeypatch.setattr(photo, "TIMEOUT_SECONDS", 1)

    assert photo.read(FIXTURES / "exif-photo.jpg") is None


def test_a_preview_is_made_all_the_same_when_the_reader_dies(conn, s3, memory, reader_script):
    """Standing a HEIC up is a nicety of the copies, not a condition of having them: with the reader
    gone, ffmpeg gets the original as it did before there was one."""
    reader_script.write_text("import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)\n")
    item_id, key = add_item(conn, memory, "photo")
    s3.upload_file(str(FIXTURES / "portrait-photo.heic"), storage.client().bucket, key)
    worker.enqueue(conn, "media_preview", item_id)

    assert worker.run_once(conn) is True

    assert read_item(conn, item_id)[0] == "ready"


def test_the_reader_leaves_alone_what_is_not_a_heic(tmp_path):
    out = tmp_path / "upright.jpg"

    assert photo.stand_up(FIXTURES / "exif-photo.jpg", out) is False
    assert not out.exists()
