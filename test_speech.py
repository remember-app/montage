"""The `media_speech` job on answers recorded by hand — see README.md.

Real everything but the model: a real Postgres, a real bucket, real ffmpeg, and clips ffmpeg draws
on the spot. The model is the one thing that cannot be real — there is no key (decision-10 is
open) — so `ai.from_env` is replaced by a provider whose transport serves a recorded answer out of
`fixtures/`, exactly as `test_ai.py` does it. Nothing here goes to the network.

Warning: these wipe `jobs`, `ai_requests` and `media_speech` of whatever DATABASE_URL points at.
"""

import json

import pytest

import ai
import media
import speech
import worker
from test_ai import FIXTURES, Recorded, body
# The fixtures and the ffmpeg-drawn material of the preview tests, rather than a second copy of
# them: a `media_items` row as the api leaves it is the same row here.
from test_media import add_item, conn, ffmpeg, memory, photo, s3, video  # noqa: F401


@pytest.fixture(autouse=True)
def alone(conn, monkeypatch):
    """A clean slate, and no key on the machine running this: a test that means to see «no cloud
    model» must see it, whatever the environment happens to hold one day."""
    conn.execute("DELETE FROM media_speech")
    conn.execute("DELETE FROM ai_requests")
    for name in ("AI_PROVIDER", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def silent_video(path):
    """Three seconds of picture and nothing else — a clip shot with the microphone off."""
    ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x240:rate=24:duration=3",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path))
    return path


def ready_clip(conn, s3, memory, tmp_path, make=video):
    """A clip as the gallery holds it: the row `ready`, the original in the bucket."""
    item_id, key = add_item(conn, memory, "video", status="ready")
    s3.upload_file(str(make(tmp_path / "original.mp4")), media.bucket(), key)
    return item_id


def listens(monkeypatch, transport):
    """The cloud model, replaced by whoever answers here. `AI` builds its provider from the
    environment, so that is the one thing worth replacing — nothing else changes."""
    monkeypatch.setattr(ai, "from_env", lambda *_, **__: ai.Gemini("test-key", transport=transport))
    return transport


def stored(conn, item_id):
    return conn.execute(
        "SELECT transcript FROM media_speech WHERE media_id = %s", (item_id,)
    ).fetchone()


def speech_job(conn, item_id):
    return conn.execute(
        "SELECT status FROM jobs WHERE type = 'media_speech' AND ref = %s", (item_id,)
    ).fetchall()


def test_a_clip_is_transcribed_and_its_words_are_kept(conn, s3, memory, tmp_path, monkeypatch):
    item_id = ready_clip(conn, s3, memory, tmp_path)
    net = listens(monkeypatch, Recorded("gemini_transcribe.json"))
    worker.enqueue(conn, "media_speech", item_id)

    assert worker.run_once(conn) is True

    # Word for word, and nothing else: no speaker, no timing, no «говорит человек-1» (decision-11).
    assert stored(conn, item_id) == ("Ну что, с днём рождения! Загадывай желание и задувай.",)
    # The sound went out, not the picture — and as the audio file the gateway names it by.
    assert body(net.sent)["input"][1]["mime_type"] == speech.AUDIO_MIME
    # And the journal holds the call, under the clip it was about.
    assert conn.execute(
        "SELECT operation, status FROM ai_requests WHERE ref = %s", (item_id,)
    ).fetchone() == ("transcribe_speech", "answered")


def test_the_second_run_takes_what_is_stored_and_asks_nobody(
    conn, s3, memory, tmp_path, monkeypatch
):
    """Two montages out of the same clip, one transcription — «второй ролик берёт сохранённое»."""
    item_id = ready_clip(conn, s3, memory, tmp_path)
    net = listens(monkeypatch, Recorded("gemini_transcribe.json"))

    for _ in range(2):
        worker.enqueue(conn, "media_speech", item_id)
        assert worker.run_once(conn) is True

    assert len(net.requests) == 1, "the stored words answered the second time"
    assert conn.execute("SELECT count(*) FROM ai_requests").fetchone()[0] == 1
    assert speech_job(conn, item_id) == [("done",), ("done",)]


def test_a_clip_with_no_sound_is_neither_invented_nor_failed(
    conn, s3, memory, tmp_path, monkeypatch
):
    item_id = ready_clip(conn, s3, memory, tmp_path, make=silent_video)
    net = listens(monkeypatch, Recorded("gemini_transcribe.json"))
    worker.enqueue(conn, "media_speech", item_id)

    assert worker.run_once(conn) is True

    assert net.requests == [], "there is nothing to send, so nobody was asked"
    assert stored(conn, item_id) == ("",), "silence is an answer and is kept as one"
    assert speech_job(conn, item_id) == [("done",)]


def test_a_clip_where_nobody_speaks_is_not_asked_about_twice(
    conn, s3, memory, tmp_path, monkeypatch
):
    """Sound, but no speech in it: the model answers with nothing, and that nothing is the answer."""
    asked = []

    def heard_nothing(request):
        asked.append(request)
        return json.dumps({"output_text": "\n"}).encode()

    item_id = ready_clip(conn, s3, memory, tmp_path)
    listens(monkeypatch, heard_nothing)

    for _ in range(2):
        worker.enqueue(conn, "media_speech", item_id)
        assert worker.run_once(conn) is True

    assert stored(conn, item_id) == ("",)
    assert len(asked) == 1, "an empty transcript is a settled clip, not an unanswered one"


def test_without_a_cloud_model_nothing_is_written_and_nothing_breaks(conn, s3, memory, tmp_path):
    """Today's state: a worker with no key. The clip is left untranscribed rather than remembered
    as silent, so it is really transcribed the day a key arrives."""
    item_id = ready_clip(conn, s3, memory, tmp_path)
    worker.enqueue(conn, "media_speech", item_id)

    assert worker.run_once(conn) is True

    assert stored(conn, item_id) is None
    assert speech_job(conn, item_id) == [("done",)], "nothing to ask is not a failure"


def test_a_clip_that_went_away_while_the_model_listened_leaves_no_words_behind(
    conn, s3, memory, tmp_path, monkeypatch
):
    """A member left mid-call and took their material with them. What they said does not stay in a
    table nobody will ever look at again."""
    item_id = ready_clip(conn, s3, memory, tmp_path)

    def answer_after_the_member_left(request):
        conn.execute("DELETE FROM media_items WHERE id = %s", (item_id,))
        return (FIXTURES / "gemini_transcribe.json").read_bytes()

    listens(monkeypatch, answer_after_the_member_left)
    worker.enqueue(conn, "media_speech", item_id)

    assert worker.run_once(conn) is True

    assert stored(conn, item_id) is None


@pytest.mark.parametrize("kind, queued", [("video", 1), ("photo", 0)])
def test_a_video_that_became_ready_queues_its_own_transcription(
    conn, s3, memory, tmp_path, kind, queued
):
    """The trigger: a preview finishing is what puts the speech of a clip in the queue. A photo has
    no speech and gets no job."""
    item_id, key = add_item(conn, memory, kind)
    original = (video(tmp_path / "original.mp4") if kind == "video"
                else photo(tmp_path / "original.jpg"))
    s3.upload_file(str(original), media.bucket(), key)
    worker.enqueue(conn, "media_preview", item_id)

    assert worker.run_once(conn) is True

    assert len(speech_job(conn, item_id)) == queued
