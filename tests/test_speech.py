"""The `media_speech` job on answers recorded by hand — see README.md.

Real everything but the model: a real Postgres, a real bucket, real ffmpeg, and clips ffmpeg draws
on the spot. The model is the one thing that cannot be real — there is no key (decision-10 is
open) — so `ai.from_env` is replaced by a provider whose transport serves a recorded answer out of
`fixtures/`, exactly as `test_ai.py` does it. Nothing here goes to the network.

Warning: these wipe `jobs`, `ai_requests` and `media_speech` of whatever DATABASE_URL points at.
"""

import json

import pytest

from montage import ai, storage, worker
from montage.ai import providers
from montage.media import speech
from montage.video import montage as montages
from montage.video import selection

from material import FIXTURES, draw, jpeg, video
from recorded import Recorded, body, saying
from rows import add_item


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
    draw(
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=320x240:rate=24:duration=3",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    )
    return path


def ready_clip(conn, s3, memory, tmp_path, make=video):
    """A clip as the gallery holds it: the row `ready`, the original in the bucket."""
    item_id, key = add_item(conn, memory, "video", status="ready")
    s3.upload_file(str(make(tmp_path / "original.mp4")), storage.client().bucket, key)
    return item_id


def listens(monkeypatch, transport):
    """The cloud model, replaced by whoever answers here. `AI` builds its provider from the
    environment, so that is the one thing worth replacing — nothing else changes."""
    monkeypatch.setattr(
        ai, "from_env", lambda *_, **__: providers.Gemini("test-key", transport=transport)
    )
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
    """Sound, but no speech in it: the model answers with nothing, and that nothing is the
    answer."""
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


def test_a_character_no_table_can_hold_is_dropped_from_the_words_and_they_are_kept(
    conn, s3, memory, tmp_path, monkeypatch
):
    """A transcript is plain text and never goes through the model's JSON, but it goes into the
    journal's `jsonb` and into `media_speech.transcript` all the same — and neither takes a NUL or
    half of a surrogate pair. Both go; the words, which are paid for, stay."""
    item_id = ready_clip(conn, s3, memory, tmp_path)
    listens(monkeypatch, saying("Ну что\x00, с днём\ud800 рождения!"))
    worker.enqueue(conn, "media_speech", item_id)

    assert worker.run_once(conn) is True

    assert stored(conn, item_id) == ("Ну что, с днём рождения!",)
    assert speech_job(conn, item_id) == [("done",)]
    assert conn.execute(
        "SELECT status, response FROM ai_requests WHERE ref = %s", (item_id,)
    ).fetchone() == ("answered", "Ну что, с днём рождения!")


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


def test_an_order_transcribes_a_clip_nobody_wrote_the_words_of(
    conn, s3, memory, tmp_path, monkeypatch
):
    """The montage's analysis stage is where a clip's words are asked for now: an order that found
    no transcript gets one, with nothing pressed and nothing listened to at upload (m-43)."""
    item_id = ready_clip(conn, s3, memory, tmp_path)
    listens(monkeypatch, Recorded("gemini_transcribe.json"))
    assert stored(conn, item_id) is None

    montages.analysis(conn, str(memory[0]), [selection.Piece(media_id=item_id, kind="video")])

    assert stored(conn, item_id) is not None


def test_an_order_transcribes_the_clip_preparation_already_downloaded(
    conn, s3, memory, tmp_path, monkeypatch
):
    """The order's preparation downloads every clip; the analysis stage hands the file it already
    has to the transcriber, so the same bytes do not come out of the bucket twice (m-43)."""
    on_disk = video(tmp_path / "on-disk.mp4")
    item_id, _ = add_item(conn, memory, "video", status="ready")  # nothing in the bucket at all
    listens(monkeypatch, Recorded("gemini_transcribe.json"))

    montages.analysis(
        conn,
        str(memory[0]),
        [selection.Piece(media_id=item_id, kind="video")],
        judged=[],  # preparation looked at everything; this test is about the words alone
        originals={item_id: on_disk},
    )

    assert stored(conn, item_id) is not None


def test_an_order_does_not_transcribe_a_clip_preparation_found_unreadable(
    conn, memory, monkeypatch
):
    """The отсев struck the file out — the same bytes would fail the transcriber too, and inline in
    the order that failure takes the whole film with it. One bad file costs the film that file's
    words, not the film (m-43)."""
    item_id, _ = add_item(conn, memory, "video", status="ready")  # nothing in the bucket at all
    listens(monkeypatch, Recorded("gemini_transcribe.json"))  # nobody to ask is not the point

    montages.analysis(
        conn,
        str(memory[0]),
        [selection.Piece(media_id=item_id, kind="video", rejected=selection.UNREADABLE)],
        judged=[],  # preparation looked at everything, and struck this one out
    )

    assert stored(conn, item_id) is None


@pytest.mark.parametrize("kind", ["video", "photo"])
def test_a_preview_queues_no_transcription(conn, s3, memory, tmp_path, kind):
    """An upload transcribes nothing (m-43, decision-22): the words are asked for by the analysis
    stage of an order, and a photo has none at all."""
    item_id, key = add_item(conn, memory, kind)
    original = (
        video(tmp_path / "original.mp4") if kind == "video" else jpeg(tmp_path / "original.jpg")
    )
    s3.upload_file(str(original), storage.client().bucket, key)
    worker.enqueue(conn, "media_preview", item_id)

    assert worker.run_once(conn) is True

    assert speech_job(conn, item_id) == []
