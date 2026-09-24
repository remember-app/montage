"""Both providers and the gateway, on answers recorded by hand — see README.md.

Nothing here goes to the network, and nothing here can: every provider is built with a transport
that serves a file out of `fixtures/` and keeps the request it was handed, so a test looks at what
*would* have gone out. There is no key to go out with anyway (decision-10 is open) — the recorded
answers are the shapes the two APIs document, filled with made-up material.

The one exception is the redirect test, which is about the door itself rather than about a provider:
it opens a server on this machine's loopback and points `ai._urlopen` at that.

The gateway half needs a real Postgres, like the rest of the suite: the journal it writes is the
point of it, and a fake table would prove nothing.

Warning: these wipe the `ai_requests` table of whatever DATABASE_URL points at.
"""

import base64
import http.server
import json
import logging
import pathlib
import threading
import urllib.error
import urllib.request

import pytest

import ai
import worker

FIXTURES = pathlib.Path(__file__).with_name("fixtures")

PHOTO = ai.Blob("image/jpeg", b"\xff\xd8\xff\xe0 not really a photo")
CLIP = ai.Blob("video/mp4", b"\x00\x00\x00\x18ftypmp42 not really a clip")
SPEECH = ai.Blob("audio/m4a", b"not really the sound of a clip")


class Recorded:
    """The network, replaced by one answer somebody wrote down. Keeps every request it was given."""

    def __init__(self, fixture):
        self.answer = (FIXTURES / fixture).read_bytes()
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return self.answer

    @property
    def sent(self):
        """The one request that went out, or a failure saying how many did."""
        assert len(self.requests) == 1, f"{len(self.requests)} requests, expected exactly one"
        return self.requests[0]


def body(request):
    return json.loads(request.data)


# --- the providers ----------------------------------------------------------
# No database here: an adapter is a request built and an answer read.


def test_gemini_attaches_the_photo_and_asks_for_json():
    net = Recorded("gemini_describe.json")

    answered = ai.Gemini("test-key", transport=net).generate("инструкция", "вопрос", [PHOTO])

    assert json.loads(answered)["cast"] == ["человек-1", "животное-1"]
    sent = net.sent
    assert sent.full_url == ai.Gemini.ENDPOINT
    assert sent.get_header("X-goog-api-key") == "test-key"
    asked = body(sent)
    assert asked["model"] == ai.Gemini.DEFAULT_MODEL
    assert asked["input"][0]["text"] == "инструкция\n\nвопрос"
    assert asked["input"][1] == {
        "type": "image",
        "mime_type": "image/jpeg",
        "data": base64.b64encode(PHOTO.data).decode(),
    }
    assert asked["response_format"]["mime_type"] == "application/json"


def test_gemini_takes_a_clip_whole():
    """The reason Gemini is the cheap one: a video is an input, not a pile of frames (doc-11)."""
    net = Recorded("gemini_describe.json")

    ai.Gemini("test-key", transport=net).generate("инструкция", "вопрос", [CLIP])

    assert body(net.sent)["input"][1]["type"] == "video"


def test_gemini_reads_the_answer_out_of_the_steps_when_there_is_no_shortcut():
    """`output_text` is a convenience the interaction may or may not carry; the model's own step
    always does."""
    net = Recorded("gemini_transcribe.json")

    assert "output_text" not in json.loads(net.answer)
    assert ai.Gemini("test-key", transport=net).transcribe(SPEECH).startswith("Ну что")
    assert body(net.sent)["input"][1]["mime_type"] == "audio/m4a"


def test_openai_sends_the_photo_as_a_data_uri_and_the_instruction_apart():
    net = Recorded("openai_describe.json")

    answered = ai.OpenAI("test-key", transport=net).generate("инструкция", "вопрос", [PHOTO])

    assert json.loads(answered)["cast"] == ["человек-1", "животное-1"]
    sent = net.sent
    assert sent.full_url == ai.OpenAI.ENDPOINT
    assert sent.get_header("Authorization") == "Bearer test-key"
    asked = body(sent)
    assert asked["instructions"] == "инструкция"
    content = asked["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "вопрос"}
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,"), "a bare base64 is dropped"
    assert asked["text"]["format"] == {"type": "json_object"}
    # And the message was found past the reasoning item the recorded response opens with, which is
    # where a response usually keeps it.
    assert json.loads(net.answer)["output"][0]["type"] == "reasoning"


def test_openai_will_not_be_handed_a_clip():
    """OpenAI takes no video at all — the caller has to cut frames, and has to be told so."""
    with pytest.raises(ValueError, match="video"):
        ai.OpenAI("test-key", transport=Recorded("openai_describe.json")).generate("и", "в", [CLIP])


def test_openai_uploads_speech_as_a_file():
    net = Recorded("openai_transcribe.json")

    assert ai.OpenAI("test-key", transport=net).transcribe(SPEECH).startswith("Ну что")

    sent = net.sent
    assert sent.full_url == ai.OpenAI.TRANSCRIPTIONS
    assert sent.get_header("Content-type").startswith("multipart/form-data; boundary=")
    assert b'name="model"' in sent.data
    assert b'filename="speech.m4a"' in sent.data, "the decoder is picked by the name"
    assert SPEECH.data in sent.data


# --- the one thing that does reach a socket ---------------------------------


@pytest.fixture
def redirecting():
    """A server on this machine that answers a POST with «go and ask over there», and records what
    actually turns up over there. The only socket in this file, and it never leaves the loopback."""
    followed = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{self.server.server_port}/elsewhere")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            followed.append(dict(self.headers))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *_):
            pass  # the test's own output is the point, not an access log

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/", followed
    server.shutdown()
    server.server_close()


def test_a_redirect_is_refused_rather_than_followed_with_the_key_on_it(redirecting):
    """The key leaving for a host nobody chose. urllib's own redirect handler copies every header
    but `content-*` onto the next request, so one 30x — from either endpoint, from anything that can
    answer as one — would hand `Authorization` straight to wherever it points. `ai._urlopen` does
    not follow redirects at all; the 30x comes back as the failure it is."""
    where, followed = redirecting
    request = urllib.request.Request(
        where, data=b"{}", method="POST",
        headers={"Authorization": "Bearer test-key", "Content-Type": "application/json"},
    )

    with pytest.raises(urllib.error.HTTPError) as refused:
        ai._urlopen(request)

    assert refused.value.code == 302
    assert followed == [], "nothing went to where it pointed, and the key least of all"


# --- which provider ---------------------------------------------------------


def test_the_setting_alone_decides_who_answers(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert isinstance(ai.from_env(), ai.OpenAI)

    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-pro")
    chosen = ai.from_env()
    assert isinstance(chosen, ai.Gemini) and chosen.model == "gemini-3.8-pro"


def test_a_provider_nobody_has_is_nobody(monkeypatch, caplog):
    """A typo in the setting must read as «no model», not as a model: the worker keeps working."""
    monkeypatch.setenv("AI_PROVIDER", "skynet")
    monkeypatch.setenv("SKYNET_API_KEY", "test-key")

    with caplog.at_level(logging.ERROR):
        assert ai.from_env() is None
    assert "skynet" in caplog.text


# --- the gateway and its journal --------------------------------------------


@pytest.fixture
def conn():
    with worker.connect() as connection:
        worker.init_schema(connection)
        connection.execute("DELETE FROM ai_requests")
        yield connection


@pytest.fixture(autouse=True)
def no_provider_in_the_environment(monkeypatch):
    """The machine running the tests may have a key on it one day. The gateway is still given its
    provider by hand here, and the «no model» tests must see none."""
    for name in ("AI_PROVIDER", "GEMINI_API_KEY", "OPENAI_API_KEY", "OPENAI_TRANSCRIBE_MODEL"):
        monkeypatch.delenv(name, raising=False)


def journal(conn):
    return conn.execute(
        """SELECT provider, model, operation, ref, status, request, response, error, took_ms
           FROM ai_requests ORDER BY id"""
    ).fetchall()


def test_a_call_and_its_answer_are_both_in_the_journal(conn):
    net = Recorded("gemini_describe.json")
    gateway = ai.AI(conn, ai.Gemini("test-key", transport=net))

    described = gateway.describe_media([PHOTO], cast=["человек-1"], ref="media-7")

    assert described["description"].startswith("Женщина")
    (provider, model, operation, ref, status, request, response, error, took_ms), = journal(conn)
    assert (provider, model) == ("gemini", ai.Gemini.DEFAULT_MODEL)
    assert (operation, ref) == ("describe_media", "media-7")
    assert (status, error) == ("answered", None)
    assert response == described, "what the caller got is what the journal says it got"
    assert "человек-1" in request["prompt"], "the memory's cast went into the prompt"
    assert request["attached"] == [{"mime_type": "image/jpeg", "bytes": len(PHOTO.data)}]
    assert took_ms is not None


def test_the_journal_has_the_question_before_the_model_has_answered(conn):
    """Not «written at the end, if all went well». The row is there while the model is still
    thinking, which is what makes it a record of what we asked."""
    seen = []

    def look(request):
        seen.append(conn.execute("SELECT status FROM ai_requests").fetchall())
        return (FIXTURES / "gemini_describe.json").read_bytes()

    ai.AI(conn, ai.Gemini("test-key", transport=look)).describe_media([PHOTO])

    assert seen == [[("sent",)]]


def test_a_call_that_failed_is_in_the_journal_too(conn):
    def refuse(request):
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {}, None)

    gateway = ai.AI(conn, ai.OpenAI("test-key", transport=refuse))
    with pytest.raises(urllib.error.HTTPError):
        gateway.transcribe_speech(SPEECH, ref="media-7")

    (*_, status, request, response, error, _), = journal(conn)
    assert (status, response) == ("failed", None)
    assert "429" in error
    assert request["attached"] == [{"mime_type": "audio/m4a", "bytes": len(SPEECH.data)}]


def test_an_answer_that_is_not_json_fails_the_call_and_says_what_came_instead(conn):
    def prose(request):
        return json.dumps({"output_text": "Конечно! Вот описание кадра:"}).encode()

    gateway = ai.AI(conn, ai.Gemini("test-key", transport=prose))
    with pytest.raises(RuntimeError, match="Конечно"):
        gateway.describe_media([PHOTO])

    (*_, status, _, _, error, _), = journal(conn)
    assert status == "failed" and "Конечно" in error


def test_an_answer_that_parses_but_is_not_an_object_fails_the_call_too(conn):
    """A bare `null` is valid JSON and nothing a caller can read a field off. Failed here, where the
    answer itself is kept — rather than three handlers down as an `AttributeError` about `NoneType`,
    which says nothing about what the model actually said."""
    def nothing(request):
        return json.dumps({"output_text": "null"}).encode()

    gateway = ai.AI(conn, ai.Gemini("test-key", transport=nothing))
    with pytest.raises(RuntimeError, match="null"):
        gateway.describe_media([PHOTO])

    (*_, status, _, _, error, _), = journal(conn)
    assert status == "failed" and "null" in error


def saying(text):
    """A model that says `text` — its words as they come, inside Gemini's envelope."""
    return lambda request: json.dumps({"output_text": text}).encode()


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"])
def test_a_number_the_journal_cannot_hold_is_dropped_and_the_answer_kept(conn, number):
    """`json.loads` reads each of these into a float `jsonb` does not take. The answer is paid for
    by then: refused, or breaking the journal's UPDATE on the way back, it is paid for again on the
    retry. The number goes, the rest of the answer stays."""
    said = f'{{"description": "кадр", "cast": ["человек-1"], "seconds": {number}}}'
    gateway = ai.AI(conn, ai.Gemini("test-key", transport=saying(said)))

    described = gateway.describe_media([PHOTO])

    assert described == {"description": "кадр", "cast": ["человек-1"], "seconds": None}
    (*_, status, _, response, error, _), = journal(conn)
    assert (status, response, error) == ("answered", described, None)


def test_a_character_the_journal_cannot_hold_is_dropped_and_the_answer_kept(conn):
    """A NUL, and half of a surrogate pair: neither `jsonb` nor `text` takes one. Escaped in the
    model's JSON or already a character of the envelope around it, both are gone from what the
    journal and the caller get — and nothing else of the answer is."""
    said = ('{"description": "Торт\\ud800 со свечами\\u0000", '
            '"cast": ["человек-1\udc00", "живот\x00ное-1"]}')
    gateway = ai.AI(conn, ai.Gemini("test-key", transport=saying(said)))

    described = gateway.describe_media([PHOTO])

    assert described == {"description": "Торт со свечами", "cast": ["человек-1", "животное-1"]}
    (*_, status, _, response, error, _), = journal(conn)
    assert (status, response, error) == ("answered", described, None)


def test_prose_carrying_such_a_character_still_fails_into_the_journal(conn):
    """The failed call keeps the text that failed it in `error`, a `text` column: with half a
    surrogate pair in it, the row itself would not close, and the journal would say «sent» for
    good."""
    gateway = ai.AI(conn, ai.Gemini("test-key", transport=saying("Конечно\ud800! Вот описание:")))

    with pytest.raises(RuntimeError, match="Конечно! Вот описание"):
        gateway.describe_media([PHOTO])

    (*_, status, _, _, error, _), = journal(conn)
    assert status == "failed" and "Конечно! Вот описание" in error


def test_the_story_reaches_the_director(conn):
    """Two orders on the same material differ only by what was asked — so it has to be asked."""
    net = Recorded("gemini_plan.json")
    gateway = ai.AI(conn, ai.Gemini("test-key", transport=net))

    planned = gateway.plan_montage(
        [{"media_id": "m-1", "description": "накрывают на стол"}],
        story="день рождения, всё вокруг торта",
        target_seconds=30,
        ref="montage-3",
    )

    assert [piece["media_id"] for piece in planned["plan"]] == ["m-1", "m-2", "m-3"]
    asked = body(net.sent)["input"][0]["text"]
    assert "всё вокруг торта" in asked and "30" in asked
    (*_, operation, ref, status, _, _, _, _), = journal(conn)
    assert (operation, ref, status) == ("plan_montage", "montage-3", "answered")


def test_without_a_key_nothing_is_asked_and_nothing_is_pretended(conn, monkeypatch, caplog):
    """The state the worker is in today: a provider chosen, no key to use it with."""
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    with caplog.at_level(logging.INFO):
        gateway = ai.AI(conn)
        assert gateway.describe_media([PHOTO], ref="media-7") is None

    assert gateway.provider is None, "nothing to call means nothing was built to call with"
    assert "GEMINI_API_KEY" in caplog.text
    assert "skipped" in caplog.text
    (provider, model, operation, ref, status, _, response, _, _), = journal(conn)
    assert (provider, model, status, response) == ("gemini", None, "skipped", None)
    assert (operation, ref) == ("describe_media", "media-7")


def test_the_journal_keeps_neither_the_key_nor_the_material(conn):
    """It outlives every key in it and is read by hand in psql; and a copy of the memory's photos
    in jsonb would make it the biggest table in the database."""
    gateway = ai.AI(conn, ai.Gemini("test-key", transport=Recorded("gemini_describe.json")))

    gateway.describe_media([PHOTO])

    written = json.dumps(journal(conn), ensure_ascii=False)
    assert "test-key" not in written
    assert base64.b64encode(PHOTO.data).decode() not in written
