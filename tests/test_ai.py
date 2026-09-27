"""Both providers and the gateway, on answers recorded by hand — see README.md.

Nothing here goes to the network, and nothing here can: every provider is built with a transport
that serves a file out of `fixtures/` and keeps the request it was handed, so a test looks at what
*would* have gone out. There is no key to go out with anyway (decision-10 is open) — the recorded
answers are the shapes the two APIs document, filled with made-up material.

The one exception is the redirect test, which is about the door itself rather than about a provider:
it opens a server on this machine's loopback and points `providers._urlopen` at that.

The gateway half needs a real Postgres, like the rest of the suite: the journal it writes is the
point of it, and a fake table would prove nothing.

Warning: these wipe the `ai_requests` table of whatever DATABASE_URL points at.
"""

import base64
import http.server
import json
import logging
import sys
import threading
import tracemalloc
import urllib.error
import urllib.request

import pytest

from montage import ai, worker
from montage.ai import answers, providers

from material import FIXTURES
from recorded import Recorded, body, saying

PHOTO = ai.Blob("image/jpeg", b"\xff\xd8\xff\xe0 not really a photo")
CLIP = ai.Blob("video/mp4", b"\x00\x00\x00\x18ftypmp42 not really a clip")
SPEECH = ai.Blob("audio/m4a", b"not really the sound of a clip")


# --- the providers ----------------------------------------------------------
# No database here: an adapter is a request built and an answer read.


def test_gemini_attaches_the_photo_and_asks_for_json():
    net = Recorded("gemini_describe.json")

    answered = providers.Gemini("test-key", transport=net).generate("инструкция", "вопрос", [PHOTO])

    assert json.loads(answered.text)["cast"] == ["человек-1", "животное-1"]
    sent = net.sent
    assert sent.full_url == providers.Gemini.ENDPOINT
    assert sent.get_header("X-goog-api-key") == "test-key"
    asked = body(sent)
    assert asked["model"] == providers.Gemini.DEFAULT_MODEL
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

    providers.Gemini("test-key", transport=net).generate("инструкция", "вопрос", [CLIP])

    assert body(net.sent)["input"][1]["type"] == "video"


def test_gemini_reads_the_answer_out_of_the_steps_when_there_is_no_shortcut():
    """`output_text` is a convenience the interaction may or may not carry; the model's own step
    always does."""
    net = Recorded("gemini_transcribe.json")

    assert "output_text" not in json.loads(net.answer)
    assert providers.Gemini("test-key", transport=net).transcribe(SPEECH).text.startswith("Ну что")
    assert body(net.sent)["input"][1]["mime_type"] == "audio/m4a"


def test_openai_sends_the_photo_as_a_data_uri_and_the_instruction_apart():
    net = Recorded("openai_describe.json")

    answered = providers.OpenAI("test-key", transport=net).generate("инструкция", "вопрос", [PHOTO])

    assert json.loads(answered.text)["cast"] == ["человек-1", "животное-1"]
    sent = net.sent
    assert sent.full_url == providers.OpenAI.ENDPOINT
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
        providers.OpenAI("test-key", transport=Recorded("openai_describe.json")).generate(
            "и", "в", [CLIP]
        )


def test_openai_uploads_speech_as_a_file():
    net = Recorded("openai_transcribe.json")

    assert providers.OpenAI("test-key", transport=net).transcribe(SPEECH).text.startswith("Ну что")

    sent = net.sent
    assert sent.full_url == providers.OpenAI.TRANSCRIPTIONS
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
    answer as one — would hand `Authorization` straight to wherever it points. `providers._urlopen`
    does
    not follow redirects at all; the 30x comes back as the failure it is."""
    where, followed = redirecting
    request = urllib.request.Request(
        where,
        data=b"{}",
        method="POST",
        headers={"Authorization": "Bearer test-key", "Content-Type": "application/json"},
    )

    with pytest.raises(urllib.error.HTTPError) as refused:
        providers._urlopen(request)

    assert refused.value.code == 302
    assert followed == [], "nothing went to where it pointed, and the key least of all"


# --- which provider ---------------------------------------------------------


def test_the_setting_alone_decides_who_answers(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert isinstance(ai.from_env(), providers.OpenAI)

    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-pro")
    chosen = ai.from_env()
    assert isinstance(chosen, providers.Gemini) and chosen.model == "gemini-3.8-pro"


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


def journalled(conn):
    return conn.execute(
        """SELECT provider, model, operation, ref, status, request, response, error, took_ms
           FROM ai_requests ORDER BY id"""
    ).fetchall()


def counts(conn):
    return conn.execute(
        "SELECT input_tokens, output_tokens, total_tokens, usage FROM ai_requests ORDER BY id"
    ).fetchall()


def test_a_call_and_its_answer_are_both_in_the_journal(conn):
    net = Recorded("gemini_describe.json")
    gateway = ai.AI(conn, providers.Gemini("test-key", transport=net))

    described = gateway.describe_media([PHOTO], cast=["человек-1"], ref="media-7")

    assert described["description"].startswith("Женщина")
    ((provider, model, operation, ref, status, request, response, error, took_ms),) = journalled(
        conn
    )
    assert (provider, model) == ("gemini", providers.Gemini.DEFAULT_MODEL)
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
        seen.append(counts(conn))
        return (FIXTURES / "gemini_describe.json").read_bytes()

    ai.AI(conn, providers.Gemini("test-key", transport=look)).describe_media([PHOTO])

    assert seen == [[("sent",)], [(None, None, None, None)]], "nothing counted before the answer"


def test_a_call_that_failed_is_in_the_journal_too(conn):
    def refuse(request):
        raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {}, None)

    gateway = ai.AI(conn, providers.OpenAI("test-key", transport=refuse))
    with pytest.raises(urllib.error.HTTPError):
        gateway.transcribe_speech(SPEECH, ref="media-7")

    ((*_, status, request, response, error, _),) = journalled(conn)
    assert (status, response) == ("failed", None)
    assert "429" in error
    assert request["attached"] == [{"mime_type": "audio/m4a", "bytes": len(SPEECH.data)}]
    assert counts(conn) == [(None, None, None, None)], "no answer, nothing counted"


def test_an_answer_that_is_not_json_fails_the_call_and_says_what_came_instead(conn):
    def prose(request):
        return json.dumps({"output_text": "Конечно! Вот описание кадра:"}).encode()

    gateway = ai.AI(conn, providers.Gemini("test-key", transport=prose))
    with pytest.raises(RuntimeError, match="Конечно"):
        gateway.describe_media([PHOTO])

    ((*_, status, _, _, error, _),) = journalled(conn)
    assert status == "failed" and "Конечно" in error


def test_an_answer_that_parses_but_is_not_an_object_fails_the_call_too(conn):
    """A bare `null` is valid JSON and nothing a caller can read a field off. Failed here, where the
    answer itself is kept — rather than three handlers down as an `AttributeError` about `NoneType`,
    which says nothing about what the model actually said."""

    def nothing(request):
        return json.dumps({"output_text": "null"}).encode()

    gateway = ai.AI(conn, providers.Gemini("test-key", transport=nothing))
    with pytest.raises(RuntimeError, match="null"):
        gateway.describe_media([PHOTO])

    ((*_, status, _, _, error, _),) = journalled(conn)
    assert status == "failed" and "null" in error


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"])
def test_a_number_the_journal_cannot_hold_is_dropped_and_the_answer_kept(conn, number):
    """`json.loads` reads each of these into a float `jsonb` does not take. The answer is paid for
    by then: refused, or breaking the journal's UPDATE on the way back, it is paid for again on the
    retry. The number goes, the rest of the answer stays."""
    said = f'{{"description": "кадр", "cast": ["человек-1"], "seconds": {number}}}'
    gateway = ai.AI(conn, providers.Gemini("test-key", transport=saying(said)))

    described = gateway.describe_media([PHOTO])

    assert described == {"description": "кадр", "cast": ["человек-1"], "seconds": None}
    ((*_, status, _, response, error, _),) = journalled(conn)
    assert (status, response, error) == ("answered", described, None)


def test_a_character_the_journal_cannot_hold_is_dropped_and_the_answer_kept(conn):
    """A NUL, and half of a surrogate pair: neither `jsonb` nor `text` takes one. Escaped in the
    model's JSON or already a character of the envelope around it, both are gone from what the
    journal and the caller get — and nothing else of the answer is."""
    said = (
        '{"description": "Торт\\ud800 со свечами\\u0000", '
        '"cast": ["человек-1\udc00", "живот\x00ное-1"]}'
    )
    gateway = ai.AI(conn, providers.Gemini("test-key", transport=saying(said)))

    described = gateway.describe_media([PHOTO])

    assert described == {"description": "Торт со свечами", "cast": ["человек-1", "животное-1"]}
    ((*_, status, _, response, error, _),) = journalled(conn)
    assert (status, response, error) == ("answered", described, None)


def nested(depth, inside=""):
    return "[" * depth + inside + "]" * depth


@pytest.mark.parametrize("depth", [answers.DEPTH_LIMIT, sys.getrecursionlimit(), 100_000])
def test_nesting_past_the_limit_is_dropped_and_the_answer_kept(conn, depth):
    """A model can answer brackets a thousand deep, prompt-injected or not. Past Python's recursion
    limit `_holdable` gave out, past its own `json.loads` did — both after the answer was paid for,
    and the retry paid for it again. What is nested past `DEPTH_LIMIT` becomes `null`; everything
    up to it stays whole, and brackets inside a string are words, not nesting."""
    words = 'Торт "с короной" ' + "[" * (answers.DEPTH_LIMIT + 1)
    said = (
        f'{{"description": {json.dumps(words, ensure_ascii=False)}, "cast": ["человек-1"], '
        f'"whole": {nested(answers.DEPTH_LIMIT - 1)}, "cut": {nested(depth)}}}'
    )
    gateway = ai.AI(conn, providers.Gemini("test-key", transport=saying(said)))

    described = gateway.describe_media([PHOTO])

    assert described == {
        "description": words,
        "cast": ["человек-1"],
        "whole": json.loads(nested(answers.DEPTH_LIMIT - 1)),
        "cut": json.loads(nested(answers.DEPTH_LIMIT - 1, "null")),
    }
    ((*_, status, _, response, error, _),) = journalled(conn)
    assert (status, response, error) == ("answered", described, None)


def test_a_long_string_costs_the_depth_scan_no_memory_per_character():
    """The depth scan reads every string of an answer through. Free to backtrack into it, the regex
    would keep a frame per character — a hundred bytes each, a gigabyte for ten megabytes."""
    text = json.dumps("слово " * 20_000)
    tracemalloc.start()
    answers._shallow(text)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()

    assert peak < 10 * len(text)


def test_prose_carrying_such_a_character_still_fails_into_the_journal(conn):
    """The failed call keeps the text that failed it in `error`, a `text` column: with half a
    surrogate pair in it, the row itself would not close, and the journal would say «sent» for
    good."""
    gateway = ai.AI(
        conn, providers.Gemini("test-key", transport=saying("Конечно\ud800! Вот описание:"))
    )

    with pytest.raises(RuntimeError, match="Конечно! Вот описание"):
        gateway.describe_media([PHOTO])

    ((*_, status, _, _, error, _),) = journalled(conn)
    assert status == "failed" and "Конечно! Вот описание" in error


def test_the_story_reaches_the_director(conn):
    """Two orders on the same material differ only by what was asked — so it has to be asked."""
    net = Recorded("gemini_plan.json")
    gateway = ai.AI(conn, providers.Gemini("test-key", transport=net))

    planned = gateway.plan_montage(
        [{"media_id": "m-1", "description": "накрывают на стол"}],
        story="день рождения, всё вокруг торта",
        target_seconds=30,
        ref="montage-3",
    )

    assert [piece["media_id"] for piece in planned["plan"]] == ["m-1", "m-2", "m-3"]
    asked = body(net.sent)["input"][0]["text"]
    assert "всё вокруг торта" in asked and "30" in asked
    ((*_, operation, ref, status, _, _, _, _),) = journalled(conn)
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
    ((provider, model, operation, ref, status, _, response, _, _),) = journalled(conn)
    assert (provider, model, status, response) == ("gemini", None, "skipped", None)
    assert (operation, ref) == ("describe_media", "media-7")
    assert counts(conn) == [(None, None, None, None)], "no call, nothing counted"


def test_the_journal_keeps_neither_the_key_nor_the_material(conn):
    """It outlives every key in it and is read by hand in psql; and a copy of the memory's photos
    in jsonb would make it the biggest table in the database."""
    gateway = ai.AI(conn, providers.Gemini("test-key", transport=Recorded("gemini_describe.json")))

    gateway.describe_media([PHOTO])

    (written,) = conn.execute("SELECT to_jsonb(ai_requests)::text FROM ai_requests").fetchone()
    assert "test-key" not in written
    assert base64.b64encode(PHOTO.data).decode() not in written


# --- what it cost -----------------------------------------------------------


ASKED = {
    "describe_media": lambda gateway: gateway.describe_media([PHOTO]),
    "transcribe_speech": lambda gateway: gateway.transcribe_speech(SPEECH),
    "plan_montage": lambda gateway: gateway.plan_montage([{"media_id": "m-1"}]),
}


@pytest.mark.parametrize(
    "operation, fixture, thinking",
    [
        ("describe_media", "gemini_describe.json", {"thinking_level": "low"}),
        ("transcribe_speech", "gemini_transcribe.json", {"thinking_level": "low"}),
        ("plan_montage", "gemini_plan.json", None),
    ],
)
def test_a_look_thinks_the_least_and_the_director_as_long_as_it_likes(
    conn, operation, fixture, thinking
):
    """Thinking is billed at the output rate, and left to itself the model spent up to a thousand
    tokens of it on a sentence about a photo. `low` is the floor gemini-3.8-flash takes — `minimal`
    it refuses, and interactions has no budget in tokens; the director's plan is not held back."""
    net = Recorded(fixture)

    ASKED[operation](ai.AI(conn, providers.Gemini("test-key", transport=net)))

    assert body(net.sent).get("generation_config") == thinking


@pytest.mark.parametrize(
    "provider, operation, fixture, counted",
    [
        (providers.Gemini, "describe_media", "gemini_describe.json", (1170, 61, 1231)),
        (providers.Gemini, "transcribe_speech", "gemini_transcribe.json", (886, 17, 903)),
        # 124 tokens of answer and 96 of thinking, both billed at the output rate.
        (providers.Gemini, "plan_montage", "gemini_plan.json", (402, 220, 622)),
        (providers.OpenAI, "describe_media", "openai_describe.json", (1146, 48, 1194)),
        (providers.OpenAI, "transcribe_speech", "openai_transcribe.json", (112, 19, 131)),
        (providers.OpenAI, "plan_montage", "openai_plan.json", (1204, 214, 1418)),
    ],
)
def test_what_the_provider_counted_is_on_the_row_next_to_the_answer(
    conn, provider, operation, fixture, counted
):
    """The bill is in tokens, and only the provider knows how many: its count goes on the same row
    as the answer, in three numbers to add up and whole as it came."""
    net = Recorded(fixture)

    ASKED[operation](ai.AI(conn, provider("test-key", transport=net)))

    ((status, response, *tokens, usage),) = conn.execute(
        """SELECT status, response, input_tokens, output_tokens, total_tokens, usage
           FROM ai_requests"""
    ).fetchall()
    assert status == "answered" and response, "the answer is on the row as before"
    assert tuple(tokens) == counted
    assert usage == json.loads(net.answer)["usage"]


def test_an_answer_that_counts_nothing_leaves_the_count_empty(conn):
    """Empty, not zero: a zero would be a count nobody made."""
    gateway = ai.AI(conn, providers.Gemini("test-key", transport=saying('{"description": "кадр"}')))

    gateway.describe_media([PHOTO])

    assert counts(conn) == [(None, None, None, None)]


def test_a_transcription_billed_in_seconds_keeps_the_seconds_and_no_tokens(conn):
    """OpenAI's whisper counts audio in seconds: no tokens to take out, the count itself kept."""
    usage = {"type": "duration", "seconds": 9}
    answer = json.dumps({"text": "Ну что", "usage": usage}).encode()
    gateway = ai.AI(conn, providers.OpenAI("test-key", transport=lambda request: answer))

    gateway.transcribe_speech(SPEECH)

    assert counts(conn) == [(None, None, None, usage)]


@pytest.mark.parametrize(
    "number, kept",
    [
        ("1170", None),
        (1170.5, None),
        (-1, None),
        (True, None),
        (2**31, None),
        (2**31 - 1, 2**31 - 1),
    ],
)
def test_a_count_that_is_no_count_is_left_empty_and_the_answer_kept(conn, number, kept):
    """A count is the provider's word, not a trusted one. Past what the journal holds it would fail
    the UPDATE of a call already paid for — so it goes, and the answer and the other counts stay."""
    usage = {"total_input_tokens": number, "total_output_tokens": 61, "total_tokens": 1231}
    answer = json.dumps({"output_text": '{"description": "кадр"}', "usage": usage}).encode()
    gateway = ai.AI(conn, providers.Gemini("test-key", transport=lambda request: answer))

    assert gateway.describe_media([PHOTO]) == {"description": "кадр"}

    ((*tokens, _),) = counts(conn)
    assert tokens == [kept, 61, 1231]


def test_thinking_that_adds_the_output_past_what_the_journal_holds_is_no_count(conn):
    usage = {"total_output_tokens": answers.COUNT_LIMIT, "total_thought_tokens": 1}
    answer = json.dumps({"output_text": '{"description": "кадр"}', "usage": usage}).encode()
    gateway = ai.AI(conn, providers.Gemini("test-key", transport=lambda request: answer))

    gateway.describe_media([PHOTO])

    assert counts(conn) == [(None, None, None, usage)]


def test_prose_is_a_failed_call_that_was_paid_for_and_its_count_is_kept(conn):
    usage = {"total_input_tokens": 1170, "total_output_tokens": 9, "total_tokens": 1179}
    answer = json.dumps({"output_text": "Конечно! Вот описание кадра:", "usage": usage}).encode()
    gateway = ai.AI(conn, providers.Gemini("test-key", transport=lambda request: answer))

    with pytest.raises(RuntimeError, match="Конечно"):
        gateway.describe_media([PHOTO])

    ((*_, status, _, _, _, _),) = journalled(conn)
    assert status == "failed"
    assert counts(conn) == [(1170, 9, 1179, usage)]


def test_init_brings_an_older_journal_up_to_date_and_keeps_its_rows(conn):
    """A database the worker ran on before calls were counted: `init` adds the columns, empty on
    the rows already there and nothing else of them changed — and a second `init` is a no-op."""
    conn.execute(
        """ALTER TABLE ai_requests DROP COLUMN usage, DROP COLUMN input_tokens,
                                   DROP COLUMN output_tokens, DROP COLUMN total_tokens"""
    )
    conn.execute(
        """INSERT INTO ai_requests (provider, model, operation, ref, status, request, response,
                                    took_ms)
           VALUES ('gemini', 'gemini-3.8-flash', 'describe_media', 'media-7', 'answered',
                   '{"prompt": "вопрос"}', '{"description": "кадр"}', 2300)"""
    )
    before = conn.execute("SELECT * FROM ai_requests").fetchall()

    worker.init_schema(conn)
    worker.init_schema(conn)

    assert conn.execute("SELECT * FROM ai_requests").fetchall() == [
        (*row, None, None, None, None) for row in before
    ]
    assert counts(conn) == [(None, None, None, None)]
