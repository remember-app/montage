import http.client
import json
import os
from unittest import mock

from src.integrations.ai.contract import Tool, ToolCall, ToolResult, Usage
from src.integrations.ai.gemini import GeminiProvider

os.environ.setdefault("GEMINI_API_KEY", "test")
seen = []


class Fake:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


class Stream:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __iter__(self):
        return iter(self.lines)


def respond(payload, dynamic: bool = False):
    def call(request, timeout):
        seen.append(json.loads(request.data))
        current = dict(payload)
        if dynamic:
            current["id"] = f"i{len(seen)}"
        return Fake(current)

    return call


with mock.patch(
    "urllib.request.urlopen",
    respond({"id": "i1", "output_text": "{}", "usage": {"total_input_tokens": 1}}, dynamic=True),
):
    first = GeminiProvider().ask("be a director", "first")
    second = GeminiProvider().ask(
        "be a director", "second", previous_interaction_id=first.interaction_id
    )
assert (first.interaction_id, second.interaction_id) == ("i1", "i2")
assert first.usage == Usage(input_tokens=1)
assert GeminiProvider().name == "gemini"
assert GeminiProvider().model
assert seen[0]["input"][0]["text"] == "be a director"
assert "previous_interaction_id" not in seen[0]
assert seen[1]["input"][0]["text"] == "second"
assert seen[1]["previous_interaction_id"] == "i1"
assert all(part.get("text") != "be a director" for part in seen[1]["input"])

tool = Tool("get_shots", "Full analysis for a batch of shots", {"type": "object"})
with mock.patch(
    "urllib.request.urlopen",
    respond(
        {
            "id": "i3",
            "steps": [
                {
                    "type": "function_call",
                    "id": "c1",
                    "name": "get_shots",
                    "arguments": {"itemIds": ["a"]},
                }
            ],
            "usage": {"total_input_tokens": 2},
        }
    ),
):
    asked = GeminiProvider().ask("be a director", "first", tools=(tool,))
assert asked.tool_calls == (ToolCall("c1", "get_shots", {"itemIds": ["a"]}),)
assert seen[-1]["tools"][0] == {
    "type": "function",
    "name": "get_shots",
    "description": "Full analysis for a batch of shots",
    "parameters": {"type": "object"},
}

with mock.patch(
    "urllib.request.urlopen", respond({"id": "i4", "output_text": "done", "usage": {}})
):
    answered = GeminiProvider().ask(
        "be a director",
        "",
        previous_interaction_id="i3",
        tools=(tool,),
        tool_results=(ToolResult("c1", "{}", name="get_shots"),),
    )
assert answered.text == "done"
assert seen[-1]["previous_interaction_id"] == "i3"
assert seen[-1]["input"][0]["type"] == "function_result"
assert seen[-1]["input"][0]["call_id"] == "c1"
assert seen[-1]["input"][0]["name"] == "get_shots"
assert seen[-1]["tools"][0]["name"] == "get_shots"


def background(request, timeout):
    if request.get_method() == "POST":
        seen.append(json.loads(request.data))
        return Fake({"id": "bg1", "status": "in_progress"})
    return Fake(
        {
            "id": "bg1",
            "status": "completed",
            "output_text": "{}",
            "usage": {"total_input_tokens": 3},
        }
    )


with mock.patch("urllib.request.urlopen", background):
    bg = GeminiProvider().ask("be a director", "first", background=True)
assert bg.text == "{}" and bg.interaction_id == "bg1"
assert seen[-1]["background"] is True


def action(request, timeout):
    if request.get_method() == "POST":
        return Fake({"id": "bg2", "status": "in_progress"})
    return Fake(
        {
            "id": "bg2",
            "status": "requires_action",
            "steps": [
                {
                    "type": "function_call",
                    "id": "c9",
                    "name": "get_shots",
                    "arguments": {"itemIds": ["a"]},
                }
            ],
            "usage": {"total_input_tokens": 4},
        }
    )


with mock.patch("urllib.request.urlopen", action):
    acted = GeminiProvider().ask("be a director", "first", background=True)
assert acted.tool_calls == (ToolCall("c9", "get_shots", {"itemIds": ["a"]}),)


def stream_respond(lines):
    def call(request, timeout):
        seen.append(json.loads(request.data))
        return Stream(lines)

    return call


stream_lines = [
    b'data: {"interaction":{"id":"s1"},"event_type":"interaction.created"}\n\n',
    b'data: {"index":0,"step":{"type":"model_output"},"event_type":"step.start"}\n\n',
    b'data: {"index":0,"delta":{"text":"Hello","type":"text"},"event_type":"step.delta"}\n\n',
    b'data: {"index":0,"event_type":"step.stop"}\n\n',
    b'data: {"interaction":{"id":"s1","status":"completed","usage":{"total_input_tokens":2}},'
    b'"event_type":"interaction.completed"}\n\n',
    b"data: [DONE]\n\n",
]
with mock.patch("urllib.request.urlopen", stream_respond(stream_lines)):
    streamed = GeminiProvider().ask("be a director", "first", stream=True)
assert streamed.text == "Hello" and streamed.interaction_id == "s1"
assert seen[-1]["stream"] is True

call_lines = [
    b'data: {"interaction":{"id":"s2"},"event_type":"interaction.created"}\n\n',
    b'data: {"index":0,"step":{"type":"function_call","id":"c1","name":"get_shots"},'
    b'"event_type":"step.start"}\n\n',
    b'data: {"index":0,"delta":{"arguments":"{\\"itemIds\\":","type":"arguments_delta"},'
    b'"event_type":"step.delta"}\n\n',
    b'data: {"index":0,"delta":{"arguments":"[\\"a\\"]}","type":"arguments_delta"},'
    b'"event_type":"step.delta"}\n\n',
    b'data: {"index":0,"event_type":"step.stop"}\n\n',
    b'data: {"interaction":{"id":"s2","status":"requires_action","usage":{"total_input_tokens":3}},'
    b'"event_type":"interaction.completed"}\n\n',
    b"data: [DONE]\n\n",
]
with mock.patch("urllib.request.urlopen", stream_respond(call_lines)):
    streamed_call = GeminiProvider().ask("be a director", "first", stream=True)
assert streamed_call.tool_calls == (ToolCall("c1", "get_shots", {"itemIds": ["a"]}),)

with mock.patch("urllib.request.urlopen", side_effect=http.client.RemoteDisconnected("closed")):
    try:
        GeminiProvider().ask("be a director", "third")
    except RuntimeError as exc:
        assert "model request failed" in str(exc)
    else:
        raise AssertionError("a dropped connection must fail loudly")
print("ok")
