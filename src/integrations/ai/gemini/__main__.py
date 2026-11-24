import http.client
import json
import os
from unittest import mock

from src.integrations.ai.contract import Usage
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


def respond(request, timeout):
    seen.append(json.loads(request.data))
    return Fake({"id": f"i{len(seen)}", "output_text": "{}", "usage": {"total_input_tokens": 1}})


with mock.patch("urllib.request.urlopen", respond):
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

with mock.patch("urllib.request.urlopen", side_effect=http.client.RemoteDisconnected("closed")):
    try:
        GeminiProvider().ask("be a director", "third")
    except RuntimeError as exc:
        assert "model request failed" in str(exc)
    else:
        raise AssertionError("a dropped connection must fail loudly")
print("ok")
