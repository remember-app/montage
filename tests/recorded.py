"""Записанные ответы модели и транспорт, который их отдаёт. В сеть отсюда не ходит ничто.

Ответы лежат в `fixtures/` — их однажды сняли руками; здесь они просто подставляются вместо сети.
"""

import json

from montage import ai
from montage.ai import providers

from material import FIXTURES


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


def saying(text):
    """A model that says `text` — its words as they come, inside Gemini's envelope."""
    return lambda request: json.dumps({"output_text": text}).encode()


def planning_over(monkeypatch, media_ids, fixture="gemini_plan.json"):
    """The recorded plan (`fixtures/gemini_plan*.json`) as the only cloud model this worker can
    find.

    The ids in it are this run's own: a montage is planned over material whose ids Postgres makes up
    at insert time, so the three the answer was written about are rewritten into the three actually
    ordered. Everything else — what a plan is, what a step of it carries — is the recorded file's,
    and nothing here goes anywhere near the network.
    """
    recorded = Recorded(fixture)
    for index, media_id in enumerate(media_ids, start=1):
        recorded.answer = recorded.answer.replace(f"m-{index}".encode(), media_id.encode())
    monkeypatch.setattr(
        ai, "from_env", lambda *_, **__: providers.Gemini("test-key", transport=recorded)
    )
    return recorded
