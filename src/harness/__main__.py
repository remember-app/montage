"""Сквозная проверка harness: контекст, инструменты, цикл и план — без реального ИИ."""

from src.harness import agent, policies
from src.harness.tools import registry
from src.harness.tools.context import make
from src.harness.tools.plan_update import run as plan_update
from src.integrations import ai
from src.movie import editor
from src.movie.preparation import Order, ShotRow


class Result:
    rowcount = 1


class Conn:
    def __init__(self):
        self.rows = []

    def execute(self, sql, args):
        self.rows.append((sql, args))
        return Result()


def shot(item: str) -> ShotRow:
    return ShotRow(
        item,
        f"media-{item}",
        "photo",
        None,
        {"summary": "a moment", "scoreImportance": 7, "scoreQuality": 6},
        2.0,
        1080,
        1920,
        "2026-11-16T12:00:00+00:00",
        "thumb",
        "0000000000000000",
        "original",
        [],
    )


order = Order(
    {
        "targetSeconds": 2,
        "dimensions": {"width": 1080, "height": 1920},
        "includeItemIds": ["a"],
        "requiredItemIds": [],
        "musicIds": [],
    },
    {"title": "Evening", "description": ""},
    [shot("a")],
    [],
)
conn = Conn()
context = make(conn, "montage", 0, policies.LOW, order)
specs = registry.specs(context)
assert tuple(specs) == policies.LOW.tools

commands = [
    {
        "type": "shot",
        "itemId": "a",
        "mediaId": "media-a",
        "kind": "photo",
        "sourceStartSeconds": 0.0,
        "seconds": 2.0,
        "liveLevel": 0.0,
        "framing": None,
        "motion": "none",
        "transitionSeconds": 0.0,
        "role": "action",
    }
]


def execute(call: ai.ToolCall) -> ai.ToolResult:
    assert call.name == "plan_update", call.name
    return plan_update(context, call)


def reply(text: str = "", calls=(), interaction: str = "i") -> ai.Reply:
    return ai.Reply(text, None, interaction, tuple(calls))


first = reply(calls=(ai.ToolCall("c1", "plan_update", {"commands": commands}),), interaction="i1")
script = [reply("done", interaction="i2")]


def send(prompt: str, results: tuple, previous: str | None) -> ai.Reply:
    return script.pop(0)


original = editor.validate
editor.validate = lambda *_args, **_kwargs: None
try:
    summary = agent._loop(policies.LOW, first, send, execute, lambda: bool(context.versions))
finally:
    editor.validate = original

assert summary == "done"
assert context.versions == [commands]
agent._journal(conn, "montage", summary, context.versions[-1])
assert len(conn.rows) == 2, "one plan write and one journal write"
assert "journal" in conn.rows[-1][0]
print("ok")
