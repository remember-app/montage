import json

from src.harness.tools.context import ToolContext
from src.integrations.ai import ToolCall, ToolResult
from src.movie import editor


def run(context: ToolContext, call: ToolCall) -> ToolResult:
    commands = call.arguments.get("commands")
    if not isinstance(commands, list) or not commands:
        return _error(call, "commands must be a non-empty array")
    if len(context.versions) >= context.policy.max_plan_versions:
        return _error(
            call,
            f"at most {context.policy.max_plan_versions} plan versions; "
            "finish with the current one",
        )
    try:
        editor.validate(context.conn, context.montage_id, commands)
    except RuntimeError as exc:
        errors = [part for part in str(exc).split("; ") if part]
        return ToolResult(
            call.id,
            json.dumps({"ok": False, "errors": errors}, ensure_ascii=False),
            is_error=True,
            name=call.name,
        )

    version = len(context.versions) + 1
    landed = context.conn.execute(
        """
        update montages
        set manifest = jsonb_set(
                manifest, '{editingParams}',
                jsonb_build_object(
                    'commands', %s::jsonb,
                    'versions',
                    coalesce(manifest->'editingParams'->'versions', '[]'::jsonb) || %s::jsonb
                )
            )
        where id = %s
        """,
        (json.dumps(commands), json.dumps([commands]), context.montage_id),
    ).rowcount
    if not landed:
        return _error(call, "the montage went away")
    context.versions.append(commands)
    shots = [command for command in commands if command.get("type") == "shot"]
    return ToolResult(
        call.id,
        json.dumps(
            {"ok": True, "version": version, "seconds": round(editor.timeline_length(shots), 2)},
            ensure_ascii=False,
        ),
        name=call.name,
    )


def _error(call: ToolCall, message: str) -> ToolResult:
    return ToolResult(
        call.id,
        json.dumps({"ok": False, "error": message}, ensure_ascii=False),
        is_error=True,
        name=call.name,
    )


if __name__ == "__main__":
    from src.harness.policies import LOW
    from src.harness.tools.context import ToolContext
    from src.movie.preparation import Order

    class Result:
        rowcount = 1

    class Conn:
        def __init__(self):
            self.args = None

        def execute(self, _sql, args):
            self.args = args
            return Result()

    class FailConn(Conn):
        def execute(self, _sql, args):
            raise RuntimeError("item a is not in this project; the plan is too long")

    commands = [
        {
            "type": "shot",
            "itemId": "a",
            "mediaId": "a",
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
    conn = Conn()
    context = ToolContext(conn, "montage", 0, LOW, Order({}, {}, [], []), {}, {})
    original = editor.validate
    editor.validate = lambda *_args, **_kwargs: None
    try:
        ok = run(context, ToolCall("c1", "plan_update", {"commands": commands}))
        payload = json.loads(ok.text)
        assert payload == {"ok": True, "version": 1, "seconds": 2.0}
        assert context.versions == [commands]
        assert conn.args[0] == json.dumps(commands)
    finally:
        editor.validate = original

    failing = ToolContext(FailConn(), "montage", 0, LOW, Order({}, {}, [], []), {}, {})
    refused = run(failing, ToolCall("c2", "plan_update", {"commands": commands}))
    assert refused.is_error
    assert "not in this project" in json.loads(refused.text)["errors"][0]
    print("ok")
