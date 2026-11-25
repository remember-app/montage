"""Цикл агентного монтажа: сессия, инструменты и бюджет."""

import json
import logging
import uuid

import psycopg

from src import config
from src.harness import context as harness_context
from src.harness import policies
from src.harness.tools import registry
from src.harness.tools.context import make as make_tool_context
from src.integrations import ai, ai_sessions
from src.prompts.harness import session, system

log = logging.getLogger("harness.agent")


def run(conn: psycopg.Connection, montage_id: uuid.UUID, level: str, job_id: int) -> None:
    policy = policies.get(level)
    built = harness_context.build(conn, montage_id, policy)
    tools_context = make_tool_context(conn, montage_id, job_id, policy, built.order)
    specs = registry.specs(tools_context)
    tools = tuple(spec[0] for spec in specs.values())
    first, session_id = ai_sessions.turn(
        conn,
        operation="text",
        instruction=system.SYSTEM,
        prompt=session.render(built.data, policy, built.labels),
        blobs=(built.contact,) if built.contact else (),
        tools=tools,
        session_id=None,
        stream=config.ai_stream(),
        job_id=job_id,
    )

    def send(prompt: str, results: tuple, previous_id: str | None) -> ai.Reply:
        nonlocal session_id
        reply, session_id = ai_sessions.turn(
            conn,
            operation="text",
            instruction=system.SYSTEM,
            prompt=prompt,
            previous_interaction_id=previous_id,
            tools=tools,
            tool_results=results,
            session_id=session_id,
            stream=config.ai_stream(),
            job_id=job_id,
        )
        return reply

    def execute(call: ai.ToolCall) -> ai.ToolResult:
        spec = specs.get(call.name)
        if spec is None:
            return ai.ToolResult(
                call.id,
                json.dumps({"ok": False, "error": f"unknown tool {call.name}"}),
                is_error=True,
                name=call.name,
            )
        return spec[1](tools_context, call)

    summary = _loop(policy, first, send, execute, lambda: bool(tools_context.versions))
    _journal(conn, montage_id, summary, tools_context.versions[-1])
    log.info(
        "montage %s directed by the agent: %s plan versions",
        montage_id,
        len(tools_context.versions),
    )


def _loop(policy, first, send, execute, has_plan) -> str:
    reply = first
    turns = 1
    calls = 0
    while True:
        if reply.tool_calls:
            if turns >= policy.max_turns:
                break
            results = []
            for call in reply.tool_calls:
                calls += 1
                if calls > policy.max_tool_calls:
                    results.append(
                        ai.ToolResult(
                            call.id,
                            json.dumps({"ok": False, "error": "tool call limit reached"}),
                            is_error=True,
                            name=call.name,
                        )
                    )
                    continue
                results.append(execute(call))
            warning = ""
            if turns + 1 >= policy.max_turns:
                warning = (
                    "You have one turn left. Call plan_update now with the best complete plan "
                    "you can; do not inspect anything else."
                )
            reply = send(warning, tuple(results), reply.interaction_id)
            turns += 1
            continue
        if has_plan():
            return reply.text or ""
        if turns >= policy.max_turns:
            break
        reply = send(
            "Call plan_update with a valid plan before finishing.", (), reply.interaction_id
        )
        turns += 1
    if has_plan():
        return reply.text or ""
    raise RuntimeError("the agent did not produce a valid plan")


def _journal(conn: psycopg.Connection, montage_id: uuid.UUID, summary: str, plan: list) -> None:
    used: list[str] = []
    for command in plan:
        if command.get("type") != "shot":
            continue
        item_id = str(command.get("itemId"))
        if item_id not in used:
            used.append(item_id)
    text = (summary or "").strip() or "The movie is assembled."
    conn.execute(
        """
        update montages
        set journal = jsonb_set(
                coalesce(journal, '{}'::jsonb), '{directing}',
                jsonb_build_object('text', %s::text, 'usedItemIds', %s::jsonb)
            )
        where id = %s
        """,
        (text, json.dumps(used), montage_id),
    )


if __name__ == "__main__":
    from src.harness.policies import LOW

    def reply(text: str = "", calls=(), interaction: str = "i") -> ai.Reply:
        return ai.Reply(text, None, interaction, tuple(calls))

    plans: list[list] = []

    def execute(call: ai.ToolCall) -> ai.ToolResult:
        if call.name == "plan_update":
            plans.append(call.arguments["commands"])
        return ai.ToolResult(call.id, "{}")

    scripted = [
        reply(calls=(ai.ToolCall("c1", "get_shots", {"itemIds": ["a"]}),), interaction="i1"),
        reply(
            calls=(ai.ToolCall("c2", "plan_update", {"commands": [{"type": "shot"}]}),),
            interaction="i2",
        ),
        reply("done", interaction="i3"),
    ]
    sent = []

    def send(prompt: str, results: tuple, previous: str | None) -> ai.Reply:
        sent.append((prompt, results, previous))
        return scripted.pop(0)

    summary = _loop(LOW, scripted.pop(0), send, execute, lambda: bool(plans))
    assert summary == "done"
    assert plans == [[{"type": "shot"}]]
    assert sent[0][2] == "i1"

    stuck = [
        reply(calls=(ai.ToolCall("c1", "get_shots", {}),), interaction=f"i{index}")
        for index in range(10)
    ]
    try:
        _loop(
            LOW,
            stuck.pop(0),
            lambda prompt, results, previous: stuck.pop(0),
            lambda call: ai.ToolResult(call.id, "{}"),
            lambda: False,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("a session without a plan must fail")

    plans.clear()
    sequence = [
        reply("thinking", interaction="i1"),
        reply(
            calls=(ai.ToolCall("c1", "plan_update", {"commands": [{"type": "shot"}]}),),
            interaction="i2",
        ),
        reply("ok", interaction="i3"),
    ]
    summary = _loop(
        LOW,
        sequence.pop(0),
        lambda prompt, results, previous: sequence.pop(0),
        execute,
        lambda: bool(plans),
    )
    assert summary == "ok"
    print("ok")
