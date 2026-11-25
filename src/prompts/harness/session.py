import json

from src.harness.policies import LevelPolicy


def render(context: dict, policy: LevelPolicy, contact: list[dict]) -> str:
    lines = [
        "Here is the order and the material.",
        json.dumps(context, ensure_ascii=False),
        f"Budget: {policy.max_turns} turns, {policy.max_tool_calls} tool calls, "
        f"{policy.max_distinct_tracks} tracks to listen to.",
    ]
    if contact:
        lines.append("The attached contact sheet shows these shots, in this order:")
        lines.append(json.dumps(contact, ensure_ascii=False))
    return "\n".join(lines)
