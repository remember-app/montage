"""Бюджеты и наборы инструментов по уровням монтажа."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class LevelPolicy:
    name: str
    available: bool
    max_turns: int
    max_tool_calls: int
    max_audio_inspects: int
    max_distinct_tracks: int
    max_shots: int
    max_get_shots_batch: int
    max_plan_versions: int
    contact_sheet_shots: int
    tools: tuple[str, ...]


LOW = LevelPolicy(
    name="low",
    available=True,
    max_turns=10,
    max_tool_calls=12,
    max_audio_inspects=3,
    max_distinct_tracks=3,
    max_shots=60,
    max_get_shots_batch=8,
    max_plan_versions=4,
    contact_sheet_shots=20,
    tools=("get_shots", "inspect", "plan_update"),
)

POLICIES: dict[str, LevelPolicy | None] = {"low": LOW, "medium": None, "high": None}


def get(level: str) -> LevelPolicy:
    policy = POLICIES.get(level)
    if policy is None or not policy.available:
        raise RuntimeError(f"montage level {level!r} is not implemented yet")
    return policy


if __name__ == "__main__":
    low = get("low")
    assert low.tools == ("get_shots", "inspect", "plan_update")
    assert low.max_turns > 0 and low.max_tool_calls >= low.max_turns
    assert low.max_distinct_tracks == low.max_audio_inspects
    for level in ("medium", "high"):
        try:
            get(level)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"{level} must stay unavailable until it is written")
    print("ok")
