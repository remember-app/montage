from typing import Callable

from src.harness.tools import get_shots, inspect, plan_update
from src.harness.tools.context import ToolContext
from src.integrations.ai import Tool, ToolCall, ToolResult

SHOT_COMMAND = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["shot"]},
        "itemId": {"type": "string"},
        "mediaId": {"type": "string"},
        "kind": {"type": "string", "enum": ["photo", "video", "audio"]},
        "sourceStartSeconds": {"type": "number"},
        "seconds": {"type": "number"},
        "liveLevel": {"type": "number"},
        "framing": {
            "type": "object",
            "nullable": True,
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "w": {"type": "number"},
                "h": {"type": "number"},
            },
            "required": ["x", "y", "w", "h"],
        },
        "motion": {"type": "string", "enum": ["none", "zoom_in", "zoom_out"]},
        "transitionSeconds": {"type": "number"},
        "role": {"type": "string"},
    },
    "required": [
        "type",
        "itemId",
        "mediaId",
        "kind",
        "sourceStartSeconds",
        "seconds",
        "liveLevel",
        "motion",
        "transitionSeconds",
        "role",
    ],
}

MUSIC_COMMAND = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["music"]},
        "songId": {"type": "string"},
        "mediaId": {"type": "string"},
        "sourceStartSeconds": {"type": "number"},
        "seconds": {"type": "number"},
        "atSeconds": {"type": "number"},
        "gain": {"type": "number"},
        "fadeInSeconds": {"type": "number"},
        "fadeOutSeconds": {"type": "number"},
    },
    "required": [
        "type",
        "songId",
        "mediaId",
        "sourceStartSeconds",
        "seconds",
        "atSeconds",
        "gain",
        "fadeInSeconds",
        "fadeOutSeconds",
    ],
}


def specs(
    context: ToolContext,
) -> dict[str, tuple[Tool, Callable[[ToolContext, ToolCall], ToolResult]]]:
    max_batch = context.policy.max_get_shots_batch
    max_tracks = context.policy.max_distinct_tracks
    all_specs = {
        "get_shots": (
            Tool(
                "get_shots",
                f"Full analysis for a batch of shots, at most {max_batch} per call.",
                {
                    "type": "object",
                    "properties": {
                        "itemIds": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": max_batch,
                        }
                    },
                    "required": ["itemIds"],
                    "additionalProperties": False,
                },
            ),
            get_shots.run,
        ),
        "inspect": (
            Tool(
                "inspect",
                "Load audio into the conversation so you can hear it. "
                f"Songs only, at most {max_tracks} different tracks per montage.",
                {
                    "type": "object",
                    "properties": {
                        "mediaIds": {"type": "array", "items": {"type": "string"}, "minItems": 1}
                    },
                    "required": ["mediaIds"],
                    "additionalProperties": False,
                },
            ),
            inspect.run,
        ),
        "plan_update": (
            Tool(
                "plan_update",
                "Replace the whole edit plan. Validates and stores it only when valid; "
                "returns the version or the errors.",
                {
                    "type": "object",
                    "properties": {
                        "commands": {
                            "type": "array",
                            "items": {"anyOf": [SHOT_COMMAND, MUSIC_COMMAND]},
                            "minItems": 1,
                        }
                    },
                    "required": ["commands"],
                    "additionalProperties": False,
                },
            ),
            plan_update.run,
        ),
    }
    return {name: all_specs[name] for name in context.policy.tools if name in all_specs}
