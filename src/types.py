from dataclasses import dataclass
from typing import Literal, TypedDict

JobKind = Literal["media_derive", "montage"]
MediaKind = Literal["photo", "video", "audio"]
MontageStage = Literal["prep", "analysis", "directing", "music", "assembly"]


class ProjectMediaItemAIAnalysis(TypedDict):
    description: str
    speech: str | None
    tags: list[str]
    score: float


@dataclass(frozen=True)
class MontageConfig:
    target_seconds: int
    prompt: str | None
    include_item_ids: tuple[str, ...]
    required_item_ids: tuple[str, ...]
    music_ids: tuple[str, ...] | None
    width: int
    height: int
