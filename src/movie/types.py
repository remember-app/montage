from typing import Literal, TypedDict

ShotKind = Literal["photo", "video", "audio"]
Motion = Literal["none", "zoom_in", "zoom_out"]


class Framing(TypedDict):
    x: float
    y: float
    w: float
    h: float


class ShotCommand(TypedDict):
    type: Literal["shot"]
    itemId: str
    mediaId: str
    kind: ShotKind
    sourceStartSeconds: float
    seconds: float
    liveLevel: float
    framing: Framing | None
    motion: Motion
    transitionSeconds: float


class MusicCommand(TypedDict):
    type: Literal["music"]
    songId: str
    mediaId: str
    sourceStartSeconds: float
    seconds: float
    atSeconds: float
    gain: float
    fadeInSeconds: float
    fadeOutSeconds: float


EditCommand = ShotCommand | MusicCommand
EditPlan = list[EditCommand]
