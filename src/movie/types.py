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


class ReportDimensions(TypedDict):
    width: int
    height: int


class ReportMovie(TypedDict):
    targetSeconds: int
    dimensions: ReportDimensions
    story: str | None
    requiredItemIds: list[str]


class ReportShot(TypedDict):
    itemId: str
    mediaId: str
    kind: ShotKind
    capturedAt: str | None
    width: int | None
    height: int | None
    durationSeconds: float | None
    hasAudio: bool
    shownWithCover: bool
    description: str | None
    labels: list[str]
    analysis: dict | None


class ReportMusic(TypedDict):
    songId: str
    mediaId: str
    name: str
    author: str
    durationSeconds: float | None
    analysis: dict | None


class MontageReport(TypedDict):
    movie: ReportMovie
    shots: list[ReportShot]
    music: list[ReportMusic]
