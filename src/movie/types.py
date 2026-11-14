from typing import Literal, NotRequired, TypedDict

ShotKind = Literal["photo", "video", "audio"]
Motion = Literal["none", "zoom_in", "zoom_out"]
MontageLevel = Literal["low", "medium", "high"]


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
    role: str


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


class ReportEvent(TypedDict):
    title: str
    description: str


class ReportMovie(TypedDict):
    event: ReportEvent
    targetSeconds: int
    dimensions: ReportDimensions
    story: str | None
    requiredItemIds: list[str]
    level: MontageLevel


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
    comment: str | None
    labels: list[str]
    score: NotRequired[float]
    summary: NotRequired[str]
    description: NotRequired[str]
    tags: NotRequired[dict[str, str]]
    frames: NotRequired[list[dict]]
    sounds: NotRequired[list[dict]]
    words: NotRequired[list[dict]]


class ReportMusic(TypedDict):
    songId: str
    mediaId: str
    name: str
    author: str
    durationSeconds: float | None
    score: NotRequired[float]
    summary: NotRequired[str]
    description: NotRequired[str]
    tags: NotRequired[dict[str, str]]
    sounds: NotRequired[list[dict]]
    words: NotRequired[list[dict]]


class MontageReport(TypedDict):
    movie: ReportMovie
    shots: list[ReportShot]
    music: list[ReportMusic]
