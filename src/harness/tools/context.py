import dataclasses
import uuid

import psycopg

from src.harness.policies import LevelPolicy
from src.movie.preparation import Order, ShotRow, SongRow


@dataclasses.dataclass
class ToolContext:
    conn: psycopg.Connection
    montage_id: uuid.UUID
    job_id: int
    policy: LevelPolicy
    order: Order
    shots: dict[str, ShotRow]
    songs: dict[str, SongRow]
    inspected: set[str] = dataclasses.field(default_factory=set)
    versions: list[list] = dataclasses.field(default_factory=list)


def make(
    conn: psycopg.Connection, montage_id: uuid.UUID, job_id: int, policy: LevelPolicy, order: Order
) -> ToolContext:
    return ToolContext(
        conn,
        montage_id,
        job_id,
        policy,
        order,
        {shot.item_id: shot for shot in order.shots},
        {song.media_id: song for song in order.music},
    )
