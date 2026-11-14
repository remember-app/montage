"""Склейки под бит: режиссёр решает, что и примерно сколько, ритм ставит границы на удар."""

import bisect
import json
import logging
import math
import pathlib
import statistics
import tempfile
import uuid
from typing import cast

import psycopg
from psycopg.types.json import Jsonb

from src.integrations import ffmpeg, storage
from src.movie import editor
from src.movie.types import EditCommand, EditPlan, ShotCommand

log = logging.getLogger("movie.rhythm")

SAMPLE_RATE = 22050
HORIZON_SECONDS = 30 * 60
MIN_BEATS = 8

MIN_SECONDS = 1.5
MAX_SECONDS = 6.0
LIVE_HOLD = 0.5


def beats(path: pathlib.Path) -> tuple[float, ...]:
    """Моменты ударов трека в секундах; пусто, если ритма не разобрать."""
    try:
        import librosa

        with tempfile.TemporaryDirectory() as directory:
            wav = pathlib.Path(directory) / "track.wav"
            ffmpeg.run(
                "-i",
                str(path),
                "-t",
                str(HORIZON_SECONDS),
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                str(wav),
            )
            samples, rate = librosa.load(wav, sr=SAMPLE_RATE, mono=True)
        _, moments = librosa.beat.beat_track(y=samples, sr=rate, units="time")
    except Exception as exc:
        log.warning("cannot make out the rhythm of %s, cuts stay as directed: %s", path, exc)
        return ()
    if len(moments) < MIN_BEATS:
        log.info("%s has %s beats, too few to cut to", path, len(moments))
        return ()
    return tuple(float(moment) for moment in moments)


def grid(plan: EditPlan, song_beats: dict[str, tuple[float, ...]]) -> list[float]:
    """Моменты ударов на таймлайне ролика: бит трека, переведённый в место его куска."""
    moments = set()
    for cue in plan:
        if cue.get("type") != "music":
            continue
        track = song_beats.get(str(cue.get("mediaId")), ())
        start = float(cue.get("sourceStartSeconds") or 0)
        end = start + float(cue.get("seconds") or 0)
        at = float(cue.get("atSeconds") or 0)
        for beat in track:
            if start - 0.001 <= beat <= end + 0.001:
                moments.add(round(at + beat - start, 3))
    return sorted(moments)


def fit(plan: EditPlan, moments: list[float], durations: dict[str, float | None]) -> EditPlan:
    shots = _shots(plan)
    if len(shots) < 2 or len(moments) < MIN_BEATS:
        return plan
    beat = _beat(moments)
    if beat is None:
        return plan
    fitted: list[ShotCommand] = []
    at = 0.0
    planned = 0.0
    for index, shot in enumerate(shots[:-1]):
        following = shots[index + 1]
        seconds = _snapped(shot, following, moments, at, planned, beat, durations)
        fitted.append(_cut(shot, seconds, durations))
        at += _seconds(fitted[-1]) - editor.overlap(fitted[-1])
        planned += _seconds(shot) - editor.overlap(shot)
    last = shots[-1]
    seconds = max(_frames(_seconds(last) + planned - at), min(_seconds(last), MIN_SECONDS))
    fitted.append(_cut(last, seconds, durations))
    return _rebuild(plan, fitted)


def apply(conn: psycopg.Connection, montage_id: uuid.UUID, plan: EditPlan) -> EditPlan:
    keys = _song_keys(conn, plan)
    if not keys:
        return _store(conn, montage_id, plan, {"beats": 0, "cutsMoved": 0})
    song_beats: dict[str, tuple[float, ...]] = {}
    with tempfile.TemporaryDirectory() as directory:
        workdir = pathlib.Path(directory)
        for media_id, key in keys.items():
            path = workdir / f"song-{media_id}"
            storage.download(key, path)
            ffmpeg.refuse_playlists(path)
            song_beats[media_id] = beats(path)
    moments = grid(plan, song_beats)
    fitted = fit(plan, moments, _durations(conn, plan))
    shots, changed = _shots(plan), _shots(fitted)
    moved = sum(
        1
        for before, after in zip(shots, changed)
        if abs(_seconds(before) - _seconds(after)) > 0.001
    )
    before, after = editor.timeline_length(shots), editor.timeline_length(changed)
    log.info(
        "montage %s cut to the beat: %s moments, %s of %s shots moved, %.2fs → %.2fs",
        montage_id,
        len(moments),
        moved,
        len(shots),
        before,
        after,
    )
    try:
        editor.validate(conn, montage_id, fitted)
    except RuntimeError as exc:
        log.warning(
            "montage %s cannot keep its order on the beat, cuts stay as directed: %s",
            montage_id,
            exc,
        )
        return _store(conn, montage_id, plan, {"beats": len(moments), "cutsMoved": 0})
    return _store(conn, montage_id, fitted, {"beats": len(moments), "cutsMoved": moved})


def _snapped(
    shot: ShotCommand,
    following: ShotCommand,
    moments: list[float],
    at: float,
    planned: float,
    beat: float,
    durations: dict[str, float | None],
) -> float:
    if _live(shot) or _live(following):
        return _seconds(shot)
    half = editor.overlap(shot) / 2
    least = MIN_SECONDS
    most = min(_ceiling(shot), _source_room(shot, durations))
    if most < least:
        return _seconds(shot)
    wanted = planned + _seconds(shot) - half
    low = max(at + least - half, wanted - beat)
    high = min(at + most - half, wanted + beat)
    inside = moments[bisect.bisect_left(moments, low) : bisect.bisect_right(moments, high)]
    if not inside:
        return _seconds(shot)
    return min(inside, key=lambda moment: abs(moment - wanted)) - at + half


def _cut(shot: ShotCommand, seconds: float, durations: dict[str, float | None]) -> ShotCommand:
    seconds = _frames(seconds)
    room = _source_room(shot, durations)
    if room != math.inf:
        seconds = min(seconds, _frames(room))
    return {**shot, "seconds": round(max(seconds, 1 / editor.FPS), 3)}


def _ceiling(shot: ShotCommand) -> float:
    return MAX_SECONDS if _seconds(shot) <= MAX_SECONDS else math.inf


def _source_room(shot: ShotCommand, durations: dict[str, float | None]) -> float:
    if shot.get("kind") in ("video", "audio"):
        duration = durations.get(str(shot.get("itemId")))
        if duration:
            start = float(shot.get("sourceStartSeconds") or 0)
            return max(duration - start, 0.0)
    return math.inf


def _live(shot: ShotCommand) -> bool:
    return float(shot.get("liveLevel") or 0) >= LIVE_HOLD


def _rebuild(plan: EditPlan, fitted: list[ShotCommand]) -> EditPlan:
    result: EditPlan = []
    index = 0
    for command in plan:
        if command.get("type") == "shot":
            result.append(fitted[index])
            index += 1
        else:
            result.append(command)
    return result


def _shots(plan: EditPlan) -> list[ShotCommand]:
    return [cast(ShotCommand, command) for command in plan if command.get("type") == "shot"]


def _seconds(shot: ShotCommand) -> float:
    return float(shot.get("seconds") or 0)


def _frames(seconds: float) -> float:
    return round(seconds * editor.FPS) / editor.FPS


def _beat(moments: list[float]) -> float | None:
    steps = [after - before for before, after in zip(moments, moments[1:]) if after > before]
    return statistics.median(steps) if steps else None


def _song_keys(conn: psycopg.Connection, plan: EditPlan) -> dict[str, str]:
    ids = [uuid.UUID(str(cue["mediaId"])) for cue in plan if cue.get("type") == "music"]
    if not ids:
        return {}
    rows = conn.execute(
        "select id, storage_key from media_items where id = any(%s)", (list(set(ids)),)
    ).fetchall()
    return {str(media_id): key for media_id, key in rows}


def _durations(conn: psycopg.Connection, plan: EditPlan) -> dict[str, float | None]:
    ids = [uuid.UUID(str(command["itemId"])) for command in _shots(plan)]
    if not ids:
        return {}
    rows = conn.execute(
        """
        select pi.id, m.duration
        from project_items pi join media_items m on m.id = pi.media_id
        where pi.id = any(%s)
        """,
        (list(set(ids)),),
    ).fetchall()
    return {str(item_id): duration for item_id, duration in rows}


def _store(
    conn: psycopg.Connection, montage_id: uuid.UUID, plan: EditPlan, summary: dict
) -> EditPlan:
    landed = conn.execute(
        """
        update montages
        set manifest = jsonb_set(
                manifest, '{editingParams}',
                jsonb_build_object('commands', %s::jsonb, 'rhythm', %s::jsonb)
            )
        where id = %s
        """,
        (json.dumps(plan), Jsonb(summary), montage_id),
    ).rowcount
    if not landed:
        raise RuntimeError(f"montage {montage_id} went away while cutting to the beat")
    return plan


if __name__ == "__main__":
    moments = [round(index * 0.5, 3) for index in range(240)]

    def shot(item_id: str, seconds: float) -> ShotCommand:
        return {
            "type": "shot",
            "itemId": item_id,
            "mediaId": item_id,
            "kind": "photo",
            "sourceStartSeconds": 0.0,
            "seconds": seconds,
            "liveLevel": 0.0,
            "framing": None,
            "motion": "none",
            "transitionSeconds": 0.0,
            "role": "action",
        }

    plan: EditPlan = [shot("a", 2.3), shot("b", 2.2), shot("c", 2.4)]
    fitted = fit(plan, moments, {})
    at = 0.0
    for command in _shots(fitted)[:-1]:
        at += _seconds(command) - editor.overlap(command)
        assert abs(at - round(at / 0.5) * 0.5) < 1e-6, at
    assert abs(editor.timeline_length(_shots(fitted)) - editor.timeline_length(_shots(plan))) < 1e-6
    assert fit(plan, [], {}) == plan, "no beats leaves the plan alone"
    loud: EditPlan = [shot("a", 2.3), {**shot("b", 2.2), "liveLevel": 1.0}, shot("c", 2.4)]
    kept = _shots(fit(loud, moments, {}))
    assert (_seconds(kept[0]), _seconds(kept[1])) == (2.3, 2.2), "live sound keeps the cut"
    cue: EditCommand = {
        "type": "music",
        "songId": "s",
        "mediaId": "s",
        "sourceStartSeconds": 8.0,
        "seconds": 1.5,
        "atSeconds": 3.0,
        "gain": 1.0,
        "fadeInSeconds": 0.0,
        "fadeOutSeconds": 0.0,
    }
    assert grid([cue], {"s": (8.0, 8.5, 9.0, 9.5, 10.0)}) == [3.0, 3.5, 4.0, 4.5]
    print("ok")
