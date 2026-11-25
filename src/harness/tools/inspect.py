import json
import pathlib
import tempfile

from src.harness.tools.context import ToolContext
from src.integrations import ffmpeg, storage
from src.integrations.ai import Blob, ToolCall, ToolResult

EXCERPT_SECONDS = 20


def run(context: ToolContext, call: ToolCall) -> ToolResult:
    ids = call.arguments.get("mediaIds")
    if not isinstance(ids, list) or not ids or any(not isinstance(one, str) for one in ids):
        return _error(call, "mediaIds must be a non-empty array of strings")
    fresh, seen = [], set()
    for media_id in ids:
        song = context.songs.get(media_id)
        if song is None:
            return _error(
                call, f"{media_id} is not a selected song; this level listens to songs only"
            )
        if media_id in context.inspected or media_id in seen:
            continue
        seen.add(media_id)
        fresh.append(song)
    if not fresh:
        return ToolResult(
            call.id,
            json.dumps(
                {"ok": True, "loaded": [], "note": "already in the conversation"},
                ensure_ascii=False,
            ),
            name=call.name,
        )
    if len(context.inspected) + len(fresh) > context.policy.max_distinct_tracks:
        return _error(
            call, f"at most {context.policy.max_distinct_tracks} different tracks per montage"
        )

    blobs, loaded = [], []
    with tempfile.TemporaryDirectory() as directory:
        for song in fresh:
            source = pathlib.Path(directory) / f"source-{song.song_id}"
            clip = pathlib.Path(directory) / f"clip-{song.song_id}.m4a"
            storage.download(song.key, source)
            start = min(30.0, max(0.0, (song.duration or 0) / 3))
            ffmpeg.run(
                "-ss",
                f"{start:.3f}",
                "-t",
                str(EXCERPT_SECONDS),
                "-i",
                str(source),
                "-c:a",
                "aac",
                "-b:a",
                "64k",
                str(clip),
            )
            blobs.append(Blob("audio/mp4", clip.read_bytes()))
            loaded.append(
                {
                    "songId": song.song_id,
                    "mediaId": song.media_id,
                    "name": song.name,
                    "durationSeconds": song.duration,
                    "excerptSeconds": EXCERPT_SECONDS,
                }
            )
    context.inspected.update(song.media_id for song in fresh)
    return ToolResult(
        call.id,
        json.dumps({"ok": True, "loaded": loaded}, ensure_ascii=False),
        tuple(blobs),
        name=call.name,
    )


def _error(call: ToolCall, message: str) -> ToolResult:
    return ToolResult(
        call.id,
        json.dumps({"ok": False, "error": message}, ensure_ascii=False),
        is_error=True,
        name=call.name,
    )


if __name__ == "__main__":
    from src.harness.policies import LOW
    from src.harness.tools.context import ToolContext
    from src.movie.preparation import Order, SongRow

    def song(index: int) -> SongRow:
        return SongRow(
            f"song-{index}", f"media-{index}", f"track {index}", "author", 2.0, {}, "key", "mp3"
        )

    songs = [song(index) for index in range(1, 5)]
    order = Order({}, {}, [], songs)
    context = ToolContext(None, None, 0, LOW, order, {}, {entry.media_id: entry for entry in songs})

    original = storage.download
    original_run = ffmpeg.run
    storage.download = lambda key, path: path.write_bytes(b"audio")
    ffmpeg.run = lambda *args: pathlib.Path(args[-1]).write_bytes(b"clip")
    try:
        first = run(context, ToolCall("c1", "inspect", {"mediaIds": ["media-1", "media-2"]}))
        assert json.loads(first.text)["ok"] and len(first.blobs) == 2
        again = run(context, ToolCall("c2", "inspect", {"mediaIds": ["media-1"]}))
        assert json.loads(again.text)["loaded"] == []
        third = run(context, ToolCall("c3", "inspect", {"mediaIds": ["media-3", "media-4"]}))
        assert third.is_error, "the fourth different track must be refused"
        shot = run(context, ToolCall("c4", "inspect", {"mediaIds": ["unknown"]}))
        assert shot.is_error
    finally:
        storage.download = original
        ffmpeg.run = original_run
    print("ok")
