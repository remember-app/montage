"""Полка музыки: десять треков, которые продукт предлагает под ролик, их файлы и их ключи в бакете.

Свой трек организатора сюда не попадает никогда — он приходит из бакета и доезжает до сборки тем же
путём, просто из другого места (`music.py`).
"""

import pathlib

from montage import storage

# The library: the tracks of the «Под что монтировать» screen (`design/screens/MontageMusic.html`) —
# the five the screen showed, in its own order and under its names («Яркая», «Тёплая», «Спокойная»,
# «Ночь», «Дорога»), then five more of the same mood-word kind. Ids only, because the worker shows
# no names: which name is which track, who wrote it, where the file came from and under what licence
# is `tracks/README.md`, kept beside the files themselves.
TRACKS = (
    "bright",
    "warm",
    "calm",
    "night",
    "road",
    "merry",
    "tender",
    "dreamy",
    "mystery",
    "energetic",
)
DEFAULT_TRACK = "warm"  # what an order that says nothing about music means

TRACKS_DIR = pathlib.Path(__file__).parent / "tracks"


def track_file(track_id):
    """The library's file for a track. The organizer's own track never comes through here — that
    one arrives out of R2 and reaches `assemble` the same way, just from somewhere else."""
    return TRACKS_DIR / f"{track_id}.mp3"


def track_key(track_id):
    """Where the same track lives in the bucket, for «Послушать» on the «Под что монтировать»
    screen to play before any video is ordered. The montage itself never reads these objects — it
    cuts from the file on disk beside it — but that disk is this worker's alone, and the api and the
    phone have to be able to reach the music too. Same id in the key as in `TRACKS`, so the api
    builds it without knowing anything more about the library."""
    return f"tracks/{track_id}.mp3"


def publish_tracks():
    """The library into the bucket. Returns the ids that were actually uploaded.

    What is already there is left alone rather than written over: this runs on deploy, a track's
    file never changes under an id, and re-uploading five mp3s on every run would only spend the
    bucket's write quota. The flip side is that a track *replaced* under an id that is already
    published needs its object deleted first — `docs/jobs/montage.md`, «Треки».

    Storage that refuses or cannot be reached comes out of here as it is, and the one caller
    (`python -m montage tracks`) turns it into a line rather than a traceback.
    """
    published = []
    for track_id in TRACKS:
        if storage.exists(track_key(track_id)):
            continue
        storage.upload(track_file(track_id), track_key(track_id), "audio/mpeg")
        published.append(track_id)
    return published
