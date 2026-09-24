"""Полка музыки: пять треков, которые продукт предлагает под ролик, их файлы и их ключи в бакете.

Свой трек организатора сюда не попадает никогда — он приходит из бакета и доезжает до сборки тем же
путём, просто из другого места (`music.py`).
"""

import pathlib

from montage import storage

# The library: the five tracks of the «Под что монтировать» screen
# (`design/screens/MontageMusic.html`), in its order and under its names — «Яркая», «Тёплая»,
# «Спокойная», «Ночь», «Дорога». Ids only, because the worker shows no names: which name is which
# track, who wrote it, where the file came from and under what licence is `tracks/README.md`, kept
# beside the files themselves.
TRACKS = ("bright", "warm", "calm", "night", "road")
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


def publish_tracks(s3=None):
    """The library into the bucket. Returns the ids that were actually uploaded.

    What is already there is left alone rather than written over: this runs on deploy, a track's
    file never changes under an id, and re-uploading five mp3s on every run would only spend the
    bucket's write quota. The flip side is that a track *replaced* under an id that is already
    published needs its object deleted first — `docs/jobs/montage.md`, «Треки».
    """
    s3 = s3 or storage.client()
    published = []
    for track_id in TRACKS:
        try:
            s3.head_object(Bucket=storage.bucket(), Key=track_key(track_id))
            continue
        except s3.exceptions.ClientError:
            # Nothing under this key — or a bucket that refused to answer, and then the upload
            # below is refused the same way and says so. Storage that cannot be reached at all is
            # not a `ClientError` and never gets here: it comes out of `head_object`, and the one
            # caller (`python -m montage tracks`) turns it into a line rather than a traceback.
            pass
        storage.upload(s3, track_file(track_id), track_key(track_id), "audio/mpeg")
        published.append(track_id)
    return published
