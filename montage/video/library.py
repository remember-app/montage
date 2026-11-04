import pathlib

from montage import storage

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
DEFAULT_TRACK = "warm"

TRACKS_DIR = pathlib.Path(__file__).parent / "tracks"


def track_file(track_id):
    return TRACKS_DIR / f"{track_id}.mp3"


def track_key(track_id):
    return f"tracks/{track_id}.mp3"


def publish_tracks():
    published = []
    for track_id in TRACKS:
        if storage.exists(track_key(track_id)):
            continue
        storage.upload(track_file(track_id), track_key(track_id), "audio/mpeg")
        published.append(track_id)
    return published
