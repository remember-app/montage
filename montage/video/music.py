import logging

from montage import ffmpeg, storage
from montage.video import library

UPLOADED_MUSIC_PREFIX = "music/"

BY_DIRECTOR = object()

log = logging.getLogger("montage.video.music")


def ordered(params):
    if params.get("music_director"):
        return BY_DIRECTOR, None
    tracks = params.get("music_tracks")
    if tracks is None:
        tracks = [
            params.get("music_track_key") or params.get("music_track_id") or library.DEFAULT_TRACK
        ]
    named = [str(one) for one in tracks]
    unknown = [
        one
        for one in named
        if one not in library.TRACKS and not one.startswith(UPLOADED_MUSIC_PREFIX)
    ]
    if unknown:
        return None, f"неизвестный музыкальный трек «{unknown[0]}»"
    return named, None


def files(named, workdir):
    files = []
    for index, one in enumerate(named):
        if one in library.TRACKS:
            files.append(library.track_file(one))
            continue
        downloaded = workdir / f"track-{index}"
        try:
            storage.download(one, downloaded)
        except Exception as exc:
            if not storage.missing(exc):
                raise
            log.info("the uploaded track %s is not in the bucket", one)
            return None, "загруженный трек не найден"
        try:
            ffmpeg.refuse_playlists(downloaded)
            if not ffmpeg.has_sound(downloaded):
                raise RuntimeError("no audio stream in it")
        except RuntimeError as exc:
            log.info("the uploaded track %s is no good: %s", one, exc)
            return None, "загруженный трек не читается"
        files.append(downloaded)
    return files, None


def heard(named, paths, wishes):
    return [
        {
            "track": one if one in library.TRACKS else "свой трек",
            "seconds": round(ffmpeg.duration(path) or 0, 1),
            **({"wish": wishes[one]} if one in wishes else {}),
        }
        for one, path in zip(named, paths)
    ]


def shelf():
    return heard(library.TRACKS, [library.track_file(one) for one in library.TRACKS], {})
