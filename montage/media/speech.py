import logging
import pathlib
import tempfile

from montage import ai, ffmpeg, storage

AUDIO_MIME = "audio/m4a"
SOUND_ONLY = ("-vn", "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "64k")

log = logging.getLogger("montage.speech")


def transcribe(conn, ref, original=None):
    if _remembered(conn, ref):
        log.info("clip %s is transcribed already, asking nobody", ref)
        return

    row = conn.execute(
        "SELECT kind, original_key FROM media_items WHERE id = %s", (ref,)
    ).fetchone()
    if row is None:
        log.info("media item %s is gone, nothing to transcribe", ref)
        return
    kind, original_key = row
    if kind != "video":
        log.info("media item %s is a %s and not a clip: nothing to transcribe", ref, kind)
        return

    brain = ai.AI(conn)
    if brain.provider is None:
        log.info("no cloud model is configured: clip %s is left untranscribed for now", ref)
        return

    with tempfile.TemporaryDirectory() as workdir:
        workdir = pathlib.Path(workdir)
        sound = workdir / "speech.m4a"
        if original is None:
            original = workdir / "original"
            storage.download(original_key, original)
        ffmpeg.refuse_playlists(original)
        if not ffmpeg.has_sound(original):
            log.info("clip %s carries no sound at all", ref)
            _remember(conn, ref, "")
            return
        ffmpeg.run("-i", str(original), *SOUND_ONLY, str(sound))
        spoken = brain.transcribe_speech(ai.Blob(AUDIO_MIME, sound.read_bytes()), ref=ref)

    _remember(conn, ref, spoken)
    log.info("clip %s said %s character(s)", ref, len(spoken))


def _remembered(conn, ref):
    return (
        conn.execute("SELECT 1 FROM media_speech WHERE media_id = %s", (ref,)).fetchone()
        is not None
    )


def _remember(conn, ref, spoken):
    landed = conn.execute(
        """INSERT INTO media_speech (media_id, transcript)
           SELECT %s, %s WHERE EXISTS (SELECT 1 FROM media_items WHERE id = %s)
           ON CONFLICT (media_id) DO NOTHING
           RETURNING media_id""",
        (ref, spoken, ref),
    ).fetchone()
    if landed is None:
        log.info(
            "nothing written for clip %s: it went away while it was being transcribed, or "
            "another run of this job got there first",
            ref,
        )
