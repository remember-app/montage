"""The `media_speech` job: what is said in one clip, written down once and read by everyone after.

A video that becomes `ready` gets this job queued beside its preview (`media.media_preview`): the
original comes back out of the bucket, `ffmpeg` strips the sound out of it, and the gateway
(`ai.py`) puts it to whichever cloud model is configured. What comes back is one plain text on
`media_speech`, under the id of the media item — the words, and nothing but.

Who said them is deliberately not asked and deliberately not kept (decision-11): a transcript here
is never tied to a person in the frame, and there is nowhere in the table to tie it to one.

Written once and reused from then on: the job on a clip that already has its row asks nobody and
finishes, so the second and every later montage out of the same material costs nothing (doc-03,
«выходы других шагов пайплайна тоже переиспользуются»). Silence is an answer and is kept as one —
an empty text, whether the clip carries no sound at all or carries nobody speaking. What is *not*
written is the clip nobody was asked about: with no cloud model configured the job leaves no row,
so the words are really fetched the day a key arrives.

`media_items` belongs to the api's migrations; `media_speech` is this worker's own (`schema.sql`).
"""

import logging
import pathlib
import tempfile

import ai
import media

# What goes to the model: mono 16 kHz AAC — about what a transcription model listens to anyway, and
# a fraction of the bytes of the clip's own track, which matters when the file travels inline in the
# prompt, base64 and all. The mime type is the one the gateway names the upload by (`ai.Blob`).
AUDIO_MIME = "audio/m4a"
SOUND_ONLY = ("-vn", "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "64k")

log = logging.getLogger("montage.speech")


def transcribe(conn, ref):
    """One clip: the sound out of it, its words into `media_speech` — once, and never again."""
    if _remembered(conn, ref):
        log.info("clip %s is transcribed already, asking nobody", ref)
        return

    row = conn.execute(
        "SELECT kind, original_key FROM media_items WHERE id = %s", (ref,)
    ).fetchone()
    if row is None:
        # Deleted while the job waited — a member left and took their material with them. Nothing
        # to listen to and nothing wrong: retrying would only fail three times.
        log.info("media item %s is gone, nothing to transcribe", ref)
        return
    kind, original_key = row
    if kind != "video":
        # Only a clip has speech in it, and nothing queues this job for a photo — so this is a job
        # somebody put in by hand, and it is answered rather than retried.
        log.info("media item %s is a %s and not a clip: nothing to transcribe", ref, kind)
        return

    brain = ai.AI(conn)
    if brain.provider is None:
        # Asked before the download rather than after it: with no key in the environment — which is
        # today — every clip would otherwise be pulled out of the bucket and re-encoded for a call
        # that is skipped anyway.
        log.info("no cloud model is configured: clip %s is left untranscribed for now", ref)
        return

    with tempfile.TemporaryDirectory() as workdir:
        workdir = pathlib.Path(workdir)
        original, sound = workdir / "original", workdir / "speech.m4a"
        media.storage().download_file(media.bucket(), original_key, str(original))
        # Before ffmpeg is pointed at it: these bytes came off somebody's phone, and a playlist
        # dressed up as a clip would have ffmpeg read this worker's own disk into a prompt.
        media._refuse_playlists(original)
        if not media._has_sound(original):
            # A clip shot with the microphone off, or one whose track the phone never wrote. There
            # is nothing to send and nothing to invent, and the empty row settles it for good.
            log.info("clip %s carries no sound at all", ref)
            _remember(conn, ref, "")
            return
        media._ffmpeg("-i", str(original), *SOUND_ONLY, str(sound))
        spoken = brain.transcribe_speech(ai.Blob(AUDIO_MIME, sound.read_bytes()), ref=ref)

    _remember(conn, ref, spoken)
    log.info("clip %s said %s character(s)", ref, len(spoken))


def _remembered(conn, ref):
    return conn.execute(
        "SELECT 1 FROM media_speech WHERE media_id = %s", (ref,)
    ).fetchone() is not None


def _remember(conn, ref, spoken):
    """The words under the clip they came out of — unless the clip has gone in the meantime.

    It may well have: the model listens for as long as it likes, and a member can leave and take
    their material with them while it does. What they said must not stay behind in a table nothing
    will ever look at again — hence the `EXISTS`, which is the same care `media_preview` takes of a
    preview that outlived its item. `ON CONFLICT` because a second copy of the job is not an error:
    the first answer stands.
    """
    landed = conn.execute(
        """INSERT INTO media_speech (media_id, transcript)
           SELECT %s, %s WHERE EXISTS (SELECT 1 FROM media_items WHERE id = %s)
           ON CONFLICT (media_id) DO NOTHING
           RETURNING media_id""",
        (ref, spoken, ref),
    ).fetchone()
    if landed is None:
        log.info("nothing written for clip %s: it went away while it was being transcribed, or "
                 "another run of this job got there first", ref)
