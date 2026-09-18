"""The `montage` job: one video assembled out of a memory's material.

The organizer orders a video, the api writes a `montages` row and queues `('montage', <id>)`; from
`processing` onward the row is this worker's (see the api's migration `020.do.montages.sql`). Here
the memory's ready originals are pulled out of R2, looked at frame by frame, the good moments are
cut and crossfaded into one 1080p file with a track over it, and the row is left `ready` with its
keys on it — the video's and the cover's — then the api is knocked on so the phones get told.

This module is the order and the plumbing around it: it reads the row, downloads the material,
calls the pipeline's modules in turn and writes the result back. The two modules it calls are
replaceable on their own, which is what the spec asks of the montage service (doc-03, «Сервис
монтажа собран из отдельных модулей»):

    `selection` — which pieces go into the video and in what order, and why the rest do not.
    `assembly`  — those pieces onto one canvas and into one file.

The steps are architecture.md, «Montage-пайплайн», with one departure: selection (its step 4) runs
before normalisation (its step 3), so only the pieces that made the cut are ever encoded to 1080p
instead of every file the memory holds. `montages` and `media_items` belong to the api's migrations;
this module only reads and writes rows by id.
"""

import collections
import json
import logging
import os
import pathlib
import tempfile
import urllib.request

import assembly
import media
import selection

MIN_ITEMS = 3  # «Меньше 3 → сразу failed» — the api refuses the order at this number too

API_TIMEOUT_SECONDS = 10

log = logging.getLogger("montage.montage")


def montage(conn, ref):
    """One ordered video: the memory's material out of R2, the assembled file back in, the row
    to `ready` — and then the api, so the phones hear about it."""
    row = conn.execute("SELECT memory_id, params FROM montages WHERE id = %s", (ref,)).fetchone()
    if row is None:
        # The memory was deleted while the job waited, and the montage with it. Nothing to assemble
        # and nothing wrong: retrying would only fail three times.
        log.info("montage %s is gone, nothing to assemble", ref)
        return
    memory_id, params = row
    params = params or {}
    conn.execute("UPDATE montages SET status = 'processing', updated_at = now() WHERE id = %s", (ref,))

    target_seconds = float(params.get("target_seconds") or selection.DEFAULT_TARGET_SECONDS)
    track_id = params.get("music_track_id") or assembly.DEFAULT_TRACK
    if track_id not in assembly.TRACKS:
        # The api refuses an unknown id at the order, so this is the two lists having drifted apart.
        # Retrying would ask for the same missing track three times.
        return _give_up_for_good(conn, ref, f"неизвестный музыкальный трек «{track_id}»")
    orientation = params.get("orientation") or assembly.DEFAULT_ORIENTATION
    if orientation not in assembly.CANVASES:
        # Same drift, same answer: the api validates the two values it may write here, so anything
        # else is a contract that moved on one side only — and it would be no better on a retry.
        return _give_up_for_good(conn, ref, f"неизвестная ориентация ролика «{orientation}»")
    # The organizer's own word on the material, marked before the order (doc-03, «Выбор медиа для
    # ролика»): two optional lists of `media_items` ids. Nothing is decided about them here —
    # `selection.choose` is where they are honoured, and every later way of directing a video will
    # go through the same place.
    include = params.get("include_media_ids") or ()
    exclude = params.get("exclude_media_ids") or ()

    items = conn.execute(
        # Ordered the way the gallery orders the same rows (api, `src/media/gallery.ts`): the
        # finished video runs in the order the memory is already read in.
        """SELECT id, kind, original_key, captured_at, created_at
           FROM media_items WHERE memory_id = %s AND status = 'ready'
           ORDER BY coalesce(captured_at, created_at), id""",
        (memory_id,),
    ).fetchall()
    if len(items) < MIN_ITEMS:
        # Material may have been deleted between the order and here, so the api's check at the order
        # is not the last word. Not a transient failure — three retries would find the same library.
        return _give_up_for_good(conn, ref, "недостаточно материала для ролика")

    video_key = f"montages/{ref}.mp4"
    s3 = media.storage()
    with tempfile.TemporaryDirectory() as workdir:
        workdir = pathlib.Path(workdir)
        judged = []
        for index, (media_id, kind, original_key, captured_at, created_at) in enumerate(items):
            original = workdir / f"original-{index}"
            # A download that fails is left fatal on purpose: the bytes may well arrive next time,
            # and swallowing an unreachable bucket would quietly cut a video out of half a memory.
            s3.download_file(media.bucket(), original_key, str(original))
            judged.append(selection.judge(selection.Piece(
                media_id=str(media_id), kind=kind, path=original,
                when=captured_at or created_at, captured_at=captured_at,
            )))

        # The floor again, now against what survived. The count above was of rows; files drop out
        # here for being unreadable or having no usable frame in them, and three rows that come to
        # one still would otherwise be assembled into a three-second «ролик» against a sixty-second
        # order — the very thing the floor is there to refuse.
        if sum(piece.rejected is None for piece in judged) < MIN_ITEMS:
            return _give_up_for_good(
                conn, ref, "материал не годится для ролика: файлы не читаются или кадры испорчены"
            )

        chosen, rejected = selection.choose(judged, target_seconds, include, exclude)
        log.info("montage %s takes %s piece(s) and leaves out %s", ref, len(chosen),
                 dict(collections.Counter(piece.rejected for piece in rejected)))
        if not chosen:
            # Only reachable through the order itself: everything the memory holds is on the
            # organizer's «исключить» list. There is no video to make of nothing, and the same list
            # would be there on a retry.
            return _give_up_for_good(conn, ref, "в ролик нечего ставить: весь материал исключён")

        assembled = workdir / "montage.mp4"
        clips = assembly.assemble(chosen, orientation, track_id, workdir, assembled)
        duration = media._duration(assembled)
        media._upload(s3, assembled, video_key, "video/mp4")

        # The still the list of videos stands on, cut off the finished file with the same scale a
        # preview thumbnail gets (`media.media_preview`) — but not off frame zero. The montage opens
        # where its first piece was cut, and for a video that is up to half a clip *before* the
        # moment `_best_moment` actually scored: `start` centres the cut on that moment and then
        # clamps it back inside the file. A phone clip that opens black would hand the montage its
        # black, which is the one thing a cover may not be. The scored moment's own place in the
        # assembled file is its offset inside the opening clip, held clear of the dissolve into the
        # next one so the cover is one picture rather than two half-faded ones.
        opening = clips[0]
        at = min(max(opening.best_second - opening.start, 0.0),
                 max(opening.seconds - assembly.CROSSFADE_SECONDS, 0.0))
        poster_key, poster = f"montages/{ref}.jpg", workdir / "poster.jpg"
        try:
            media._ffmpeg("-ss", f"{at:.3f}", "-i", str(assembled), "-frames:v", "1",
                          "-vf", media.THUMBNAIL_SCALE, str(poster))
            media._upload(s3, poster, poster_key, "image/jpeg")
        except Exception as exc:
            # A video that took minutes to assemble is not thrown away over its cover, and retrying
            # would rebuild the whole thing for one frame. The row stays `ready` with no poster key,
            # which the api already serves as `poster_url: null` — the same as every montage made
            # before posters existed.
            log.warning("montage %s has no poster, leaving it without one: %s", ref, exc)
            poster_key = None

    landed = conn.execute(
        """UPDATE montages
           SET status = 'ready', video_key = %s, poster_key = %s, duration_seconds = %s,
               error_text = null, updated_at = now()
           WHERE id = %s
           RETURNING id""",
        (video_key, poster_key, duration, ref),
    ).fetchone()
    if landed is None:
        # The memory went away mid-assembly and took the row with it. Nothing else will ever name
        # these keys, so they are taken back out — same as a preview that outlives its item.
        log.info("montage %s went away mid-assembly, taking its video back", ref)
        for key in (video_key, poster_key):
            if key:
                s3.delete_object(Bucket=media.bucket(), Key=key)
        return
    log.info("montage %s is ready: %s clips, %.1fs", ref, len(clips), duration or 0)
    _tell_the_api(ref)


def give_up(conn, ref, error_text):
    """Out of retries. The montage is failed too, so the organizer stops waiting on it and gets
    «повторить» under it (architecture.md, «Обработка ошибок»).

    `error_text` is shown to people — the api hands the whole row to every member of the memory —
    so it is a sentence about the video, never an exception's text.
    """
    conn.execute(
        "UPDATE montages SET status = 'failed', error_text = %s, updated_at = now() WHERE id = %s",
        (error_text, ref),
    )


def _give_up_for_good(conn, ref, why):
    """A refusal rather than an accident: the answer would be the same on all three attempts, so
    the montage is failed here and the job finishes cleanly instead of retrying into it."""
    log.info("montage %s cannot be made: %s", ref, why)
    give_up(conn, ref, why)


def _tell_the_api(montage_id):
    """«Готов, скажи людям». The row is already right — this call only reaches the phones, which
    the worker cannot do itself, so a knock nobody answers is a warning and not a failure."""
    url, token = os.environ.get("API_URL"), os.environ.get("INTERNAL_WORKER_TOKEN")
    if not (url and token):
        log.warning("API_URL/INTERNAL_WORKER_TOKEN unset: montage %s is ready and nobody is told",
                    montage_id)
        return
    request = urllib.request.Request(
        f"{url.rstrip('/')}/internal/montages/{montage_id}/ready",
        data=json.dumps({}).encode(), method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
            response.read()
    except Exception as exc:
        # Raising here would retry the whole job — rebuilding a video that already exists, for a
        # push. The organizer sees the montage in the list either way, it is only the push that
        # is lost.
        log.warning("could not tell the api that montage %s is ready: %s", montage_id, exc)
