import collections
import dataclasses
import json
import logging
import os
import pathlib
import tempfile
import urllib.request

from montage import ffmpeg, journal, photo, storage
from montage.media import preview, screening, speech, understanding
from montage.video import assembly, director, layout, library, music, rhythm, selection, story

MIN_ITEMS = 3

API_TIMEOUT_SECONDS = 10

log = logging.getLogger("montage.montage")


def montage(conn, ref):
    row = conn.execute(
        "SELECT memory_id, params, status FROM montages WHERE id = %s", (ref,)
    ).fetchone()
    if row is None:
        log.info("montage %s is gone, nothing to assemble", ref)
        return
    memory_id, params, status = row
    if status == "ready":
        log.info("montage %s is ready already, nothing to assemble", ref)
        return
    params = params or {}
    conn.execute(
        "UPDATE montages SET status = 'processing', updated_at = now() WHERE id = %s", (ref,)
    )

    target_seconds = min(
        float(params.get("target_seconds") or selection.DEFAULT_TARGET_SECONDS),
        selection.MAX_SECONDS,
    )
    ordered_music, refusal = music.ordered(params)
    if refusal:
        return _give_up_for_good(conn, ref, refusal, "internal")
    by_director = ordered_music is music.BY_DIRECTOR
    music_wishes = params.get("music_wishes") or {}
    orientation = params.get("orientation") or assembly.DEFAULT_ORIENTATION
    if orientation not in assembly.CANVASES:
        return _give_up_for_good(
            conn, ref, f"неизвестная ориентация ролика «{orientation}»", "internal"
        )
    exclude = {str(one) for one in params.get("exclude_media_ids") or ()}
    story_text = (params.get("story_text") or "").strip()

    items = conn.execute(
        """SELECT id, kind, original_key, preview_key, captured_at, created_at, size_bytes,
                  uploaded_by
           FROM media_items WHERE memory_id = %s AND status = 'ready' AND NOT hidden
           ORDER BY coalesce(captured_at, created_at), id""",
        (memory_id,),
    ).fetchall()
    if len(items) < MIN_ITEMS:
        return _give_up_for_good(conn, ref, "недостаточно материала для ролика", "no_material")

    journal.begin(conn, ref, len(items), target_seconds)
    layout.write(conn, ref, ())

    video_key = f"montages/{ref}.mp4"
    with tempfile.TemporaryDirectory() as workdir:
        workdir = pathlib.Path(workdir)
        pieces, originals = [], {}
        screened = []
        photos = sum(one[1] == "photo" for one in items)
        ready = {"photos_done": 0, "videos_done": 0}
        journal.start(
            conn,
            ref,
            "prep",
            total=len(items),
            photos=photos,
            videos=len(items) - photos,
            bytes=sum(one[6] for one in items),
            senders=dict(collections.Counter(str(one[7]) for one in items)),
        )
        video_seconds = 0.0
        for index, item in enumerate(items):
            (
                media_id,
                kind,
                original_key,
                preview_key,
                captured_at,
                created_at,
                _size_bytes,
                _uploaded_by,
            ) = item
            original = workdir / f"original-{index}"
            storage.download(original_key, original)
            originals[str(media_id)] = original
            looked_at = original
            if kind == "photo" and preview_key:
                looked_at = (
                    storage.large_copy(preview_key, workdir / f"large-{index}.jpg") or original
                )
            prepared = selection.judge(
                selection.Piece(
                    media_id=str(media_id),
                    kind=kind,
                    path=looked_at,
                    when=captured_at or created_at,
                    captured_at=captured_at,
                )
            )
            screened.append(prepared)
            if prepared.rejected == selection.UNREADABLE:
                journal.entry(
                    conn,
                    ref,
                    "prep",
                    media_id=prepared.media_id,
                    decision="skip",
                    reason=selection.UNREADABLE,
                )
            else:
                prepared = dataclasses.replace(prepared, rejected=None)
            pieces.append(prepared)
            ready[f"{kind}s_done"] += 1
            if kind == "video":
                video_seconds += prepared.duration or 0.0
            journal.progress(conn, ref, "prep", index + 1, **ready)

        survived = sum(piece.rejected is None and piece.media_id not in exclude for piece in pieces)
        if survived < MIN_ITEMS:
            return _give_up_for_good(
                conn,
                ref,
                "материал не годится для ролика: файлы не читаются или исключены",
                "no_material",
            )
        journal.finish(conn, ref, "prep", video_seconds=round(video_seconds, 1), **ready)

        undescribed = []
        earlier = analysis(conn, memory_id, pieces, screened, originals, undescribed)
        story.what_was_seen(conn, ref, pieces, earlier, undescribed)

        if by_director:
            track, under_the_director = None, music.shelf()
        else:
            tracks, refusal = music.files(ordered_music, workdir)
            tracks = tracks or ()
            track = assembly.one_track(tracks, workdir)
            under_the_director = music.heard(ordered_music, tracks, music_wishes)

        journal.start(conn, ref, "directing", total=len(pieces))
        chosen, thoughts, chosen_track = director.direct(
            conn,
            pieces,
            target_seconds,
            exclude,
            story=story_text,
            ref=ref,
            music=under_the_director,
            choose_music=by_director,
        )
        turned_down = None
        if chosen is None:
            chosen, turned_down = selection.choose(pieces, target_seconds, exclude)
            log.info(
                "montage %s takes %s piece(s) and leaves out %s",
                ref,
                len(chosen),
                dict(collections.Counter(piece.rejected for piece in turned_down)),
            )
        if not chosen:
            return _give_up_for_good(
                conn, ref, "в ролик нечего ставить: весь материал исключён", "no_material"
            )

        chosen = selection.stretch(chosen, target_seconds)

        if turned_down is not None:
            story.choices(conn, ref, chosen, turned_down, pieces)
        else:
            story.plan(conn, ref, chosen, thoughts)
        journal.finish(conn, ref, "directing")

        journal.start(conn, ref, "music", total=max(len(chosen) - 1, 0))
        if refusal:
            return _give_up_for_good(conn, ref, refusal, "track_unreadable")
        if by_director:
            played = [chosen_track or library.DEFAULT_TRACK]
            if chosen_track is None:
                log.info("montage %s: no model named a track, playing %s", ref, played[0])
            track = assembly.one_track([library.track_file(played[0])], workdir)
        else:
            played = ordered_music
        moments = ()
        if track:
            moments = rhythm.beats(track)
            chosen = rhythm.fit(chosen, moments)
        chosen = selection.cap(chosen)
        journal.appended(conn, ref, "directing", **story.parts(chosen, thoughts))
        story.sound(conn, ref, chosen, thoughts, track)
        journal.finish(
            conn,
            ref,
            "music",
            tracks=played,
            bpm=rhythm.tempo(moments),
            cut_seconds=[round(at, 2) for at in layout.starts(chosen)[1:]],
        )

        layout.write(conn, ref, chosen)

        chosen = _back_to_originals(chosen, originals, orientation, workdir)

        assembled = workdir / "montage.mp4"
        journal.start(conn, ref, "assembly", total=len(chosen), **journal.assembling("clips"))

        def on_a_clip(done):
            story.gluing(conn, ref, thoughts, done)
            journal.progress(
                conn, ref, "assembly", done, **journal.assembling("clips", done, len(chosen))
            )

        clips = assembly.assemble(
            chosen,
            orientation,
            track,
            workdir,
            assembled,
            on_clip=on_a_clip,
            on_join=lambda: journal.progress(
                conn, ref, "assembly", len(chosen), **journal.assembling("sound")
            ),
            max_seconds=selection.MAX_SECONDS,
        )
        layout.write(conn, ref, clips)
        duration = ffmpeg.duration(assembled)
        size_bytes = assembled.stat().st_size

        journal.progress(conn, ref, "assembly", len(chosen), **journal.assembling("cover"))

        opening = clips[0]
        at = min(
            max(opening.best_second - opening.start, 0.0),
            max(opening.seconds - assembly.CROSSFADE_SECONDS, 0.0),
        )
        poster_key, poster = f"montages/{ref}.jpg", workdir / "poster.jpg"
        try:
            ffmpeg.run(
                "-ss",
                f"{at:.3f}",
                "-i",
                str(assembled),
                "-frames:v",
                "1",
                "-vf",
                preview.THUMBNAIL_SCALE,
                str(poster),
            )
            storage.upload(poster, poster_key, "image/jpeg")
        except Exception as exc:
            log.warning("montage %s has no poster, leaving it without one: %s", ref, exc)
            poster_key = None

        journal.progress(conn, ref, "assembly", len(chosen), **journal.assembling("file"))
        storage.upload(assembled, video_key, "video/mp4")
        journal.finish(
            conn,
            ref,
            "assembly",
            cuts=max(len(clips) - 1, 0),
            seconds=round(duration or 0, 1),
            **journal.assembling(None),
        )

    landed = conn.execute(
        """UPDATE montages
           SET status = 'ready', video_key = %s, poster_key = %s, duration_seconds = %s,
               size_bytes = %s, error_text = null, updated_at = now()
           WHERE id = %s
           RETURNING id""",
        (video_key, poster_key, duration, size_bytes, ref),
    ).fetchone()
    if landed is None:
        log.info("montage %s went away mid-assembly, taking its video back", ref)
        for key in (video_key, poster_key):
            if key:
                storage.delete(key)
        return
    log.info("montage %s is ready: %s clips, %.1fs", ref, len(clips), duration or 0)
    _tell_the_api(ref)


def analysis(conn, memory_id, pieces, judged=None, originals=None, failed=None):
    screening.screen_memory(conn, memory_id, judged)
    for piece in pieces:
        if piece.kind == "video" and piece.rejected is None:
            speech.transcribe(conn, piece.media_id, (originals or {}).get(piece.media_id))
    return understanding.describe_missing(conn, [piece.media_id for piece in pieces], failed)


def _back_to_originals(chosen, originals, orientation, workdir):
    width, height = assembly.CANVASES[orientation]
    pieces = []
    for piece in chosen:
        original = originals[piece.media_id]
        if piece.kind == "photo":
            upright = workdir / f"upright-{piece.media_id}.jpg"
            if photo.stand_up(original, upright):
                pieces.append(dataclasses.replace(piece, path=upright))
                continue
        if piece.path != original:
            size = ffmpeg.dimensions(piece.path)
            if not (size and size[0] >= width and size[1] >= height):
                log.info(
                    "large copy of %s is %s, too small for the canvas — off the original",
                    piece.media_id,
                    size,
                )
                piece = dataclasses.replace(piece, path=original)
        pieces.append(piece)
    return pieces


def give_up(conn, ref, error_text, stopped):
    conn.execute(
        "UPDATE montages SET status = 'failed', error_text = %s, updated_at = now() WHERE id = %s",
        (error_text, ref),
    )
    journal.failed(conn, ref, stopped)


def _give_up_for_good(conn, ref, why, stopped):
    log.info("montage %s cannot be made: %s", ref, why)
    give_up(conn, ref, why, stopped)


def _tell_the_api(montage_id):
    url, token = os.environ.get("API_URL"), os.environ.get("INTERNAL_WORKER_TOKEN")
    if not (url and token):
        log.warning(
            "API_URL/INTERNAL_WORKER_TOKEN unset: montage %s is ready and nobody is told",
            montage_id,
        )
        return
    request = urllib.request.Request(
        f"{url.rstrip('/')}/internal/montages/{montage_id}/ready",
        data=json.dumps({}).encode(),
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
            response.read()
    except Exception as exc:
        log.warning("could not tell the api that montage %s is ready: %s", montage_id, exc)
