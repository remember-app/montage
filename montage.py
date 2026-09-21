"""The `montage` job: one video assembled out of a memory's material.

The organizer orders a video, the api writes a `montages` row and queues `('montage', <id>)`; from
`processing` onward the row is this worker's (see the api's migration `020.do.montages.sql`). Here
the memory's ready material is pulled out of R2 — every original, and beside each photograph the
large copy the preview already made of it — looked at frame by frame, the good moments are
cut and crossfaded into one 1080p file under the music that was ordered — or with the clips' own
live sound, when no music was ordered at all — and the row is left `ready` with its
keys on it — the video's and the cover's — then the api is knocked on so the phones get told.

This module is the order and the plumbing around it: it reads the row, downloads the material,
calls the pipeline's modules in turn and writes the result back. The modules it calls are
replaceable on their own, which is what the spec asks of the montage service (doc-03, «Сервис
монтажа собран из отдельных модулей»):

    `director`  — the film the cloud model plans out of the material, when there is a model to ask.
    `selection` — which pieces go into the video and in what order, and why the rest do not.
    `rhythm`    — how long each of them is held, so the joins land on the track's beats.
    `assembly`  — those pieces onto one canvas and into one file.

The first two answer the same question and hand back the same thing; the director is asked first
and the отбор cuts the video whenever it answers None, which is every montage made today.

The steps are architecture.md, «Montage-пайплайн», with one departure: selection (its step 4) runs
before normalisation (its step 3), so only the pieces that made the cut are ever encoded to 1080p
instead of every file the memory holds. `montages` and `media_items` belong to the api's migrations;
this module only reads and writes rows by id.
"""

import collections
import dataclasses
import json
import logging
import os
import pathlib
import tempfile
import urllib.request

import assembly
import director
import media
import rhythm
import selection

MIN_ITEMS = 3  # «Меньше 3 → сразу failed» — the api refuses the order at this number too

# Под каким ключом в бакете лежит трек, который организатор загрузил сам (api, `music_uploads`).
# Тем и отличается от id трека полки в одном и том же списке `params.music_tracks`.
UPLOADED_MUSIC_PREFIX = "music/"

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
    # Под что собирать: названия треков по порядку, ещё не файлы — качать их незачем, пока не
    # известно, есть ли из чего собирать ролик. Пустой список — заказ «без музыки», и тогда звучит
    # сам материал (ниже).
    ordered_music, refusal = _ordered_music(params)
    if refusal:
        return _give_up_for_good(conn, ref, refusal)
    # Пожелания организатора к трекам, его словами. Как и история, читает их только режиссёр:
    # дорожку они не перекраивают.
    music_note = (params.get("music_note") or "").strip()
    orientation = params.get("orientation") or assembly.DEFAULT_ORIENTATION
    if orientation not in assembly.CANVASES:
        # Same drift, same answer: the api validates the two values it may write here, so anything
        # else is a contract that moved on one side only — and it would be no better on a retry.
        return _give_up_for_good(conn, ref, f"неизвестная ориентация ролика «{orientation}»")
    # The organizer's own word on the material, marked before the order (doc-03, «Выбор медиа для
    # ролика»): two optional lists of `media_items` ids. Nothing is decided about them here — each
    # way of directing a video honours them in its own file, and both of them do (`selection.choose`
    # and `director.direct`), by the same rule: never the excluded, always the included. «Включить»
    # and «исключить» are both read as text ids here, because the floor below has to ask them too.
    include = {str(one) for one in params.get("include_media_ids") or ()}
    exclude = {str(one) for one in params.get("exclude_media_ids") or ()}
    # What the organizer said the video is about, in their own words. Nobody but the director has
    # anything to do with it: the algorithm cannot read.
    story = (params.get("story_text") or "").strip()

    items = conn.execute(
        # Ordered the way the gallery orders the same rows (api, `src/media/gallery.ts`): the
        # finished video runs in the order the memory is already read in.
        """SELECT id, kind, original_key, preview_key, captured_at, created_at
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
        judged, originals = [], {}
        for index, item in enumerate(items):
            media_id, kind, original_key, preview_key, captured_at, created_at = item
            original = workdir / f"original-{index}"
            # A download that fails is left fatal on purpose: the bytes may well arrive next time,
            # and swallowing an unreachable bucket would quietly cut a video out of half a memory.
            s3.download_file(media.bucket(), original_key, str(original))
            originals[str(media_id)] = original
            # A photograph is judged on the large copy and not on the forty megapixels it was
            # uploaded at — the same file the piece will be cut from below (`_back_to_originals`).
            looked_at = original
            if kind == "photo" and preview_key:
                looked_at = media.large_copy(
                    s3, preview_key, workdir / f"large-{index}.jpg"
                ) or original
            judged.append(selection.judge(selection.Piece(
                media_id=str(media_id), kind=kind, path=looked_at,
                when=captured_at or created_at, captured_at=captured_at,
            )))

        # The floor again, now against what survived. The count above was of rows; files drop out
        # here for being unreadable or having no usable frame in them, and three rows that come to
        # one still would otherwise be assembled into a three-second «ролик» against a sixty-second
        # order — the very thing the floor is there to refuse.
        #
        # A piece the organizer asked for by name counts among the survivors even when the отсев
        # turned it down: «включить» beats the algorithm in `selection.choose` and in
        # `director.direct` alike, and a floor that did not know the word would quietly overrule the
        # person before either of them was reached (doc-03, «Кадры, отмеченные включить, есть в
        # ролике»). The one thing it cannot overrule is the same one they cannot: a file with no
        # frame in it at all, which is an absence and not a judgement.
        #
        # A piece struck out of the video counts for nothing here, for the same reason: the floor is
        # what can actually end up on screen, and «исключить» is honoured by everything below this
        # line. Four good files with two of them struck out are two pieces, and a two-piece montage
        # against a minute-long order is exactly what the floor refuses.
        survived = sum(
            piece.media_id not in exclude
            and (piece.rejected is None
                 or (piece.rejected != selection.UNREADABLE and piece.media_id in include))
            for piece in judged
        )
        if survived < MIN_ITEMS:
            return _give_up_for_good(
                conn, ref,
                "материал не годится для ролика: файлы не читаются, кадры испорчены или исключены",
            )

        # Музыка раньше режиссёра: под что собирают ролик — часть того, из чего он строит план.
        tracks, refusal = _music_files(ordered_music, s3, workdir)
        if refusal:
            return _give_up_for_good(conn, ref, refusal)
        track = assembly.one_track(tracks, workdir)

        # The director first, and the отбор whenever there is no director: no cloud model, nothing
        # described for it to read, or an answer with nothing of ours in it (`director.direct`).
        chosen = director.direct(
            conn, judged, target_seconds, include, exclude, story=story, ref=ref,
            music=_music_heard(ordered_music, tracks), music_note=music_note,
        )
        if chosen is None:
            chosen, rejected = selection.choose(judged, target_seconds, include, exclude)
            log.info("montage %s takes %s piece(s) and leaves out %s", ref, len(chosen),
                     dict(collections.Counter(piece.rejected for piece in rejected)))
        if not chosen:
            # Only reachable through the order itself: everything the memory holds is on the
            # organizer's «исключить» list. There is no video to make of nothing, and the same list
            # would be there on a retry.
            return _give_up_for_good(conn, ref, "в ролик нечего ставить: весь материал исключён")

        # Both ways of directing stop at the piece that would not fit the order, so what comes back
        # from either is usually shorter than the length that was asked for. Here the clips already
        # in the video are held longer until it runs that long — one step after both of them, so
        # that no way of directing a video can go round the ordered length. Музыки это не касается:
        # ролик набирает заказанную длину и под дорожку, и с живым звуком.
        chosen = selection.stretch(chosen, target_seconds)

        # The pieces are chosen and in their places by now — all that is settled here is how long
        # each is held, so that the join lands on a beat. A track with no tempo leaves them as they
        # are, and so does a montage with no music at all: биты ищутся по всей собранной дорожке —
        # по всему списку треков разом, — а когда дорожки нет, куски держатся ровно так, как их
        # выбрали отсев или режиссёр.
        if track:
            chosen = rhythm.fit(chosen, rhythm.beats(track))

        # Which piece stands where in the video is settled by this line and nothing below moves it:
        # `_back_to_originals` swaps a file and `assembly` encodes what it is given. So the layout
        # is written here rather than at the end — assembling a montage takes minutes, and the
        # place of a frame in it is a thing to show while they are running, not after.
        _write_layout(conn, ref, chosen)

        # And only now, for the photographs that actually made the cut, whether the copy they were
        # judged on is also the file the piece is cut from.
        chosen = _back_to_originals(chosen, originals, orientation)

        assembled = workdir / "montage.mp4"
        clips = assembly.assemble(chosen, orientation, track, workdir, assembled)
        # The same layout again, off the clips as they were really encoded. `assemble` measures
        # every normalised piece instead of believing the length it asked for, and a clip whose
        # picture is shorter than its container claims comes back shorter — the crossfade chain is
        # stepped along by those measured numbers, so they are also where the frames actually are.
        _write_layout(conn, ref, clips)
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


def _ordered_music(params):
    """Под что заказан ролик: названия треков в том порядке, в каком они играют, и отказ, если
    заказано то, чего нет. Одно из двух всегда None.

    Название — либо id трека полки (`assembly.TRACKS`), либо ключ файла, который организатор
    загрузил сам (api, `music_uploads`); что именно, разбирается по ключу — свой трек лежит под
    `music/<uuid>`. Пустой список — это «без музыки»: ролик пойдёт с живым звуком видео.

    Заказа старой формы (`music_tracks` в `params` нет вовсе) это не касается: там по-прежнему
    читаются `music_track_key`, `music_track_id` и «тёплая» по умолчанию, и повтор упавшего ролика,
    заказанного до списков, собирается ровно как собирался. Ключ и id — одно или другое: api не
    пропускает заказ с обоими, так что ключ здесь значит, что выбран он, и id не читается вовсе.
    """
    tracks = params.get("music_tracks")
    if tracks is None:
        key = params.get("music_track_key")
        if key:
            return [str(key)], None
        track_id = params.get("music_track_id") or assembly.DEFAULT_TRACK
        if track_id not in assembly.TRACKS:
            # The api refuses an unknown id at the order, so this is the two lists having drifted
            # apart. Retrying would ask for the same missing track three times.
            return None, f"неизвестный музыкальный трек «{track_id}»"
        return [track_id], None
    named = [str(one) for one in tracks]
    unknown = [
        one for one in named
        if one not in assembly.TRACKS and not one.startswith(UPLOADED_MUSIC_PREFIX)
    ]
    if unknown:
        # Ни трек полки, ни загруженный файл — значит, списки разошлись у api и у воркера, и на
        # второй попытке будет ровно то же самое.
        return None, f"неизвестный музыкальный трек «{unknown[0]}»"
    return named, None


def _music_files(named, s3, workdir):
    """Те же треки файлами на диске, в том же порядке, — и отказ, если чей-то файл не читается.

    Трек полки лежит рядом с воркером, свой качается из бакета и проходит ту же дверь, что и любая
    чужая медиа: «трек», внутри которого плейлист, отправил бы ffmpeg читать диски воркера.
    """
    files = []
    for index, one in enumerate(named):
        if one in assembly.TRACKS:
            files.append(assembly.track_file(one))
            continue
        downloaded = workdir / f"track-{index}"
        s3.download_file(media.bucket(), one, str(downloaded))
        try:
            media._refuse_playlists(downloaded)
        except RuntimeError as exc:
            # Unreadable, or a playlist where the music should be: identical on a second try.
            log.info("the uploaded track %s is no good: %s", one, exc)
            return None, "загруженный трек не читается"
        files.append(downloaded)
    return files, None


def _music_heard(named, files):
    """Дорожка словами для режиссёра: что играет, в каком порядке и сколько каждый трек идёт.

    Свой трек назван своим, а не ключом: ключ в бакете модели ничего не говорит, а платить за него
    словом в каждом запросе пришлось бы. Пустой список так и уходит пустым — это «без музыки», и
    режиссёр видит, что под ролик звучит сам материал.
    """
    return [
        {"track": one if one in assembly.TRACKS else "свой трек",
         "seconds": round(media._duration(path) or 0, 1)}
        for one, path in zip(named, files)
    ]


def _back_to_originals(chosen, originals, orientation):
    """The chosen pieces again, each photograph put back on its original where the large copy it
    was judged on is too small for the canvas.

    Everything above this line has been looking at the 2048px JPEG the preview already made of every
    photo (`media.media_preview`), and the canvas is 1920×1080 either way up, so that copy almost
    always has pixels to spare: cutting the piece from it instead of from the original is the same
    picture off a tenth of the bytes and a twentieth of the work. «Big enough» is measured and not
    assumed — the cap is on the long side, so a panorama's copy can come back shorter than the
    canvas, and a photo smaller than 2048 to begin with was never enlarged. Anything under the
    canvas in either direction would have to be blown up, and then the original is the sharper file
    to cut from — even though the отсев's own verdict on it was made on the copy, which is a
    judgement about the picture and not about how many pixels it is stored in.

    Videos are never on a copy to begin with: their preview is a 720p re-encode, which is half the
    height the montage is assembled at and smoothed over exactly where sharpness is measured.
    """
    width, height = assembly.CANVASES[orientation]
    pieces = []
    for piece in chosen:
        original = originals[piece.media_id]
        if piece.path != original:
            size = media._dimensions(piece.path)
            if not (size and size[0] >= width and size[1] >= height):
                log.info("large copy of %s is %s, too small for the canvas — off the original",
                         piece.media_id, size)
                piece = dataclasses.replace(piece, path=original)
        pieces.append(piece)
    return pieces


def _write_layout(conn, ref, pieces):
    """Where each piece stands in this montage, into `montage_frames` — in place of whatever an
    earlier assembly of the same order left there.

    The table is the api's (its migration `100.do.frame-info.sql`) and this is the only thing that
    writes it: the api reads it to answer «в каких роликах стоит этот кадр и на какой секунде» on
    the frame sheet (doc-02), and it asks only about montages that are already `ready` — so rows
    under a video that then failed to assemble are seen by nobody until it is made again. One row
    per *appearance* rather than per frame, which is why the key carries the second: a file may
    legitimately stand in a video twice — a clip cut in two, a photograph that comes back at the
    end. The whole of m-24 and its neighbours read this one table; a montage's layout written down
    a second time somewhere else would be two answers to the same question.

    The seconds are counted the way `assembly` lays the chain out: each join eats half a second of
    the total, so a piece starts where the one before it ends, less the crossfade the two share.

    Nothing here may cost the montage. A video that takes minutes to assemble is not thrown away
    because an insert did not land, so a failure is a line in the log and the assembly goes on —
    the frame sheet would then show this montage under none of its frames, which is a montage
    missing from a list and not a montage missing.
    """
    at, rows = 0.0, []
    for piece in pieces:
        rows.append((ref, piece.media_id, at, piece.seconds))
        at += piece.seconds - assembly.CROSSFADE_SECONDS
    try:
        # Both statements or neither, on a connection that otherwise commits each on its own: this
        # layout is read while the video is still being assembled, and between the two an emptied
        # table reads as «этого кадра нет ни в одном ролике».
        with conn.transaction():
            conn.execute("DELETE FROM montage_frames WHERE montage_id = %s", (ref,))
            with conn.cursor() as cursor:
                cursor.executemany(
                    """INSERT INTO montage_frames
                         (montage_id, media_id, start_seconds, duration_seconds)
                       VALUES (%s, %s, %s, %s)""",
                    rows,
                )
    except Exception as exc:
        log.warning("montage %s: its layout was not written, assembling it anyway: %s", ref, exc)


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
