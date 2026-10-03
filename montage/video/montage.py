"""The `montage` job: one video assembled out of a memory's material.

The organizer orders a video, the api writes a `montages` row and queues `('montage', <id>)`; from
`processing` onward the row is this worker's (see the api's migration `020.do.montages.sql`). Here
the memory's ready material is pulled out of R2 — every original, and beside each photograph the
large copy the preview already made of it — looked at frame by frame, the good moments are
cut and crossfaded into one 1080p file under the music that was ordered, each piece heard with
it, with its own sound or with both, as it was directed — or with the clips' own live sound, when
no music was ordered at all — and the row is left `ready` with its
keys on it — the video's and the cover's — then the api is knocked on so the phones get told.

This module is the order and the plumbing around it: it reads the row, downloads the material,
calls the pipeline's modules in turn and writes the result back. The modules it calls are
replaceable on their own, which is what the spec asks of the montage service (doc-03, «Сервис
монтажа собран из отдельных модулей»):

    `music`     — what the order asks to play, as files on disk and as words for the director.
    `director`  — the film the cloud model plans out of the material, when there is a model to ask.
    `selection` — which pieces go into the video and in what order, and why the rest do not.
    `rhythm`    — how long each of them is held, so the joins land on the track's beats.
    `assembly`  — those pieces onto one canvas and into one file.
    `story`     — что из всего этого рассказать в журнале монтажа.
    `layout`    — где в готовом ролике стоит каждый взятый кадр.

The first two of the middle three answer the same question and hand back the same thing; the
director is asked first and the отбор cuts the video whenever it answers None, which is every
montage made today.

**Отсева внутри сборки нет** (doc-03, «What it doesn't», человек 2026-10-01): кадры, оставленные
организатором при выборе, равны, и «обязательно включить» больше нет — выбирает режиссёр, из всех.
Взять нельзя только файл, в котором нет ни кадра, ни звука. Живёт это в журнале так: этапы ролика —
`prep`, `analysis`, `directing`, `music`, `assembly` (контракт api, `docs/journal.md`), и говорит о
них `journal.py`.

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

from montage import ffmpeg, journal, photo, storage
from montage.media import preview, understanding
from montage.video import assembly, director, layout, music, rhythm, selection, story

MIN_ITEMS = 3  # «Меньше 3 → сразу failed» — the api refuses the order at this number too

API_TIMEOUT_SECONDS = 10

log = logging.getLogger("montage.montage")


def montage(conn, ref):
    """One ordered video: the memory's material out of R2, the assembled file back in, the row
    to `ready` — and then the api, so the phones hear about it."""
    row = conn.execute(
        "SELECT memory_id, params, status FROM montages WHERE id = %s", (ref,)
    ).fetchone()
    if row is None:
        # The memory was deleted while the job waited, and the montage with it. Nothing to assemble
        # and nothing wrong: retrying would only fail three times.
        log.info("montage %s is gone, nothing to assemble", ref)
        return
    memory_id, params, status = row
    if status == "ready":
        # Its job run a second time: the worker that made it died between marking the video ready
        # and marking the job done (`worker.reap`). Assembling it again would swap the video the
        # organizer may have watched already for another cut of it, and push it twice. No order of
        # theirs comes here: «повторить» only takes a failed one (the api, `montages.ts`).
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
    # Под что собирать: названия треков по порядку, ещё не файлы — качать их незачем, пока не
    # известно, есть ли из чего собирать ролик. Пустой список — заказ «без музыки», и тогда звучит
    # сам материал (ниже).
    ordered_music, refusal = music.ordered(params)
    if refusal:
        # Не трек, который не открылся, а название, которого нет ни в полке, ни среди загруженных:
        # списки разошлись у api и у воркера. Это не «трек не читается» (`track_unreadable` — про
        # файл), а расхождение сторон, то есть `internal`. Код назван здесь явно, как и на каждом
        # другом отказе: значение по умолчанию — это «никто не подумал», а не решение.
        return _give_up_for_good(conn, ref, refusal, "internal")
    # Пожелания организатора к трекам, его словами, — у каждого трека своё. Как и история, читает
    # их только режиссёр: дорожку они не перекраивают. Общее `music_note` старых заказов не читается
    # вовсе (docs/jobs/montage.md, «Пожелания к трекам»).
    music_wishes = params.get("music_wishes") or {}
    orientation = params.get("orientation") or assembly.DEFAULT_ORIENTATION
    if orientation not in assembly.CANVASES:
        # Same drift, same answer: the api validates the two values it may write here, so anything
        # else is a contract that moved on one side only — and it would be no better on a retry.
        return _give_up_for_good(
            conn, ref, f"неизвестная ориентация ролика «{orientation}»", "internal"
        )
    # The organizer's own word on the material, marked before the order (doc-03, «Выбор медиа для
    # ролика»): one optional list of `media_items` ids — те, что он убрал. Оставленные кадры равны:
    # «обязательно включить» больше нет, и выбирает из них режиссёр (doc-03, «What it doesn't»).
    # Список читается как текстовые id, потому что нижний порог должен спрашивать его же.
    exclude = {str(one) for one in params.get("exclude_media_ids") or ()}
    # What the organizer said the video is about, in their own words. Nobody but the director has
    # anything to do with it: the algorithm cannot read.
    story_text = (params.get("story_text") or "").strip()

    items = conn.execute(
        # Ordered the way the gallery orders the same rows (api, `src/media/gallery.ts`): the
        # finished video runs in the order the memory is already read in. A frame its owner hid is
        # not material at all: it never reaches the отбор or the director, so neither the choice nor
        # the hiding can pull it into the video — and the hiding wins (TASK-424).
        """SELECT id, kind, original_key, preview_key, captured_at, created_at, size_bytes,
                  uploaded_by
           FROM media_items WHERE memory_id = %s AND status = 'ready' AND NOT hidden
           ORDER BY coalesce(captured_at, created_at), id""",
        (memory_id,),
    ).fetchall()
    if len(items) < MIN_ITEMS:
        # Material may have been deleted between the order and here, so the api's check at the order
        # is not the last word. Not a transient failure — three retries would find the same library.
        return _give_up_for_good(conn, ref, "недостаточно материала для ролика", "no_material")

    # Журнал сборки — до первого скачанного байта: пять этапов с оценкой времени стоят на экране с
    # первой секунды, а рассказ о каждом дописывается по ходу (`journal.py`). Повтор того же ролика
    # начинает журнал с чистого листа.
    journal.begin(conn, ref, len(items), target_seconds)
    # Раскладка прошлой попытки уходит вместе с её журналом: API присоединяет места кадров к записям
    # по одному `montage_id`, и до новой раскладки (после ритма) новая попытка показывала бы кадры
    # там, где их поставила прошлая.
    layout.write(conn, ref, ())

    video_key = f"montages/{ref}.mp4"
    with tempfile.TemporaryDirectory() as workdir:
        workdir = pathlib.Path(workdir)
        pieces, originals = [], {}
        # Подготовка: скачать каждый файл заказа и сказать в журнал, если взять его нельзя.
        # Счётчики идут в итог этапа с самого начала и растут по ходу — «фото 98 из 168» видно
        # в тот же момент, когда идёт скачивание (контракт api, «prep»).
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
            # A download that fails is left fatal on purpose: the bytes may well arrive next time,
            # and swallowing an unreachable bucket would quietly cut a video out of half a memory.
            storage.download(original_key, original)
            originals[str(media_id)] = original
            # A photograph is judged on the large copy and not on the forty megapixels it was
            # uploaded at — the same file the piece will be cut from below (`_back_to_originals`).
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
            # Отсева внутри сборки нет: смазанный и тёмный кадр в ролик идёт так же, как резкий и
            # светлый — брак предлагается убрать при выборе кадров, и там решает человек (doc-03,
            # «What it doesn't»). Взять нельзя только файл, в котором нет ни кадра, ни звука: это
            # отсутствие, а не оценка, и о нём журнал говорит словами.
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
            # По файлу, а не по всей папке разом: этап идёт минутами, и «61 из 212» на экране
            # должно двигаться, пока он идёт.
            journal.progress(conn, ref, "prep", index + 1, **ready)
        journal.finish(conn, ref, "prep", video_seconds=round(video_seconds, 1), **ready)

        # The floor again, now against what the download brought. The count above was of rows;
        # files drop out here for being unreadable or having no usable frame in them, and three
        # rows that come to one still would otherwise be assembled into a three-second «ролик»
        # against a sixty-second order — the very thing the floor is there to refuse.
        #
        # A piece struck out of the video counts for nothing here, for the same reason: the floor is
        # what can actually end up on screen, and «исключить» is honoured by everything below this
        # line. Four good files with two of them struck out are two pieces, and a two-piece montage
        # against a minute-long order is exactly what the floor refuses. Everything else counts
        # equally: ни один оставленный кадр не выпадает из счёта за резкость или за свет.
        survived = sum(piece.rejected is None and piece.media_id not in exclude for piece in pieces)
        if survived < MIN_ITEMS:
            return _give_up_for_good(
                conn,
                ref,
                "материал не годится для ролика: файлы не читаются или исключены",
                "no_material",
            )

        # Разбор кадров — описание каждого, кто ещё не описан, и чтение всего, что уже есть. Кадр,
        # описанный раньше, в нейросеть повторно не уходит: `describe_missing` отдаёт его журналу
        # готовым, помеченным «описан ранее», и по этому же делению считается счёт этапа.
        earlier = understanding.describe_missing(conn, [piece.media_id for piece in pieces])
        story.what_was_seen(conn, ref, pieces, earlier)

        # Музыка раньше режиссёра: под что собирают ролик — часть того, из чего он строит план.
        tracks, refusal = music.files(ordered_music, workdir)
        if refusal:
            # Свой трек не нашёлся в хранилище или не читается — на второй попытке будет то же
            # самое, и на третьей. Идущего этапа в этот момент нет: `prep` и `analysis` уже
            # кончились, а `music` ещё не начался, — и код ложится на первый незаконченный этап
            # (`journal.failed`).
            return _give_up_for_good(conn, ref, refusal, "track_unreadable")
        track = assembly.one_track(tracks, workdir)

        # The director first, and the отбор whenever there is no director: no cloud model, nothing
        # described for it to read, or an answer with nothing of ours in it (`director.direct`).
        journal.start(conn, ref, "directing", total=len(pieces))
        # Вторым — слова режиссёра-нейросети о том, что он планировал: мысль о плане в целом и по
        # записи на каждый показанный ему кадр. Ролик они не меняют, они и есть рассказ о нём.
        chosen, thoughts = director.direct(
            conn,
            pieces,
            target_seconds,
            exclude,
            story=story_text,
            ref=ref,
            music=music.heard(ordered_music, tracks, music_wishes),
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
            # Only reachable through the order itself: everything the memory holds is on the
            # organizer's «исключить» list. There is no video to make of nothing, and the same list
            # would be there on a retry.
            return _give_up_for_good(
                conn, ref, "в ролик нечего ставить: весь материал исключён", "no_material"
            )

        # Both ways of directing stop at the piece that would not fit the order, so what comes back
        # from either is usually shorter than the length that was asked for. Here the clips already
        # in the video are held longer until it runs that long — one step after both of them, so
        # that no way of directing a video can go round the ordered length. Музыки это не касается:
        # ролик набирает заказанную длину и под дорожку, и с живым звуком.
        chosen = selection.stretch(chosen, target_seconds)

        # Кто резал ролик, тот о нём и рассказывает: отбор — своими решениями, режиссёр-нейросеть —
        # своими словами. Итог этапа один на обоих, частями истории его наполняет только второй.
        if turned_down is not None:
            story.choices(conn, ref, chosen, turned_down, pieces)
        else:
            story.plan(conn, ref, chosen, thoughts)
        # Итог этапа — части истории с тем местом в ролике, где часть стоит (контракт api,
        # «directing»). Счётчики «взял N / мимо N» воркер не пишет: их считает API по записям.
        journal.finish(conn, ref, "directing", **story.parts(chosen, thoughts))

        # Музыка: куски выбраны и стоят на своих местах, и всё, что здесь решается, — как долго
        # каждый из них держать, чтобы склейка легла на бит. Дорожки без темпа оставляет куски как
        # есть, и так же — ролик без музыки вовсе: биты ищутся по всей собранной дорожке, по всему
        # списку треков разом, — а когда дорожки нет, куски держатся ровно так, как их выбрал
        # режиссёр.
        journal.start(conn, ref, "music", total=max(len(chosen) - 1, 0))
        moments = ()
        if track:
            moments = rhythm.beats(track)
            chosen = rhythm.fit(chosen, moments)
        # Десять минут — предел любого ролика, а за него могут вывести и режиссёр, и биты. Держится
        # он один раз, после обоих: лишнее срезается с хвоста (`selection.cap`) — и до того, как
        # этап расскажет, где стоят склейки, иначе он назвал бы их там, где их в ролике нет.
        chosen = selection.cap(chosen)
        # Что в каждой части истории слышно, — по одной записи на часть, без кадра и без решения.
        story.sound(conn, ref, chosen, thoughts, track)
        # Под что подогнаны куски: треки в том порядке, в каком они играют (id полки и ключи своих,
        # ровно как в заказе — называет их приложение), темп дорожки — только когда он есть, — и
        # где в ролике стоят склейки, секундами по возрастанию: теми же, что `layout.starts` кладёт
        # на раскладку и что приложение рисует штрихами под роликом. Ролик без музыки говорит это
        # честно: список треков пуст, склейки идут по смыслу, и подгонять было не подо что.
        journal.finish(
            conn,
            ref,
            "music",
            tracks=ordered_music,
            bpm=rhythm.tempo(moments),
            cut_seconds=[round(at, 2) for at in layout.starts(chosen)[1:]],
        )

        # Which piece stands where in the video is settled by this line and nothing below moves it:
        # `_back_to_originals` swaps a file and `assembly` encodes what it is given. So the layout
        # is written here rather than at the end — assembling a montage takes minutes, and the
        # place of a frame in it is a thing to show while they are running, not after.
        layout.write(conn, ref, chosen)

        # And only now, for the photographs that actually made the cut, whether the copy they were
        # judged on is also the file the piece is cut from.
        chosen = _back_to_originals(chosen, originals, orientation, workdir)

        assembled = workdir / "montage.mp4"
        journal.start(conn, ref, "assembly", total=len(chosen), **journal.assembling("clips"))

        def on_a_clip(done):
            """Готовый кусок: склейка засчитана, мысль о ней — в журнал, процент — по их счёту."""
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
            # По готовому куску, а не по каждому движению ffmpeg: «склеено 12 из 61» — это всё, что
            # сборка может рассказать о себе, пока идёт, и этого хватает.
            on_clip=on_a_clip,
            # Куски все нарезаны — осталась одна длинная операция: свести их с дорожкой в один
            # файл. Подшаг «звук» на её месте, и процент ползёт по нему своим весом.
            on_join=lambda: journal.progress(
                conn, ref, "assembly", len(chosen), **journal.assembling("sound")
            ),
            max_seconds=selection.MAX_SECONDS,
        )
        # The same layout again, off the clips as they were really encoded. `assemble` measures
        # every normalised piece instead of believing the length it asked for, and a clip whose
        # picture is shorter than its container claims comes back shorter — the crossfade chain is
        # stepped along by those measured numbers, so they are also where the frames actually are.
        layout.write(conn, ref, clips)
        duration = ffmpeg.duration(assembled)
        # С диска, а не у бакета после заливки: в хранилище уходят ровно эти байты, а вопрос
        # бакету — ещё один запрос, который может не ответить.
        size_bytes = assembled.stat().st_size

        # Обложка и загрузка идут **до** закрытия этапа, а не после него, и это правка порядка, а
        # не добавление ключа: подшаги `cover` и `file` пишутся в итог этапа, и этап, закрытый
        # раньше, их бы уже не записал — два последних подшага не попали бы никуда. Обложка
        # вырезается из того же файла на диске, что и уходит в хранилище, так что порядок между
        # ними ничего не ломает: оба читают один и тот же готовый ролик.
        journal.progress(conn, ref, "assembly", len(chosen), **journal.assembling("cover"))

        # The still the list of videos stands on, cut off the finished file with the same scale a
        # preview thumbnail gets (`preview.media_preview`) — but not off frame zero. The montage
        # opens where its first piece was cut, and for a video that is up to half a clip *before*
        # the moment `frames.best_moment` actually scored: `start` centres the cut on that moment
        # and then clamps it back inside the file. A phone clip that opens black would hand the
        # montage its black, which is the one thing a cover may not be. The scored moment's own
        # place in the assembled file is its offset inside the opening clip, held clear of the
        # dissolve into the next one so the cover is one picture rather than two half-faded ones.
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
            # A video that took minutes to assemble is not thrown away over its cover, and retrying
            # would rebuild the whole thing for one frame. The row stays `ready` with no poster key,
            # which the api already serves as `poster_url: null` — the same as every montage made
            # before posters existed. Ролик без обложки — ролик, а не падение: подшаг `cover` всё
            # равно закрыт, потому что работа эта сделана и делать больше нечего.
            log.warning("montage %s has no poster, leaving it without one: %s", ref, exc)
            poster_key = None

        # Финальный файл уходит в хранилище последним, и это последний подшаг: после него этапу
        # больше нечего делать, и он закрывается — всеми четырьмя подшагами сделанными и процентом
        # ровно на 100.
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
        # The memory went away mid-assembly and took the row with it. Nothing else will ever name
        # these keys, so they are taken back out — same as a preview that outlives its item.
        log.info("montage %s went away mid-assembly, taking its video back", ref)
        for key in (video_key, poster_key):
            if key:
                storage.delete(key)
        return
    log.info("montage %s is ready: %s clips, %.1fs", ref, len(clips), duration or 0)
    _tell_the_api(ref)


def _back_to_originals(chosen, originals, orientation, workdir):
    """The chosen pieces again, each photograph put back on its original where the large copy it
    was judged on is too small for the canvas.

    Everything above this line has been looking at the 2048px JPEG the preview already made of every
    photo (`preview.media_preview`), and the canvas is 1920×1080 either way up, so that copy almost
    always has pixels to spare: cutting the piece from it instead of from the original is the same
    picture off a tenth of the bytes and a twentieth of the work. «Big enough» is measured and not
    assumed — the cap is on the long side, so a panorama's copy can come back shorter than the
    canvas, and a photo smaller than 2048 to begin with was never enlarged. Anything under the
    canvas in either direction would have to be blown up, and then the original is the sharper file
    to cut from.

    Videos are never on a copy to begin with: their preview is a 720p re-encode, which is half the
    height the montage is assembled at and smoothed over exactly where sharpness is measured.

    A HEIC is the exception to all of it: ffmpeg does not apply its `irot` turn, so a portrait photo
    off an iPhone cut from the original — or from a copy made before previews stood it up — lies on
    its side in the video. It is cut from the file libheif has turned instead (`photo.stand_up`),
    whatever copy it was judged on: a copy of unknown orientation is the one thing that cannot be
    told from here.
    """
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


def give_up(conn, ref, error_text, stopped="internal"):
    """Out of retries. The montage is failed too, so the organizer stops waiting on it and gets
    «повторить» under it (architecture.md, «Обработка ошибок»).

    `error_text` is shown to people — the api hands the whole row to every member of the memory —
    so it is a sentence about the video, never an exception's text.

    `stopped` — кодом в журнал, на упавший этап: что именно остановило сборку, из трёх
    (`journal.failed`). Словарь закрыт, и `internal` — всё, что не разобрано отдельно: сюда
    попадает и очередь, исчерпавшая попытки.
    """
    conn.execute(
        "UPDATE montages SET status = 'failed', error_text = %s, updated_at = now() WHERE id = %s",
        (error_text, ref),
    )
    # Журнал остаётся как есть — до какого места ролик дошёл, видно и после падения, — а этап,
    # который в этот момент шёл, помечается упавшим вместе с кодом причины. Здесь, а не на каждом
    # исключении по дороге: у ролика, которому очередь даст ещё две попытки, этап честнее оставить
    # идущим, и упавшим он становится ровно тогда, когда упавшим называют сам ролик.
    journal.failed(conn, ref, stopped)


def _give_up_for_good(conn, ref, why, stopped):
    """A refusal rather than an accident: the answer would be the same on all three attempts, so
    the montage is failed here and the job finishes cleanly instead of retrying into it."""
    log.info("montage %s cannot be made: %s", ref, why)
    give_up(conn, ref, why, stopped)


def _tell_the_api(montage_id):
    """«Готов, скажи людям». The row is already right — this call only reaches the phones, which
    the worker cannot do itself, so a knock nobody answers is a warning and not a failure."""
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
        # Raising here would retry the whole job — rebuilding a video that already exists, for a
        # push. The organizer sees the montage in the list either way, it is only the push that
        # is lost.
        log.warning("could not tell the api that montage %s is ready: %s", montage_id, exc)
