"""Под что собирают ролик: что заказано, откуда взять эти файлы и как они звучат для режиссёра.

Заказ называет треки — id полки или ключ загруженного организатором файла, — и всё, что здесь
происходит, это превращение списка названий в список файлов на диске, по порядку. Сама дорожка из
них склеивается в `assembly.one_track`, а звучит она одинаково, чей бы трек ни был.
"""

import logging

from montage import ffmpeg, storage
from montage.video import library

# Под каким ключом в бакете лежит трек, который организатор загрузил сам (api, `music_uploads`).
# Тем и отличается от id трека полки в одном и том же списке `params.music_tracks`.
UPLOADED_MUSIC_PREFIX = "music/"

log = logging.getLogger("montage.video.music")


def ordered(params):
    """Под что заказан ролик: названия треков в том порядке, в каком они играют, и отказ, если
    заказано то, чего нет. Одно из двух всегда None.

    Название — либо id трека полки (`library.TRACKS`), либо ключ файла, который организатор
    загрузил сам (api, `music_uploads`); что именно, разбирается по ключу — свой трек лежит под
    `music/<uuid>`. Пустой список — это «без музыки»: ролик пойдёт с живым звуком видео.

    Заказ старой формы (`music_tracks` в `params` нет вовсе) — список из одного трека:
    `music_track_key`, иначе `music_track_id`, иначе «тёплая» по умолчанию, — и повтор упавшего
    ролика, заказанного до списков, собирается ровно как собирался. Ключ и id — одно или другое: api
    не пропускал заказ с обоими, так что ключ здесь значит, что выбран он, и id не читается вовсе.
    """
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
        # Ни трек полки, ни загруженный файл — значит, списки разошлись у api и у воркера, и на
        # второй попытке будет ровно то же самое.
        return None, f"неизвестный музыкальный трек «{unknown[0]}»"
    return named, None


def files(named, s3, workdir):
    """Те же треки файлами на диске, в том же порядке, — и отказ, если своего файла нет в бакете
    или он не читается.

    Трек полки лежит рядом с воркером, свой качается из бакета и проходит ту же дверь, что и любая
    чужая медиа: «трек», внутри которого плейлист, отправил бы ffmpeg читать диски воркера. И в нём
    должен быть звук: api принимает под свой трек любые байты, а фотография или немой клип на месте
    музыки роняли бы сборку на каждой из трёх попыток.
    """
    files = []
    for index, one in enumerate(named):
        if one in library.TRACKS:
            files.append(library.track_file(one))
            continue
        downloaded = workdir / f"track-{index}"
        try:
            s3.download_file(storage.bucket(), one, str(downloaded))
        except s3.exceptions.ClientError as exc:
            # Пропавший файл между попытками сам не вернётся. Всё прочее — хранилище не ответило,
            # сеть, таймаут — проходит само, и это обычные повторы очереди.
            if exc.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey"):
                raise
            log.info("the uploaded track %s is not in the bucket", one)
            return None, "загруженный трек не найден"
        try:
            ffmpeg.refuse_playlists(downloaded)
            if not ffmpeg.has_sound(downloaded):
                raise RuntimeError("no audio stream in it")
        except RuntimeError as exc:
            # Unreadable, silent, or a playlist in place of music: identical on a second try.
            log.info("the uploaded track %s is no good: %s", one, exc)
            return None, "загруженный трек не читается"
        files.append(downloaded)
    return files, None


def heard(named, paths, wishes):
    """Дорожка словами для режиссёра: что играет, в каком порядке, сколько каждый трек идёт и что
    организатор пожелал к нему.

    Свой трек назван своим, а не ключом: ключ в бакете модели ничего не говорит, а платить за него
    словом в каждом запросе пришлось бы. Пустой список так и уходит пустым — это «без музыки», и
    режиссёр видит, что под ролик звучит сам материал.

    `wishes` — `params.music_wishes`: пожелание организатора его словами по названию трека из того
    же списка. Оно стоит у своего трека, и у каждого его повтора в списке, — у трека без пожелания
    нет ничего, а пожелание к треку, которого в списке нет, ни к чему не прикладывается.
    """
    return [
        {
            "track": one if one in library.TRACKS else "свой трек",
            "seconds": round(ffmpeg.duration(path) or 0, 1),
            **({"wish": wishes[one]} if one in wishes else {}),
        }
        for one, path in zip(named, paths)
    ]
