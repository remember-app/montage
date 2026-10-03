"""Рассказ о сборке ролика в журнал монтажа: что увидели на кадрах и почему ролик вышел таким.

Журнал читают вживую, пока ролик собирается, и ведётся он отсюда и только отсюда: `journal.py`
умеет его записать, а **что** записать, видно по тому, что вернули модули пайплайна. Сами они о
журнале не знают, иначе замена любого из них тащила бы за собой рассказ о нём.
"""

from montage import journal
from montage.media import understanding
from montage.video import layout, selection

# Как воспоминание само называет тех, кого узнало на кадрах: «человек-1», «животное-2»
# (`montage.ai`, промпт описания). Журналу они нужны только чтобы сказать, скольких этап разобрал.
PEOPLE_LABEL, ANIMAL_LABEL = "человек", "животное"


@journal.quiet
def what_was_seen(conn, ref, judged, earlier):
    """Этап «разбираем, что на кадрах» в журнал: описание каждого кадра — и на тот, что описан
    сейчас, и на взятый готовым.

    Описания приходят сюда готовыми: пишет их `understanding.describe_missing`, тот же вопросом и
    по одному кадру, что и задолго до заказа, и второй раз их не спрашивает (doc-03), — а по
    разделению на «сейчас» и «ранее» (`earlier`) считается счёт этапа. Кадр из `earlier` помечен в
    карточке `earlier`, «описан ранее», и приложение рисует его такой же.

    Описание — слова модели, а не код: приложению их только показать. Метки «кто увидел» —
    `cast_labels` воспоминания, по ним же считается и итог этапа: «37 лиц, 5 людей в главных»
    (контракт api, `faces`, `people`, `animals`). Лица — сколько раз человек попал в кадр, то есть
    сумма меток людей по всем разобранным кадрам; люди и животные — сколько их всего, по разным
    меткам: один и тот же «человек-1» на двадцати кадрах — двадцать лиц и один человек.
    """
    seen = conn.execute(
        """SELECT media_id, description, cast_labels, details FROM media_understandings
           WHERE media_id = ANY(%s)""",
        ([piece.media_id for piece in judged],),
    ).fetchall()
    # Сколько описано *сейчас* — по тому, что лежит в таблице, а не по тому, сколько кадров
    # оставалось описать: с ключом от модели не описывается ни один кадр, и «описано 6» на экране
    # было бы враньём (decision-10, состояние без ключа).
    described = sum(one[0] not in earlier for one in seen)
    journal.start(
        conn,
        ref,
        "analysis",
        total=len(judged) - len(earlier),
        described=described,
        earlier=len(earlier),
    )
    spoken = _spoken(conn, [piece.media_id for piece in judged])
    comments = understanding.frame_comments(conn, [piece.media_id for piece in judged])
    cast, faces = set(), 0
    for media_id, description, labels, details in seen:
        labels = labels or ()
        cast.update(labels)
        faces += sum(label.startswith(PEOPLE_LABEL) for label in labels)
        journal.entry(
            conn,
            ref,
            "analysis",
            media_id=media_id,
            thought=description,
            labels=labels,
            # Сколько человек в кадре — это его метки, а не отдельное число от модели: она и так
            # перечислила, кто в кадре, и второе поле с тем же самым разошлось бы с ним.
            details={
                **(details or {}),
                "people": sum(label.startswith(PEOPLE_LABEL) for label in labels),
                **({"earlier": True} if media_id in earlier else {}),
                **({"speech": spoken[media_id]} if media_id in spoken else {}),
                **({"comments": comments[media_id]} if comments.get(media_id) else {}),
            },
        )
    journal.finish(
        conn,
        ref,
        "analysis",
        described=described,
        earlier=len(earlier),
        faces=faces,
        people=sum(label.startswith(PEOPLE_LABEL) for label in cast),
        animals=sum(label.startswith(ANIMAL_LABEL) for label in cast),
    )


def _spoken(conn, media_ids):
    """Что сказано в клипах заказа — из своей таблицы (`media_speech`), а не у модели.

    Отдельной строкой в карточке анализа: модель видит вырезанные кадры и звука не слышит, и
    спросить её, что слышно, значило бы выдумать и показать это людям. Словами, что на самом деле
    сказано, лежит то, что расшифровал отдельный шаг пайплайна (doc-03).
    """
    return {
        media_id: transcript
        for media_id, transcript in conn.execute(
            """SELECT media_id, transcript FROM media_speech
                WHERE media_id = ANY(%s) AND transcript <> ''""",
            (list(media_ids),),
        )
    }


def choices(conn, ref, chosen, turned_down, judged):
    """Решения отбора в журнал: что взято в ролик, что мимо и почему.

    Пишется после набора заказанной длины (`selection.stretch`): «беру 0:04–0:07» должно называть
    тот кусок исходника, который в ролик и пойдёт, а не тот, что был до растяжки.

    О чём уже рассказала подготовка, режиссёрский этап молчит: `selection.choose` возвращает
    **всё**, чего в ролике нет, и у файла, который взять нельзя, своё место — первый этап, на
    котором он и назван. Второе «мимо» с той же причиной — один кадр, пропущенный дважды, и
    завышенный счётчик «Мимо N» на экране.

    Исключённого организатором в журнале нет вовсе: он убрал его сам, и напоминать о нём не о чем
    (контракт api, «Этапы»). Причины этого этапа — только его собственные: не влез.
    """
    told = {piece.media_id for piece in judged if piece.rejected}
    for piece in chosen:
        journal.entry(
            conn, ref, "directing", media_id=piece.media_id, decision="take", **_taken(piece)
        )
    for piece in turned_down:
        if piece.media_id in told or piece.rejected == selection.EXCLUDED:
            continue  # первый этап рассказал, либо организатор убрал кадр сам
        journal.entry(
            conn, ref, "directing", media_id=piece.media_id, decision="skip", reason=piece.rejected
        )


def plan(conn, ref, chosen, thoughts):
    """Рассказ режиссёра-нейросети в журнал: мысль о плане в целом, а за ней — что он сказал о
    каждом кадре, который видел, порядком плана, и «мимо» на тех, кого он сам не взял.

    Пишется там же, где решения отбора, и по той же причине — после набора заказанной длины
    (`selection.stretch`): «беру 0:04–0:07» должно называть тот кусок исходника, который в ролик
    пойдёт. Слова здесь режиссёра, а куски наши: сколько секунд кадра взято и какой кусок исходника
    пойдёт в ролик, видно отсюда, а не из плана.

    Слово и кусок связаны порядком, а не файлом: «беру» в словах стоят тем же порядком, что куски
    в `chosen` (`director.direct`), а одно видео может стоять в ролике несколькими отрезками — и у
    каждого своя мысль и свой кусок исходника.
    """
    laid = iter(chosen)
    for thought in thoughts:
        fields = dict(thought)
        if fields.get("decision") == "take":
            fields.update(_taken(next(laid)))
        journal.entry(conn, ref, "directing", **fields)


def _parts(pieces, thoughts):
    """Части истории ролика — по порядку, как их назвал режиссёр, с кусками, что в них попали.

    Один проход на всё, чем части нужны дальше: `parts` берёт отсюда названия и секунды, `sound` —
    куски и мысль режиссёра о том, с чего часть начинается. Разойтись они не могут: по одному
    тому же проходу считается и то, что стоит на полосе под роликом, и то, что в части слышно.

    По самим репликам, а не отдельным полем плана: часть у кадра уже названа, и второй счёт рядом с
    первым разошёлся бы с ним. И только по взятым: часть, ни один кадр которой не влез в
    заказанную длину, в ролике не стоит. Частей нет вовсе — ролик резал отбор, или в плане их не
    было, — и ни итога о них, ни записей музыки не выходит: отсутствующий ключ экран просто не
    рисует.

    Секунды части — те, где она стоит в ролике: с начала первого своего куска до начала того,
    что стоит за ней, а у последней — до конца ролика. Считаются так же, как раскладку
    (`layout.starts`): это одни и те же секунды, какими приложение рисует полосу под роликом.

    Названия — слова модели, и в итог они идут той же чисткой и тем же потолком, что и в реплику
    (`journal.fit`): итог — `jsonb` без `CHECK`, и держать их там больше некому.
    """
    stood, spans = layout.starts(pieces), {}
    for thought, piece, at, after in zip(
        (one for one in thoughts if one.get("decision") == "take"),
        pieces,
        stood,
        [*stood[1:], stood[-1] + pieces[-1].seconds],
    ):
        # Кусок, которому модель части не дала, в ролике стоит и ни в какую часть не входит:
        # придумывать ему номер нечем.
        if thought.get("story_part"):
            spans.setdefault(
                thought["story_part"],
                {
                    "title": journal.fit(thought).get("story_part_title"),
                    "thought": thought.get("thought"),
                    "from_seconds": at,
                    "pieces": [],
                },
            )["to_seconds"] = after
            spans[thought["story_part"]]["pieces"].append(piece)
    return spans.items()


def parts(pieces, thoughts):
    """На какие части режиссёр разложил историю — итогом этапа: части по порядку, с названием и
    тем местом в ролике, где часть стоит (контракт api, `docs/journal.md`, «directing»)."""
    spans = list(_parts(pieces, thoughts))
    if not spans:
        return {}
    return {
        "parts": [
            {
                "title": span["title"],
                "from_seconds": round(span["from_seconds"], 1),
                "to_seconds": round(span["to_seconds"], 1),
            }
            for _part, span in spans
        ]
    }


def sound(conn, ref, pieces, thoughts, track):
    """Этап «Музыка» в журнал: по одной записи на часть истории — что в ней слышно.

    Без кадра и без решения: запись идёт о части, а не о куске (контракт api, «music»). Слов
    своих у воркера здесь нет: `thought` — мысль режиссёра о том, с чего часть начинается, и
    большего сказать о звуке тут нечем.

    `track` — дорожка под роликом или None, и он решает обе доли: с дорожкой музыка идёт под всем
    роликом, а сколько в части слышно самих кадров — то, что режиссёр оставил кускам по одной
    доле (`selection.Piece.live`, `answers._live`); без дорожки ролик весь звучит живым звуком
    кадров (`assembly.assemble`), и музыки в нём нет вовсе. Доли не дополняют друг друга до
    единицы: громкая музыка и громкий зал бывают разом.
    """
    for part, span in _parts(pieces, thoughts):
        journal.entry(
            conn,
            ref,
            "music",
            story_part=part,
            story_part_title=span["title"],
            thought=span["thought"],
            details=_shares(span["pieces"], track),
        )


def _shares(pieces, track):
    """Две доли звука в части истории — по кускам, что в неё попали.

    Музыки в части столько, сколько её вообще под роликом: дорожка одна и играет через весь
    ролик, так что от части к части меняется только вторая доля. Живого звука столько, сколько
    режиссёр оставил кускам (`Piece.live`), средним по части.
    """
    if not track:
        return {"music": 0.0, "live": 1.0}
    return {"music": 1.0, "live": round(sum(one.live for one in pieces) / len(pieces), 2)}


def _taken(piece):
    """Что о взятом куске знаем мы, а не режиссёр, — одинаково на обоих путях: из какой серии он
    остался («взял 1 из 4», размер едет на самом куске) и какой кусок исходника пойдёт в ролик,
    секундами от начала файла. У фотографии куска исходника нет — в ролик идёт вся.

    Чьим словом кусок в ролике, не пишется: кадры, оставленные при выборе, равны, и «включить»
    больше ничего не закрепляет (doc-03, «What it doesn't»).
    """
    fields = {"series_size": piece.series_size if piece.series_size > 1 else None}
    if piece.kind == "video":
        fields.update(
            source_from_seconds=round(piece.start, 2),
            source_to_seconds=round(piece.start + piece.seconds, 2),
        )
    return fields


def gluing(conn, ref, thoughts, done):
    """Мысль о склейке, которая идёт сейчас, — записью без кадра и без решения.

    Карточку «Сейчас» на экране сборки приложение рисует по записи **без кадра** (контракт api,
    «assembly»), и кроме неё на сборке показать нечего: склейка не принадлежит ни одному кадру,
    а делается с кусками сразу, поэтому и `media_id` у такой записи нет.

    По одной на каждый готовый кусок — столько же, сколько раз двигается счётчик этапа, а не на
    каждое движение ffmpeg: сборка идёт минутами, и «склеено 12 из 61» на экране должно двигаться
    вместе со сборкой. Своих слов у воркера тут нет — сказать можно номер склейки и, когда режиссёр
    разложил историю по частям, название той части, в которую кусок стоит. Ролик режет отбор (частей
    нет ни у кого) — остаётся один номер, и он и двигает эту строку по ходу.
    """
    titles = [
        journal.fit(one).get("story_part_title")
        for one in thoughts
        if one.get("decision") == "take"
    ]
    title = titles[done - 1] if 0 < done <= len(titles) else None
    journal.entry(
        conn, ref, "assembly", thought=f"Склейка {done}: {title}" if title else f"Склейка {done}"
    )
