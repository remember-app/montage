"""Рассказ о сборке ролика в журнал монтажа: что увидели на кадрах и почему ролик вышел таким.

Журнал читают вживую, пока ролик собирается, и ведётся он отсюда и только отсюда: `journal.py`
умеет его записать, а **что** записать, видно по тому, что вернули модули пайплайна. Сами они о
журнале не знают, иначе замена любого из них тащила бы за собой рассказ о нём.
"""

from montage import journal
from montage.video import selection

# Код причины, по которой кусок стоит в ролике не по чьей-то оценке, а по слову организатора:
# «включить» из заказа. Словарь причин в журнале — воркера (README api, «Запись журнала»), и это
# единственный код, которым отмечено взятое, а не пропущенное.
ORGANIZER_ASKED = "included"

# Как воспоминание само называет тех, кого узнало на кадрах: «человек-1», «животное-2»
# (`montage.ai`, промпт описания). Журналу они нужны только чтобы сказать, скольких этап разобрал.
PEOPLE_LABEL, ANIMAL_LABEL = "человек", "животное"


@journal.quiet
def what_was_seen(conn, ref, judged):
    """Этап «разбираем, что на кадрах» в журнал: описание и кто в кадре — на каждый файл, о котором
    это уже записано.

    Своей работы у этапа нет и не должно быть: кадры описывает `understanding.py` задолго до
    заказа, по одному, и второй ролик из тех же медиа ничего не описывает заново (doc-03). Этап от
    этого не исчезает с экрана — он честно говорит «описано 0 из 14», и это сегодняшнее состояние
    продукта, пока ключа от модели нет (decision-10), а не пропущенный шаг.

    Описание — слова модели, а не код: приложению их только показать. Метки «что увидел» —
    `cast_labels` воспоминания, по ним же считается и итог этапа: «37 лиц, 5 людей в главных»
    (контракт api, `faces`, `people`, `animals`). Лица — сколько раз человек попал в кадр, то есть
    сумма меток людей по всем разобранным кадрам; люди и животные — сколько их всего, по разным
    меткам: один и тот же «человек-1» на двадцати кадрах — двадцать лиц и один человек.
    """
    journal.start(conn, ref, "understanding", total=len(judged))
    seen = conn.execute(
        """SELECT media_id, description, cast_labels FROM media_understandings
           WHERE media_id = ANY(%s)""",
        ([piece.media_id for piece in judged],),
    ).fetchall()
    cast, faces = set(), 0
    for media_id, description, labels in seen:
        cast.update(labels or ())
        faces += sum(label.startswith(PEOPLE_LABEL) for label in labels or ())
        journal.entry(
            conn, ref, "understanding", media_id=media_id, thought=description, labels=labels
        )
    journal.finish(
        conn,
        ref,
        "understanding",
        described=len(seen),
        of=len(judged),
        faces=faces,
        people=sum(label.startswith(PEOPLE_LABEL) for label in cast),
        animals=sum(label.startswith(ANIMAL_LABEL) for label in cast),
    )


def choices(conn, ref, chosen, turned_down, include, judged):
    """Решения отсева в журнал: что взято в ролик, что мимо и почему.

    Пишется после набора заказанной длины (`selection.stretch`): «беру 0:04–0:07» должно называть
    тот кусок исходника, который в ролик и пойдёт, а не тот, что был до растяжки.

    Серия одинаковых кадров — одна запись «взял 1 из N» на том, кто из неё остался: три строки
    «мимо, дубль» рядом с ней рассказывают то же самое трижды, а размер серии несёт сам кусок
    (`selection._dedup`). Слово организатора видно с обеих сторон: `excluded` на вычеркнутом,
    `included` на том, что он велел взять.

    О чём уже рассказал отсев, режиссёрский этап молчит: `selection.choose` возвращает **всё**, чего
    в ролике нет, вместе со смазанным и тёмным, а у тех «мимо» своё место — первый этап, на котором
    их и отсеяли. Второе «мимо» с той же причиной — один кадр, пропущенный дважды, и завышенный
    счётчик «Мимо N» на экране. Причины этого этапа — только его собственные: дубль серии, не влез,
    вычеркнут.
    """
    told = {piece.media_id for piece in judged if piece.rejected}
    for piece in chosen:
        journal.entry(
            conn,
            ref,
            "directing",
            media_id=piece.media_id,
            decision="take",
            **_taken(piece, include),
        )
    for piece in turned_down:
        if piece.rejected == selection.DUPLICATE or piece.media_id in told:
            continue  # серия рассказана одной записью выше, отсеянный — на своём этапе
        journal.entry(
            conn, ref, "directing", media_id=piece.media_id, decision="skip", reason=piece.rejected
        )


def plan(conn, ref, chosen, thoughts, include):
    """Рассказ режиссёра-нейросети в журнал: мысль о плане в целом, а за ней — что он сказал о
    каждом кадре, который видел, порядком плана.

    Пишется там же, где решения отсева, и по той же причине — после набора заказанной длины
    (`selection.stretch`): «беру 0:04–0:07» должно называть тот кусок исходника, который в ролик и
    пойдёт. Слова здесь режиссёра, а куски наши: сколько секунд кадра взято, из какой серии он
    остался и по чьему слову он в ролике, видно отсюда, а не из плана.

    Слово и кусок связаны порядком, а не файлом: «беру» в словах стоят тем же порядком, что куски
    в `chosen` (`director.direct`), а одно видео может стоять в ролике несколькими отрезками — и у
    каждого своя мысль и свой кусок исходника.
    """
    laid = iter(chosen)
    for thought in thoughts:
        fields = dict(thought)
        if fields.get("decision") == "take":
            fields.update(_taken(next(laid), include))
        journal.entry(conn, ref, "directing", **fields)


def parts(thoughts):
    """На какие части режиссёр разложил историю — итогом этапа: сколько их и как они называются.

    По самим репликам, а не отдельным полем плана: часть у кадра уже названа, и второй счёт рядом с
    первым разошёлся бы с ним. И только по взятым: «3 части, 61 кадр» — про ролик, и часть, ни один
    кадр которой не влез в заказанную длину, в нём не стоит. Частей нет вовсе — ролик резал отсев,
    или в плане их не было, — и итог о них молчит: отсутствующий ключ экран просто не рисует.

    Названия — слова модели, и в итог они идут той же чисткой и тем же потолком, что в реплику
    (`journal.fit`): итог — `jsonb` без `CHECK`, и держать их там больше некому.
    """
    parts = {}
    for thought in thoughts:
        if thought.get("story_part") and thought.get("decision") == "take":
            parts.setdefault(thought["story_part"], journal.fit(thought).get("story_part_title"))
    if not parts:
        return {}
    return {
        "parts": len(parts),
        "part_titles": [parts[number] for number in sorted(parts) if parts[number]] or None,
    }


def _taken(piece, include):
    """Что о взятом куске знаем мы, а не режиссёр, — одинаково на обоих путях: по чьему слову он в
    ролике (`included`, поверх любой мысли модели), из какой серии остался («взял 1 из 4», размер
    едет на самом куске) и какой кусок исходника пойдёт в ролик, секундами от начала файла. У
    фотографии куска исходника нет — в ролик идёт вся.
    """
    fields = {
        "reason": ORGANIZER_ASKED if piece.media_id in include else None,
        "series_size": piece.series_size if piece.series_size > 1 else None,
    }
    if piece.kind == "video":
        fields.update(
            source_from_seconds=round(piece.start, 2),
            source_to_seconds=round(piece.start + piece.seconds, 2),
        )
    return fields
