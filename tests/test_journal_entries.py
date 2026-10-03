"""Что воркер кладёт в журнал: реплика на кадр и итог этапа — `docs/jobs/montage-journal.md`.

Здесь не про порядок этапов, а про их содержание: что рассказала подготовка, что рассказал
режиссёр-нейросеть и что из его слов вообще доезжает до базы.

Warning: эти тесты чистят таблицу `jobs` той базы, на которую смотрит DATABASE_URL, и пишут в
бакет, на который смотрит R2_ENDPOINT.
"""

import dataclasses
import datetime
import math
import uuid

from psycopg.types.json import Jsonb

from montage import journal, worker
from montage.media import understanding
from montage.video import selection, story

from pieces import picture
from recorded import planning_over
from rows import add_item, entries_of, journal_ready, order, read_montage, ready_material, stages_of


def test_the_analysis_stage_tells_what_was_seen_and_who_was_in_frame(conn, memory):
    """Разбор кадров пересказывает то, что записано в `media_understandings`: описание словами
    модели, метки «кто в кадре» и — полями — где снято, в каком свете, с каким настроением и
    планом. Итог — сколько описано, сколько взято готовыми, сколько лиц на кадрах и скольких
    воспоминание узнало: «37 лиц, 5 людей в главных» на экране «Ролик готов».

    Лица и люди — разные числа, и разойтись они должны именно здесь: «человек-1» стоит на обоих
    описанных кадрах, значит, лиц три, а людей двое.
    """
    journal_ready(conn)
    memory_id, _ = memory
    montage_id = order(conn, memory, {})
    described, _ = add_item(conn, memory, "photo", status="ready")
    again, _ = add_item(conn, memory, "photo", status="ready")
    silent, _ = add_item(conn, memory, "photo", status="ready")
    for media_id, description, cast, fields in (
        (
            described,
            "Бабушка смеётся, Лиза обнимает её",
            ["человек-1", "человек-2", "животное-1"],
            {"place": "Гостиная", "light": "Вечер, тёплые гирлянды", "mood": "Нежно"},
        ),
        (again, "Бабушка у окна", ["человек-1"], {"place": "Гостиная", "shot": "Средний"}),
    ):
        conn.execute(
            """INSERT INTO media_understandings (media_id, memory_id, description, cast_labels,
                                                 details)
               VALUES (%s, %s, %s, %s, %s)""",
            (media_id, str(memory_id), description, cast, Jsonb(fields)),
        )
    judged = [selection.Piece(media_id=one) for one in (described, again, silent)]
    earlier = {described, again}

    journal.begin(conn, montage_id, len(judged), 60)
    story.what_was_seen(conn, montage_id, judged, earlier)

    # Порядком записей этап ничего не рассказывает — они про разные кадры и ложатся одной пачкой.
    written = {one.media_id: one for one in entries_of(conn, montage_id, "analysis")}
    assert {one.media_id: (one.thought, one.labels) for one in written.values()} == {
        described: ("Бабушка смеётся, Лиза обнимает её", ["человек-1", "человек-2", "животное-1"]),
        again: ("Бабушка у окна", ["человек-1"]),
    }
    assert written[described].details == {
        "place": "Гостиная",
        "light": "Вечер, тёплые гирлянды",
        "mood": "Нежно",
        "earlier": True,
        "people": 2,
    }, "описание полями плюс «описан ранее» и сколько людей в кадре"
    assert written[again].details["shot"] == "Средний"
    assert stages_of(conn, montage_id)["analysis"]["outcome"] == {
        # Ни одного: третий кадр не описан — без модели (decision-10) или потому, что описать его
        # нечем. Считается по тому, что описано на самом деле, а не по тому, сколько оставалось.
        "described": 0,
        "earlier": 2,
        "faces": 3,
        "people": 2,
        "animals": 1,
    }
    assert stages_of(conn, montage_id)["analysis"]["total"] == 1, (
        "а счётчик стоит на том, сколько кадров пришлось бы описать"
    )


def test_a_frame_described_now_is_not_marked_as_described_earlier(conn, memory):
    """«Описан ранее» — только про то описание, которое взято готовым. Кадр, описанный этим же
    этапом, помеченства не должен: иначе на экране «описано 1 сейчас, ещё 1 ранее» стояло бы на
    карточке, которую только что описали, и не сказало бы, за что заплатили сейчас.
    """
    journal_ready(conn)
    montage_id = order(conn, memory, {})
    fresh, _ = add_item(conn, memory, "photo", status="ready")
    conn.execute(
        """INSERT INTO media_understandings (media_id, memory_id, description, cast_labels, details)
           VALUES (%s, %s, 'Бабушка у окна', %s, %s)""",
        (fresh, str(memory[0]), ["человек-1"], Jsonb({"place": "Гостиная"})),
    )

    journal.begin(conn, montage_id, 1, 60)
    story.what_was_seen(conn, montage_id, [selection.Piece(media_id=fresh)], set())

    (written,) = entries_of(conn, montage_id, "analysis")
    assert written.details == {"place": "Гостиная", "people": 1}, (
        "нет ключа earlier — описан сейчас (контракт api)"
    )
    assert stages_of(conn, montage_id)["analysis"]["outcome"]["described"] == 1
    assert stages_of(conn, montage_id)["analysis"]["outcome"]["earlier"] == 0


def test_the_description_a_model_gave_is_cut_to_what_the_journal_can_hold(conn, memory):
    """Поля описания едут из модели целиком, а потолки у них свои и мельче, чем у колонок: место,
    свет, настроение, план и камера — по 120 символов, а подробный текст — 1500, и моментов у
    клипа восемь. Без обрезки база отвергла бы реплику целиком (контракт api), и в рассказе о
    ролике осталось бы пустое место.
    """
    journal_ready(conn)
    montage_id = order(conn, memory, {})
    clip, _ = add_item(conn, memory, "video", status="ready")
    moments = [{"seconds": number, "text": "момент " + "я" * 200} for number in range(12)]
    conn.execute(
        """INSERT INTO media_understandings (media_id, memory_id, description, cast_labels, details)
           VALUES (%s, %s, 'Тост', %s, %s)""",
        (
            clip,
            str(memory[0]),
            ["человек-1"],
            Jsonb(
                {
                    "place": "п" * 300,
                    "light": "с" * 300,
                    "mood": "н" * 300,
                    "shot": "к" * 300,
                    "camera": "д" * 300,
                    "text": "т" * 2000,
                    "moments": moments,
                }
            ),
        ),
    )

    journal.begin(conn, montage_id, 1, 60)
    story.what_was_seen(conn, montage_id, [selection.Piece(media_id=clip)], {clip})

    (written,) = entries_of(conn, montage_id, "analysis")
    given = ("place", "light", "mood", "shot", "camera", "text")  # что положили в строку
    assert {name: len(written.details[name]) for name in given} == {
        name: journal.DETAIL_CHARACTERS[name] for name in given
    }
    assert len(written.details["moments"]) == journal.MOMENTS_COUNT
    assert all(len(one["text"]) == journal.MOMENT_CHARACTERS for one in written.details["moments"])
    assert [one["seconds"] for one in written.details["moments"]] == list(
        range(journal.MOMENTS_COUNT)
    )
    assert "speech" not in written.details, "у этого клипа речи не записано, и говорить не о чем"


def test_a_moment_the_model_gave_unusably_is_left_out_rather_than_shown(conn, memory):
    """Момент клипа — это секунда и текст, и оба нужны: карточка рисует его по полосе ролика, и
    без секунды на полосе его не поставить, а без текста — и подать нечего. Момент без того или
    без другого — не момент, и на карточку он не попадает вовсе.
    """
    journal_ready(conn)
    montage_id = order(conn, memory, {})
    clip, _ = add_item(conn, memory, "video", status="ready")
    conn.execute(
        """INSERT INTO media_understandings (media_id, memory_id, description, cast_labels, details)
           VALUES (%s, %s, 'Тост', %s, %s)""",
        (
            clip,
            str(memory[0]),
            ["человек-1"],
            Jsonb(
                {
                    "moments": [
                        {"seconds": 7, "text": "поднимают бокалы"},
                        {"seconds": "не семь", "text": "поднимают бокалы"},
                        {"seconds": 12, "text": "   "},
                        "пару поднимают на руки",
                        {"seconds": -3, "text": "до начала"},
                    ]
                }
            ),
        ),
    )

    journal.begin(conn, montage_id, 1, 60)
    story.what_was_seen(conn, montage_id, [selection.Piece(media_id=clip)], {clip})

    (written,) = entries_of(conn, montage_id, "analysis")
    assert written.details["moments"] == [{"seconds": 7, "text": "поднимают бокалы"}]


def test_a_frames_speech_and_comments_come_from_their_own_tables(conn, memory):
    """Речь клипа — из `media_speech`, а не от модели: модель видит вырезанные кадры и звука не
    слышит, и спросить её, что слышно, значило бы выдумать это людям. Комментарии участников —
    из своего места (`understanding.frame_comments`), а пока хранилища нет, их на карточке нет,
    и пустое место читается как «пока никто ничего не написал», а не как ошибка (решение chief,
    2026-10-03).

    Клип без расшифровки и фотография — оба молчат: речи, которой не записано, на карточке нет.
    """
    journal_ready(conn)
    montage_id = order(conn, memory, {})
    clip, _ = add_item(conn, memory, "video", status="ready")
    photo, _ = add_item(conn, memory, "photo", status="ready")
    for media_id, description in ((clip, "Тост"), (photo, "Стол")):
        conn.execute(
            """INSERT INTO media_understandings (media_id, memory_id, description, cast_labels)
               VALUES (%s, %s, %s, %s)""",
            (media_id, str(memory[0]), description, ["человек-1"]),
        )
    conn.execute(
        "INSERT INTO media_speech (media_id, transcript) VALUES (%s, %s)",
        (clip, "С днём рождения! " * 40),
    )

    journal.begin(conn, montage_id, 2, 60)
    judged = [selection.Piece(media_id=clip), selection.Piece(media_id=photo)]
    story.what_was_seen(conn, montage_id, judged, set())

    written = {one.media_id: one.details for one in entries_of(conn, montage_id, "analysis")}
    assert (
        written[clip]["speech"].startswith("С днём рождения!")
        and len(written[clip]["speech"]) == journal.DETAIL_CHARACTERS["speech"]
    ), "речь обрезана по своему потолку"
    assert "speech" not in written[photo], "у фотографии речи нет и говорить не о чем"
    assert all("comments" not in fields for fields in written.values()), (
        "пока m-32 не заведёт хранилище, комментариев ни у кого нет"
    )


def test_the_one_place_comments_come_from_is_empty_and_breaks_nothing(conn, memory):
    """`frame_comments` — единственное место, откуда берутся комментарии участников к кадру, и
    сегодня оно пусто. Пусто здесь означает «пока никто ничего не написал»: вызывающий код обязан
    быть готов к пустому ответу у любого кадра, а хранилище заведёт m-32 (doc-14).
    """
    media_ids = [add_item(conn, memory, "photo", status="ready")[0] for _ in range(2)]

    assert understanding.frame_comments(conn, media_ids) == {}
    assert understanding.frame_comments(conn, []) == {}, "и на пустом заказе тоже"


def test_every_decision_of_the_selection_is_told_with_its_reason(conn, memory):
    """Режиссёрский этап пересказывает решения отбора: «беру» на взятом с его куском исходника и
    «взял 1 из 4» на том, кто остался за серию, и «мимо» с кодом причины на каждом, кто в ролик не
    попал.

    Чего на этом этапе нет: кадра, названного на подготовке (второе «мимо» с той же причиной — это
    один кадр, пропущенный дважды, и завышенный счётчик «Мимо N»), и кадра, который организатор
    вычеркнул сам — о нём в журнале нет ни слова вовсе (контракт api, «Этапы»). Слова
    организатора тоже больше не видно на «беру»: кадры равны, и «включить» ничего не закрепляет.
    """
    journal_ready(conn)
    montage_id = order(conn, memory, {})
    of_a_burst = selection.Piece(media_id=str(uuid.uuid4()), kind="photo", series_size=4)
    clip = selection.Piece(
        media_id=str(uuid.uuid4()), kind="video", seconds=3.0, start=4.0, duration=10.0
    )
    # Причины «мимо» ставит `selection.choose`, копией куска, ровно как здесь; непрочитанный пришёл
    # с причиной с самого первого этапа.
    struck_out = selection.Piece(media_id=str(uuid.uuid4()))
    unreadable = selection.Piece(media_id=str(uuid.uuid4()), rejected=selection.UNREADABLE)
    did_not_fit = selection.Piece(media_id=str(uuid.uuid4()))
    judged = [of_a_burst, clip, struck_out, unreadable, did_not_fit]
    journal.begin(conn, montage_id, len(judged), 60)

    story.choices(
        conn,
        montage_id,
        [of_a_burst, clip],
        [
            dataclasses.replace(struck_out, rejected=selection.EXCLUDED),
            unreadable,
            dataclasses.replace(did_not_fit, rejected=selection.LEFT_OUT),
        ],
        judged,
    )

    written = {one.media_id: one for one in entries_of(conn, montage_id, "directing")}
    assert (written[did_not_fit.media_id].decision, written[did_not_fit.media_id].reason) == (
        "skip",
        selection.LEFT_OUT,
    )
    assert unreadable.media_id not in written, "о непрочитанном рассказала подготовка"
    assert struck_out.media_id not in written, "исключённый организатором в журнале не упоминается"
    assert (written[of_a_burst.media_id].decision, written[of_a_burst.media_id].series_size) == (
        "take",
        4,
    )
    assert (written[clip.media_id].decision, written[clip.media_id].reason) == ("take", None), (
        "взятый кусок не помечен ничьим словом — кадры равны"
    )
    assert (
        written[clip.media_id].source_from_seconds,
        written[clip.media_id].source_to_seconds,
    ) == (4.0, 7.0)


def test_the_series_the_director_kept_is_told_the_same_way(conn, memory):
    """«Взял 1 из 4» и на пути с нейросетью: размер серии едет до журнала на самом куске, и на
    каждом кадре, который режиссёр в план не взял, — своя запись с кодом причины."""
    journal_ready(conn)
    montage_id = order(conn, memory, {})
    of_a_burst = selection.Piece(media_id=str(uuid.uuid4()), kind="photo", series_size=4)
    alone = selection.Piece(media_id=str(uuid.uuid4()), kind="photo")
    left_out = str(uuid.uuid4())
    journal.begin(conn, montage_id, 3, 60)

    story.plan(
        conn,
        montage_id,
        [of_a_burst, alone],
        [
            {"media_id": of_a_burst.media_id, "thought": "Вносят торт", "decision": "take"},
            {"media_id": alone.media_id, "thought": "Накрывают на стол", "decision": "take"},
            {"media_id": left_out, "reason": selection.LEFT_OUT, "decision": "skip"},
        ],
    )

    written = {one.media_id: one for one in entries_of(conn, montage_id, "directing")}
    assert (written[of_a_burst.media_id].decision, written[of_a_burst.media_id].series_size) == (
        "take",
        4,
    )
    assert written[alone.media_id].series_size is None, "кадр сам по себе — не серия"
    assert (written[left_out].decision, written[left_out].reason) == ("skip", selection.LEFT_OUT)


def test_every_stretch_of_one_clip_is_its_own_entry(conn, memory):
    """Из одного клипа режиссёр взял два отрезка — и в журнале их два: у каждого своя мысль и свой
    кусок исходника, «беру 0:04–0:07» и «беру 0:15–0:19», а не последний из кусков на обеих
    записях. Третий отрезок того же клипа не влез — он «мимо», и куска исходника у него нет."""
    journal_ready(conn)
    montage_id = order(conn, memory, {})
    clip = str(uuid.uuid4())
    toast = selection.Piece(media_id=clip, kind="video", start=4.0, seconds=3.0, duration=30.0)
    laughter = dataclasses.replace(toast, start=15.0, seconds=4.0)
    journal.begin(conn, montage_id, 1, 60)

    story.plan(
        conn,
        montage_id,
        [toast, laughter],
        [
            {"thought": "Один тост, два момента."},
            {"media_id": clip, "thought": "Поднимают бокалы", "decision": "take"},
            {"media_id": clip, "thought": "Смеются после тоста", "decision": "take"},
            {
                "media_id": clip,
                "thought": "Прощаются у двери",
                "decision": "skip",
                "reason": selection.LEFT_OUT,
            },
        ],
    )

    assert [
        (one.thought, one.decision, one.source_from_seconds, one.source_to_seconds)
        for one in entries_of(conn, montage_id, "directing")[1:]
    ] == [
        ("Поднимают бокалы", "take", 4.0, 7.0),
        ("Смеются после тоста", "take", 15.0, 19.0),
        ("Прощаются у двери", "skip", None, None),
    ]


def test_the_director_tells_the_journal_the_plan_it_built(conn, s3, memory, tmp_path, monkeypatch):
    """Ролик, который режиссировала нейросеть, рассказывает о себе её словами: первой записью
    этапа — мысль о плане в целом, за ней порядком плана по реплике на каждый кадр, который модели
    показали. Взятый кадр несёт мысль, часть истории и рамку «куда смотрит»; кадр, который она в
    план не взяла, — «мимо» с её же причиной, если она её сказала, и кодом, если нет.

    Записанный ответ, живьём никто ничего не спрашивает (decision-10): `fixtures/` и тот же
    транспорт, что во всех тестах модели.
    """
    journal_ready(conn)
    ids = ready_material(conn, s3, memory, tmp_path)
    seen = ("накрывают на стол", "вносят торт", "собака под столом")
    for media_id, description in zip(ids, seen):
        conn.execute(
            """INSERT INTO media_understandings (media_id, memory_id, description, cast_labels)
               VALUES (%s, %s, %s, %s)""",
            (media_id, str(memory[0]), description, ["человек-1"]),
        )
    planning_over(monkeypatch, ids[:3], "gemini_plan_thoughts.json")
    montage_id = order(
        conn,
        memory,
        {"target_seconds": 30, "music_tracks": [], "story_text": "день рождения, всё вокруг торта"},
    )
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True
    assert read_montage(conn, montage_id)[0] == "ready"

    directing = entries_of(conn, montage_id, "directing")
    assert (directing[0].media_id, directing[0].decision) == (None, None), (
        "мысль о плане в целом — первой записью и без кадра"
    )
    assert directing[0].thought.startswith("План на две части")
    assert [(one.media_id, one.decision, one.reason) for one in directing[1:]] == [
        (ids[0], "take", None),
        (ids[1], "take", None),
        (ids[2], "skip", selection.LEFT_OUT),
        (ids[3], "skip", selection.LEFT_OUT),
    ], "порядком плана, а причина — кодом, который приложение называет своими словами"
    assert (directing[3].thought, directing[4].thought) == ("собака не про этот вечер", None), (
        "а слова модели о ней — мыслью там, где она их сказала"
    )

    tort = directing[2]
    assert tort.thought.startswith("Вносят торт")
    assert (tort.story_part, tort.story_part_title) == (2, "Торт")
    assert tort.focus == (0.3, 0.05, 0.25, 0.5), "рамка «куда смотрит» — долями кадра"
    assert directing[1].focus is None, "о чём модель не сказала, того в журнале нет"
    assert stages_of(conn, montage_id)["directing"]["outcome"] == {
        "parts": [
            {"title": "Сборы", "from_seconds": 0.0, "to_seconds": 2.5},
            {"title": "Торт", "from_seconds": 2.5, "to_seconds": 6.5},
        ]
    }, "часть стоит там, где стоит её кусок, а последняя идёт до конца ролика"


def test_a_plan_carrying_what_the_database_cannot_hold_is_still_a_video_and_a_journal(
    conn, s3, memory, tmp_path, monkeypatch
):
    """Оплаченный план с `NaN`, `1e999`, целым длиннее double, NUL и половинкой суррогатной пары:
    ни одно из них не роняет ни запись о запросе, ни ролик, ни реплику журнала — ролик собирается
    из того, что в плане годно, а рассказ о нём — теми словами, что остались."""
    journal_ready(conn)
    ids = ready_material(conn, s3, memory, tmp_path)
    for media_id, description in zip(ids, ("накрывают на стол", "вносят торт", "собака")):
        conn.execute(
            """INSERT INTO media_understandings (media_id, memory_id, description, cast_labels)
               VALUES (%s, %s, %s, %s)""",
            (media_id, str(memory[0]), description, ["человек-1"]),
        )
    planning_over(monkeypatch, ids[:3], "gemini_plan_unholdable.json")
    montage_id = order(
        conn, memory, {"target_seconds": 30, "music_tracks": [], "story_text": "день рождения"}
    )
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True
    assert read_montage(conn, montage_id)[0] == "ready"

    assert conn.execute(
        "SELECT status FROM ai_requests WHERE operation = 'plan_montage' AND ref = %s",
        (montage_id,),
    ).fetchone() == ("answered",)
    directing = entries_of(conn, montage_id, "directing")
    assert directing[0].thought == "План на три части"
    assert [(one.media_id, one.decision, one.thought, one.focus) for one in directing[1:4]] == [
        (ids[0], "take", "Стол накрывают", None),
        (ids[1], "take", "Вносят торт", None),
        (ids[2], "take", "Собака под столом", (0.1, 0.2, 0.3, 0.4)),
    ]


def test_the_parts_of_the_story_are_the_parts_the_video_has():
    """Части истории в итоге этапа — по порядку в ролике, с названием и с тем местом, где часть
    стоит: приложение рисует по ним полосу под роликом. Часть, ни один кадр которой не влез в
    заказанную длину, в итоге не стоит вовсе — её кадр ушёл «мимо», и полоса без неё не врёт."""
    pieces = [
        selection.Piece(media_id="a", seconds=3.0),
        selection.Piece(media_id="b", seconds=4.0),
        selection.Piece(media_id="d", seconds=2.0),
    ]
    thoughts = [
        {"thought": "Три части: сборы, торт, прощание."},
        {"media_id": "a", "decision": "take", "story_part": 1, "story_part_title": "Сборы"},
        {"media_id": "b", "decision": "take", "story_part": 2, "story_part_title": "Торт"},
        {
            "media_id": "c",
            "decision": "skip",
            "reason": selection.LEFT_OUT,
            "story_part": 3,
            "story_part_title": "Прощание",
        },
        {"media_id": "d", "decision": "take", "story_part": 3, "story_part_title": "Прощание"},
    ]

    assert story.parts(pieces, thoughts) == {
        "parts": [
            {"title": "Сборы", "from_seconds": 0.0, "to_seconds": 2.5},
            {"title": "Торт", "from_seconds": 2.5, "to_seconds": 6.0},
            {"title": "Прощание", "from_seconds": 6.0, "to_seconds": 8.0},
        ]
    }


def test_a_plan_without_parts_leaves_the_directors_outcome_silent():
    """План без частей вовсе (записанный ответ, снятый до того, как режиссёра спросили говорить):
    итог молчит — отсутствующий ключ экран просто не рисует, а ноль частей был бы враньём."""
    one = selection.Piece(media_id="a")
    assert story.parts([one], [{"media_id": "a", "decision": "take"}]) == {}


def test_the_titles_in_the_directors_outcome_are_cleaned_like_the_entries():
    """Названия частей — слова модели, а итог этапа видят все участники, как и реплики: в итог они
    идут той же чисткой и тем же потолком, иначе служебный символ и простыня текста, отсечённые
    у реплики, въехали бы на экран «Ролик готов» через итог."""
    thoughts = [
        {
            "media_id": "a",
            "decision": "take",
            "story_part": 1,
            "story_part_title": "Сборы\u202e\nи\u0007 " + "я" * 200,
        }
    ]

    (one,) = story.parts([selection.Piece(media_id="a")], thoughts)["parts"]

    assert one["title"].startswith("Сборы и я") and len(one["title"]) == journal.TITLE_CHARACTERS


def test_service_characters_in_the_models_words_do_not_reach_the_journal(conn, memory):
    """Мысль модели увидят все участники воспоминания, а сказана она недоверенным ртом: перевод
    строки, управляющий символ, невидимая метка, разворачивающая текст, — всё это в рассказе о
    ролике не нужно, и до базы оно не доходит."""
    journal_ready(conn)
    montage_id = order(conn, memory, {})
    journal.begin(conn, montage_id, 1, 60)

    journal.entry(
        conn,
        montage_id,
        "directing",
        media_id=str(uuid.uuid4()),
        decision="take",
        thought="Бабушка\u0007 смеётся‮,\n\nсвет тёплый",
        story_part=2,
        story_part_title="Клятвы​ и объятия",
        labels=["смех\u0000", "​"],
    )

    (written,) = entries_of(conn, montage_id, "directing")
    assert written.thought == "Бабушка смеётся, свет тёплый"
    assert written.story_part_title == "Клятвы и объятия"
    assert written.labels == ["смех"], "метка из одних невидимок — не метка"


def test_a_reply_too_long_for_the_journal_is_cut_rather_than_lost(conn, memory):
    """Потолки колонок стоят `CHECK`-ами в базе, а длинное приходит от модели: описание кадра,
    название части, метки. Обрезать их — здесь, иначе база отвергла бы всю реплику, и в рассказе
    осталось бы пустое место вместо хвоста фразы."""
    journal_ready(conn)
    montage_id = order(conn, memory, {})
    journal.begin(conn, montage_id, 1, 60)

    journal.entry(
        conn,
        montage_id,
        "directing",
        media_id=str(uuid.uuid4()),
        decision="take",
        thought="а" * 900,
        story_part=2,
        story_part_title="б" * 200,
        labels=["в" * 100, "г" * 100, "д" * 100, *(f"метка-{index}" for index in range(15))],
    )

    (written,) = entries_of(conn, montage_id, "directing")
    assert len(written.thought) == journal.THOUGHT_CHARACTERS
    assert len(written.labels) <= journal.LABELS_COUNT
    assert len(",".join(written.labels)) <= journal.LABELS_CHARACTERS


def test_the_worker_and_the_api_promise_the_same_minutes():
    """«Соберём за 15 минут» на итоге заказа и «осталось ≈» на экране монтажа считаются по одним и
    тем же числам: 3 секунды на кадр и 1,5 секунды на секунду готового ролика (api,
    `estimateMinutes`, выверено по макету MontageSummary — 214 кадров на 2:30). Разъедутся они
    ровно тогда, когда кто-то подвинет ручку здесь и не подвинет там."""
    assert sum(per_file for per_file, _ in journal.ESTIMATE.values()) == 3.0
    assert sum(per_second for _, per_second in journal.ESTIMATE.values()) == 1.5

    whole = sum(journal.seconds_for(stage, 214, 150) for stage in journal.STAGES)
    assert max(math.ceil(whole / 60), 1) == 15


def test_the_journal_takes_the_series_the_dedup_counted():
    """Размер серии не выдуман оркестратором: его ставит сам дедуп на том кадре, который за серию
    остался, и дальше он едет с куском до журнала."""
    when = datetime.datetime(2026, 9, 21, 16, 0, tzinfo=datetime.timezone.utc)
    burst = [
        selection.Piece(
            media_id="sharp", sharpness=9.0, when=when, captured_at=when, frame=picture(1)
        ),
        selection.Piece(
            media_id="soft",
            sharpness=1.0,
            when=when,
            captured_at=when + datetime.timedelta(seconds=1),
            frame=picture(1),
        ),
    ]

    kept, turned_down = selection._dedup(burst)

    assert [(piece.media_id, piece.series_size) for piece in kept] == [("sharp", 2)]
    assert [piece.rejected for piece in turned_down] == [selection.DUPLICATE]
    assert dataclasses.replace(kept[0], seconds=1.0).series_size == 2, (
        "и переживает любую перекройку куска по дороге"
    )
