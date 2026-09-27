"""Что воркер кладёт в журнал: реплика на кадр и итог этапа — `docs/jobs/montage-journal.md`.

Здесь не про порядок этапов, а про их содержание: что рассказал отсев, что рассказал
режиссёр-нейросеть и что из его слов вообще доезжает до базы.

Warning: эти тесты чистят таблицу `jobs` той базы, на которую смотрит DATABASE_URL, и пишут в
бакет, на который смотрит R2_ENDPOINT.
"""

import dataclasses
import datetime
import math
import uuid

from montage import journal, worker
from montage.video import director, selection, story

from pieces import picture
from recorded import planning_over
from rows import add_item, entries_of, journal_ready, order, read_montage, ready_material, stages_of


def test_the_understanding_stage_tells_what_was_seen_and_who_was_in_frame(conn, memory):
    """Разбор кадров пересказывает то, что уже записано (`media_understandings`): описание словами
    модели и метки «что увидел». Итог — сколько кадров описано, сколько лиц на них и скольких
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
    for media_id, description, cast in (
        (described, "Бабушка смеётся, Лиза обнимает её", ["человек-1", "человек-2", "животное-1"]),
        (again, "Бабушка у окна", ["человек-1"]),
    ):
        conn.execute(
            """INSERT INTO media_understandings (media_id, memory_id, description, cast_labels)
               VALUES (%s, %s, %s, %s)""",
            (media_id, str(memory_id), description, cast),
        )
    judged = [selection.Piece(media_id=one) for one in (described, again, silent)]

    journal.begin(conn, montage_id, len(judged), 60)
    story.what_was_seen(conn, montage_id, judged)

    # Порядком записей этап ничего не рассказывает — они про разные кадры и ложатся одной пачкой.
    assert {
        one.media_id: (one.thought, one.labels)
        for one in entries_of(conn, montage_id, "understanding")
    } == {
        described: ("Бабушка смеётся, Лиза обнимает её", ["человек-1", "человек-2", "животное-1"]),
        again: ("Бабушка у окна", ["человек-1"]),
    }
    assert stages_of(conn, montage_id)["understanding"]["outcome"] == {
        "described": 2,
        "of": 3,
        "faces": 3,
        "people": 2,
        "animals": 1,
    }


def test_every_shot_of_a_series_is_told_and_the_organizer_is_visible_in_the_decisions(conn, memory):
    """Серия одинаковых кадров — «взял 1 из 4» на том, кто из неё остался, и своё «мимо, дубль» на
    каждом, кого он собой заменил: описание каждого человек уже видел на этапе разбора, и без
    записи кадр пропал бы после него молча. Слово организатора видно с обеих сторон: `included` на
    том, что он велел взять, `excluded` на вычеркнутом.

    А смазанного кадра на этом этапе нет вовсе: о нём рассказал отсев своей записью, и второе
    «мимо» с той же причиной — тот же кадр, пропущенный дважды, и завышенный счётчик «Мимо N».
    """
    journal_ready(conn)
    montage_id = order(conn, memory, {})
    of_a_burst = selection.Piece(media_id=str(uuid.uuid4()), kind="photo", series_size=4)
    asked_for = selection.Piece(
        media_id=str(uuid.uuid4()), kind="video", seconds=3.0, start=4.0, duration=10.0
    )
    # Вычеркнутый и съеденный дедупом отсев прошли — их причины ставит уже `selection.choose`,
    # копией куска, ровно как здесь; смазанный пришёл с причиной с самого отсева.
    struck_out = selection.Piece(media_id=str(uuid.uuid4()))
    swallowed = selection.Piece(media_id=str(uuid.uuid4()))
    blurred = selection.Piece(media_id=str(uuid.uuid4()), rejected=selection.BLUR)
    judged = [of_a_burst, asked_for, struck_out, swallowed, blurred]
    journal.begin(conn, montage_id, len(judged), 60)

    story.choices(
        conn,
        montage_id,
        [of_a_burst, asked_for],
        [
            dataclasses.replace(struck_out, rejected=selection.EXCLUDED),
            dataclasses.replace(swallowed, rejected=selection.DUPLICATE),
            blurred,
        ],
        {asked_for.media_id},
        judged,
    )

    written = {one.media_id: one for one in entries_of(conn, montage_id, "directing")}
    assert (written[swallowed.media_id].decision, written[swallowed.media_id].reason) == (
        "skip",
        selection.DUPLICATE,
    )
    assert blurred.media_id not in written, "о смазанном рассказал отсев, второго «мимо» нет"
    assert (written[of_a_burst.media_id].decision, written[of_a_burst.media_id].series_size) == (
        "take",
        4,
    )
    assert (written[asked_for.media_id].decision, written[asked_for.media_id].reason) == (
        "take",
        story.ORGANIZER_ASKED,
    )
    assert (
        written[asked_for.media_id].source_from_seconds,
        written[asked_for.media_id].source_to_seconds,
    ) == (4.0, 7.0)
    assert (written[struck_out.media_id].decision, written[struck_out.media_id].reason) == (
        "skip",
        selection.EXCLUDED,
    )


def test_the_series_the_director_kept_is_told_the_same_way(conn, memory):
    """«Взял 1 из 4» и «мимо, дубль» — и на пути с нейросетью: серию прорежает тот же дедуп ещё до
    вопроса модели (`director.direct`), размер серии едет до журнала на самом куске, а на каждом
    съеденном им кадре — своя запись, хоть модель его и не видела."""
    journal_ready(conn)
    montage_id = order(conn, memory, {})
    of_a_burst = selection.Piece(media_id=str(uuid.uuid4()), kind="photo", series_size=4)
    alone = selection.Piece(media_id=str(uuid.uuid4()), kind="photo")
    swallowed = str(uuid.uuid4())
    journal.begin(conn, montage_id, 3, 60)

    story.plan(
        conn,
        montage_id,
        [of_a_burst, alone],
        [
            {"media_id": of_a_burst.media_id, "thought": "Вносят торт", "decision": "take"},
            {"media_id": alone.media_id, "thought": "Накрывают на стол", "decision": "take"},
            {"media_id": swallowed, "reason": selection.DUPLICATE, "decision": "skip"},
        ],
        set(),
    )

    written = {one.media_id: one for one in entries_of(conn, montage_id, "directing")}
    assert (written[of_a_burst.media_id].decision, written[of_a_burst.media_id].series_size) == (
        "take",
        4,
    )
    assert written[alone.media_id].series_size is None, "кадр сам по себе — не серия"
    assert (written[swallowed].decision, written[swallowed].reason) == ("skip", selection.DUPLICATE)


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
                "reason": selection.NO_ROOM,
            },
        ],
        set(),
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
        (ids[2], "skip", director.LEFT_OUT),
        (ids[3], "skip", director.LEFT_OUT),
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
        "frames": 2,
        "of": 4,
        "parts": 2,
        "part_titles": ["Сборы", "Торт"],
    }


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
    """«3 части, 61 кадр» — про ролик: часть, ни один кадр которой не влез в заказанную длину, в
    итоге не считается, как не считаются в `frames` её кадры."""
    thoughts = [
        {"thought": "Три части: сборы, торт, прощание."},
        {"media_id": "a", "decision": "take", "story_part": 1, "story_part_title": "Сборы"},
        {"media_id": "b", "decision": "take", "story_part": 2, "story_part_title": "Торт"},
        {
            "media_id": "c",
            "decision": "skip",
            "reason": selection.NO_ROOM,
            "story_part": 3,
            "story_part_title": "Прощание",
        },
    ]

    assert story.parts(thoughts) == {"parts": 2, "part_titles": ["Сборы", "Торт"]}


def test_the_titles_in_the_directors_outcome_are_cleaned_like_the_entries():
    """Названия частей — слова модели, а итог этапа видят все участники, как и реплики: в итог они
    идут той же чисткой и тем же потолком, иначе служебный символ и простыня текста, отсечённые
    у реплики, въехали бы на экран «Ролик готов» через итог."""
    thoughts = [
        {
            "media_id": "a",
            "decision": "take",
            "story_part": 1,
            "story_part_title": "Сборы‮\nи\u0007 " + "я" * 200,
        }
    ]

    (title,) = story.parts(thoughts)["part_titles"]

    assert title.startswith("Сборы и я") and len(title) == journal.TITLE_CHARACTERS


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

    chosen, turned_down = selection.choose(burst, 30)

    assert [(piece.media_id, piece.series_size) for piece in chosen] == [("sharp", 2)]
    assert [piece.rejected for piece in turned_down] == [selection.DUPLICATE]
    assert dataclasses.replace(chosen[0], seconds=1.0).series_size == 2, (
        "и переживает любую перекройку куска по дороге"
    )
