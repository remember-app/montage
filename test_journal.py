"""Журнал монтажа: что воркер рассказывает о сборке, пока она идёт — см. README.md.

Real everything, как и в `test_montage.py`: настоящий Postgres, настоящий бакет, настоящий ffmpeg.
Журнал для того и пишется, чтобы его читали вживую, поэтому здесь есть и **второе соединение** —
то самое, которым его читает api, пока ролик ещё собирается.

Warning: эти тесты чистят таблицу `jobs` той базы, на которую смотрит DATABASE_URL, и пишут в
бакет, на который смотрит R2_ENDPOINT.
"""

import collections
import dataclasses
import datetime
import math
import uuid

import pytest

import assembly
import director
import journal
import media
import montage
import selection
import worker

# Фикстуры и рисованный ffmpeg-ом материал — `test_media.py`'s, как и везде в этом наборе.
from test_media import (  # noqa: F401
    add_item, conn, ffmpeg, fetch, memory, photo, probe, s3, video,
)
from test_montage import crushed, material, order, planning_over, read_montage

Entry = collections.namedtuple(
    "Entry", "stage media_id thought decision reason series_size labels "
             "source_from_seconds source_to_seconds story_part story_part_title focus"
)


def journal_ready(conn):
    if conn.execute("SELECT to_regclass('montage_stages')").fetchone()[0] is None:
        pytest.skip("montage_stages is missing — run the api's migrations first (see README.md)")


def stages_of(conn, montage_id):
    """Этапы ролика — в том порядке, в котором их прошли: у этапа своего номера нет, порядок в
    базе держится только временем начала (а в ответе api — его собственным списком)."""
    return {
        stage: {"state": state, "started_at": started_at, "finished_at": finished_at, "eta": eta,
                "done": done, "total": total, "outcome": outcome}
        for stage, state, started_at, finished_at, eta, done, total, outcome in conn.execute(
            """SELECT stage, state, started_at, finished_at, eta_seconds, done_count, total_count,
                      outcome
                 FROM montage_stages WHERE montage_id = %s ORDER BY started_at NULLS LAST, stage""",
            (montage_id,),
        ).fetchall()
    }


def entries_of(conn, montage_id, stage=None):
    """Реплики журнала в том порядке, в каком они легли, — `id` и есть курсор приложения.

    Рамка «куда смотрит» читается так же, как её отдаёт api, — одним значением из четырёх долей
    или пустотой: три доли из четырёх база не приняла бы вовсе.
    """
    return [
        Entry(*row[:-4], row[-4:] if row[-4] is not None else None)
        for row in conn.execute(
            """SELECT stage, media_id::text, thought, decision, reason, series_size, labels,
                      source_from_seconds, source_to_seconds, story_part, story_part_title,
                      focus_x, focus_y, focus_w, focus_h
                 FROM montage_journal_entries
                WHERE montage_id = %s AND (%s::text IS NULL OR stage = %s) ORDER BY id""",
            (montage_id, stage, stage),
        ).fetchall()
    ]


# --- весь рассказ целиком ----------------------------------------------------


def test_the_journal_tells_the_whole_assembly_stage_by_stage(conn, s3, memory, tmp_path):
    """Пять этапов с первой секунды и до готового ролика: каждый прошёл ожидание, работу и
    готово, у каждого записано, сколько он занял и чем кончился. Материал подобран так, чтобы
    рассказывать было о чём: три фотографии и клип, один снимок пересвечен в черноту (его отсев
    не берёт), один вычеркнут организатором.
    """
    journal_ready(conn)
    ids = material(conn, s3, memory, tmp_path)
    dark, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(crushed(tmp_path / "crushed.jpg")), media.bucket(), key)
    montage_id = order(conn, memory, {"target_seconds": 30, "music_track_id": "warm",
                                      "exclude_media_ids": [ids[0]]})
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True
    assert read_montage(conn, montage_id)[0] == "ready"

    stages = stages_of(conn, montage_id)
    assert list(stages) == list(journal.STAGES), "этапы идут в порядке пайплайна"
    for name, stage in stages.items():
        assert stage["state"] == "done", name
        assert stage["started_at"] <= stage["finished_at"], f"{name}: сколько занял"
        assert stage["eta"] == 0, f"{name}: готовому этапу нечего ждать"

    # Отсев: запись на каждый отсеянный файл и «годятся N из M» итогом.
    assert (stages["screening"]["done"], stages["screening"]["total"]) == (5, 5)
    assert stages["screening"]["outcome"] == {"kept": 4, "of": 5, "reasons": {"exposure": 1}}
    assert [(one.media_id, one.decision, one.reason)
            for one in entries_of(conn, montage_id, "screening")] == [(dark, "skip", "exposure")]

    # Разбор кадров: описаний нет ни у одного файла — этап проходит и честно говорит «0 из 5».
    assert stages["understanding"]["outcome"] == {"described": 0, "of": 5, "faces": 0,
                                                  "people": 0, "animals": 0}
    assert entries_of(conn, montage_id, "understanding") == []

    # Режиссёрский этап: по файлу решение с причиной, у видео — какой кусок исходника взят.
    # Пересвеченного кадра здесь нет вовсе: о нём рассказал отсев, и второе «мимо» с той же
    # причиной — тот же кадр, пропущенный дважды.
    directing = entries_of(conn, montage_id, "directing")
    assert {one.media_id: (one.decision, one.reason) for one in directing} == {
        ids[1]: ("take", None), ids[2]: ("take", None), ids[3]: ("take", None),
        ids[0]: ("skip", "excluded"),
    }
    assert stages["directing"]["outcome"] == {"frames": 3, "of": 5}
    clip = next(one for one in directing if one.media_id == ids[3])
    assert clip.source_to_seconds > clip.source_from_seconds >= 0, "кусок клипа — секундами"
    still = next(one for one in directing if one.media_id == ids[1])
    assert (still.source_from_seconds, still.source_to_seconds) == (None, None), (
        "у фотографии куска исходника нет — она идёт в ролик целиком"
    )

    # Ритм: подо что подогнаны куски.
    assert stages["rhythm"]["outcome"]["track"] == ["warm"]
    assert stages["rhythm"]["outcome"]["cuts"] >= 0

    # Склейка: сколько склеек и какая длина.
    assert (stages["assembly"]["done"], stages["assembly"]["total"]) == (3, 3)
    assert stages["assembly"]["outcome"]["cuts"] == 2, "три куска — две склейки"
    assert stages["assembly"]["outcome"]["seconds"] == pytest.approx(
        read_montage(conn, montage_id)[2], abs=0.2
    )


def test_a_montage_without_music_says_so_where_the_rhythm_would_be(conn, s3, memory, tmp_path):
    """Ролик с живым звуком: этап ритма всё равно проходит, и итог у него честный — дорожки нет,
    подгонять было не подо что, ни один кусок не сдвинут."""
    journal_ready(conn)
    montage_id = order(conn, memory, {"target_seconds": 10, "music_tracks": []})
    for index in range(3):
        _, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(photo(tmp_path / f"photo-{index}.jpg")), media.bucket(), key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    stages = stages_of(conn, montage_id)
    assert stages["rhythm"]["state"] == "done"
    assert stages["rhythm"]["outcome"] == {"track": [], "cuts": 0}


# --- вживую, а не пачкой в конце ---------------------------------------------


def from_another_connection(montage_id):
    """Журнал глазами api: отдельное соединение, открытое посреди сборки, — так его и читает
    приложение, опрашивая маршрут раз в секунду-две."""
    with worker.connect() as watcher:
        status = watcher.execute(
            "SELECT status FROM montages WHERE id = %s", (montage_id,)
        ).fetchone()[0]
        return status, entries_of(watcher, montage_id), stages_of(watcher, montage_id)


def test_the_journal_is_read_from_another_connection_while_the_montage_is_still_being_assembled(
    conn, s3, memory, tmp_path, monkeypatch
):
    """Живость: то, что записано, видно чужому соединению *до* конца этапа и до конца сборки.
    Запись пачкой в конце — ровно то, чего на экране «монтаж идёт» быть не должно.

    Смотрим дважды: перед самой склейкой (к этому времени отсев и режиссёр уже всё рассказали) и
    после каждого готового куска — прогресс склейки должен двигаться, а не появиться разом.
    """
    journal_ready(conn)
    material(conn, s3, memory, tmp_path)
    _, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(crushed(tmp_path / "crushed.jpg")), media.bucket(), key)
    montage_id = order(conn, memory, {"target_seconds": 30, "music_tracks": []})

    before, after = [], []
    assembling = assembly.assemble

    def watched(pieces, orientation, track, workdir, out, on_clip=None):
        def watching(done):
            on_clip(done)
            after.append(from_another_connection(montage_id))
        before.append(from_another_connection(montage_id))
        return assembling(pieces, orientation, track, workdir, out, on_clip=watching)

    monkeypatch.setattr(assembly, "assemble", watched)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, entries, stages = before[0]
    assert status == "processing", "смотрели, пока ролик ещё собирался"
    assert [one for one in entries if one.stage == "screening"], "отсев уже рассказал о себе"
    assert [one for one in entries if one.decision == "take"], "и режиссёр тоже"
    assert stages["screening"]["state"] == "done" and stages["assembly"]["state"] == "running"

    assert [stages["assembly"]["done"] for _, _, stages in after] == [1, 2, 3, 4], (
        "прогресс склейки идёт по готовому куску, а не появляется разом в конце"
    )
    assert all(stages["assembly"]["eta"] is not None for _, _, stages in after), (
        "у идущего этапа есть «осталось ≈», посчитанное по тому, как он идёт"
    )


# --- повтор, падение и цена ошибки -------------------------------------------


def test_a_montage_assembled_again_starts_its_journal_from_a_clean_sheet(conn, memory):
    """«Повторить» упавшего ролика и ретрай задачи — это вторая попытка, а не продолжение первой:
    иначе на экране шли бы два отсева подряд и решения по одному кадру дважды."""
    journal_ready(conn)
    montage_id = order(conn, memory, {})
    journal.begin(conn, montage_id, 5, 60)
    journal.start(conn, montage_id, "screening", total=5)
    journal.entry(conn, montage_id, "screening", media_id=str(uuid.uuid4()), decision="skip",
                  reason=selection.BLUR)
    journal.finish(conn, montage_id, "screening", kept=4, of=5)

    journal.begin(conn, montage_id, 2, 30)

    assert entries_of(conn, montage_id) == []
    stages = stages_of(conn, montage_id)
    assert len(stages) == len(journal.STAGES)
    assert all(
        (one["state"], one["done"], one["total"], one["outcome"], one["started_at"])
        == ("pending", 0, None, {}, None)
        for one in stages.values()
    )
    assert stages["screening"]["eta"] == journal.seconds_for("screening", 2, 30), (
        "и оценка времени пересчитана по новому материалу"
    )


def layout_of(conn, montage_id):
    return conn.execute(
        """SELECT media_id::text, start_seconds FROM montage_frames
            WHERE montage_id = %s ORDER BY start_seconds""",
        (montage_id,),
    ).fetchall()


def test_a_montage_assembled_again_does_not_show_the_last_attempts_layout(
    conn, s3, memory, tmp_path, monkeypatch
):
    """Раскладка живёт и умирает вместе с журналом попытки. API присоединяет места кадров к записям
    журнала по одному `montage_id`, без статуса ролика, — и повтор, оставивший прошлую раскладку до
    своей, показывал бы на отсеве кадр там, где его поставил ролик, которого уже нет."""
    journal_ready(conn)
    ids = material(conn, s3, memory, tmp_path)
    montage_id = order(conn, memory, {"target_seconds": 30, "music_tracks": []})
    # Прошлая попытка дошла до склейки и упала: её журнал и её раскладка лежат в базе.
    journal.begin(conn, montage_id, len(ids), 30)
    journal.entry(conn, montage_id, "screening", media_id=ids[0], decision="skip",
                  reason=selection.BLUR)
    conn.execute(
        """INSERT INTO montage_frames (montage_id, media_id, start_seconds, duration_seconds)
           VALUES (%s, %s, 42, 3)""",
        (montage_id, ids[0]),
    )

    before_the_new_one = []
    writing = montage._write_layout

    def watched(connection, ref, pieces):
        if pieces:
            with worker.connect() as watcher:
                before_the_new_one.append(layout_of(watcher, montage_id))
        return writing(connection, ref, pieces)

    monkeypatch.setattr(montage, "_write_layout", watched)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True
    assert read_montage(conn, montage_id)[0] == "ready"

    assert before_the_new_one[0] == [], "до новой раскладки прошлой нет"
    layout = layout_of(conn, montage_id)
    assert {media_id for media_id, _ in layout} == set(ids), "а после сборки — новая"
    assert layout[0][1] == 0 and all(start != 42 for _, start in layout)


def test_a_montage_that_fell_leaves_the_stage_it_fell_on_marked_failed(conn, memory):
    """Упавший ролик оставляет журнал как есть — до какого места дошёл, видно, — а этап, который
    шёл в этот момент, помечен упавшим. Сделанное до него остаётся сделанным."""
    journal_ready(conn)
    montage_id = order(conn, memory, {})
    journal.begin(conn, montage_id, 5, 60)
    journal.start(conn, montage_id, "screening", total=5)
    journal.finish(conn, montage_id, "screening", kept=5, of=5)
    journal.start(conn, montage_id, "understanding", total=5)

    montage.give_up(conn, montage_id, "не удалось собрать ролик")

    stages = stages_of(conn, montage_id)
    assert read_montage(conn, montage_id)[0] == "failed"
    assert [stages[name]["state"] for name in journal.STAGES] == [
        "done", "failed", "pending", "pending", "pending"
    ]
    assert stages["understanding"]["eta"] is None, "упавшему этапу ждать нечего"


def test_a_journal_that_cannot_be_written_does_not_cost_the_montage(conn):
    """Журнал — зрелище, а не суть: ролик, который собирается минутами, не выбрасывают из-за
    строки о нём. Здесь ролика нет вовсе, так что не пройдёт ни одна запись."""
    journal_ready(conn)
    gone = str(uuid.uuid4())

    journal.begin(conn, gone, 5, 60)
    journal.start(conn, gone, "screening", total=5)
    journal.progress(conn, gone, "screening", 1)
    journal.entry(conn, gone, "screening", media_id=str(uuid.uuid4()), decision="skip",
                  reason=selection.BLUR)
    journal.finish(conn, gone, "screening", kept=4, of=5)
    journal.failed(conn, gone)
    journal.entry(conn, gone, "screening", nonsense="колонки с таким именем в журнале нет")

    assert stages_of(conn, gone) == {} and entries_of(conn, gone) == []


# --- что и чем пишется -------------------------------------------------------


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
    montage._what_was_seen(conn, montage_id, judged)

    # Порядком записей этап ничего не рассказывает — они про разные кадры и ложатся одной пачкой.
    assert {one.media_id: (one.thought, one.labels)
            for one in entries_of(conn, montage_id, "understanding")} == {
        described: ("Бабушка смеётся, Лиза обнимает её",
                    ["человек-1", "человек-2", "животное-1"]),
        again: ("Бабушка у окна", ["человек-1"]),
    }
    assert stages_of(conn, montage_id)["understanding"]["outcome"] == {
        "described": 2, "of": 3, "faces": 3, "people": 2, "animals": 1
    }


def test_a_series_is_one_entry_and_the_organizer_is_visible_in_the_decisions(conn, memory):
    """Серия одинаковых кадров — одна запись «взял 1 из 4» на том, кто из неё остался, а не три
    «мимо, дубль» рядом. Слово организатора видно с обеих сторон: `included` на том, что он велел
    взять, `excluded` на вычеркнутом.

    А смазанного кадра на этом этапе нет вовсе: о нём рассказал отсев своей записью, и второе
    «мимо» с той же причиной — тот же кадр, пропущенный дважды, и завышенный счётчик «Мимо N».
    """
    journal_ready(conn)
    montage_id = order(conn, memory, {})
    of_a_burst = selection.Piece(media_id=str(uuid.uuid4()), kind="photo", series_size=4)
    asked_for = selection.Piece(media_id=str(uuid.uuid4()), kind="video", seconds=3.0, start=4.0,
                                duration=10.0)
    # Вычеркнутый и съеденный дедупом отсев прошли — их причины ставит уже `selection.choose`,
    # копией куска, ровно как здесь; смазанный пришёл с причиной с самого отсева.
    struck_out = selection.Piece(media_id=str(uuid.uuid4()))
    swallowed = selection.Piece(media_id=str(uuid.uuid4()))
    blurred = selection.Piece(media_id=str(uuid.uuid4()), rejected=selection.BLUR)
    judged = [of_a_burst, asked_for, struck_out, swallowed, blurred]
    journal.begin(conn, montage_id, len(judged), 60)

    montage._write_choices(
        conn, montage_id, [of_a_burst, asked_for],
        [dataclasses.replace(struck_out, rejected=selection.EXCLUDED),
         dataclasses.replace(swallowed, rejected=selection.DUPLICATE), blurred],
        {asked_for.media_id}, judged,
    )

    written = {one.media_id: one for one in entries_of(conn, montage_id, "directing")}
    assert swallowed.media_id not in written, "серию рассказала одна запись выше"
    assert blurred.media_id not in written, "о смазанном рассказал отсев, второго «мимо» нет"
    assert (written[of_a_burst.media_id].decision, written[of_a_burst.media_id].series_size) == (
        "take", 4
    )
    assert (written[asked_for.media_id].decision, written[asked_for.media_id].reason) == (
        "take", montage.ORGANIZER_ASKED
    )
    assert (written[asked_for.media_id].source_from_seconds,
            written[asked_for.media_id].source_to_seconds) == (4.0, 7.0)
    assert (written[struck_out.media_id].decision, written[struck_out.media_id].reason) == (
        "skip", selection.EXCLUDED
    )


def test_the_series_the_director_kept_is_one_entry_too(conn, memory):
    """«Взял 1 из 4» — и на пути с нейросетью: серию прорежает тот же дедуп ещё до вопроса модели
    (`director.direct`), размер серии едет до журнала на самом куске, и рассказана она одной
    записью — мыслью режиссёра о том кадре, который из неё остался."""
    journal_ready(conn)
    montage_id = order(conn, memory, {})
    of_a_burst = selection.Piece(media_id=str(uuid.uuid4()), kind="photo", series_size=4)
    alone = selection.Piece(media_id=str(uuid.uuid4()), kind="photo")
    journal.begin(conn, montage_id, 2, 60)

    montage._write_thoughts(
        conn, montage_id, [of_a_burst, alone],
        [{"media_id": of_a_burst.media_id, "thought": "Вносят торт", "decision": "take"},
         {"media_id": alone.media_id, "thought": "Накрывают на стол", "decision": "take"}],
        set(),
    )

    written = {one.media_id: one for one in entries_of(conn, montage_id, "directing")}
    assert (written[of_a_burst.media_id].decision,
            written[of_a_burst.media_id].series_size) == ("take", 4)
    assert written[alone.media_id].series_size is None, "кадр сам по себе — не серия"


def test_the_director_tells_the_journal_the_plan_it_built(
    conn, s3, memory, tmp_path, monkeypatch
):
    """Ролик, который режиссировала нейросеть, рассказывает о себе её словами: первой записью
    этапа — мысль о плане в целом, за ней порядком плана по реплике на каждый кадр, который модели
    показали. Взятый кадр несёт мысль, часть истории и рамку «куда смотрит»; кадр, который она в
    план не взяла, — «мимо» с её же причиной, если она её сказала, и кодом, если нет.

    Записанный ответ, живьём никто ничего не спрашивает (decision-10): `fixtures/` и тот же
    транспорт, что во всех тестах модели.
    """
    journal_ready(conn)
    ids = material(conn, s3, memory, tmp_path)
    seen = ("накрывают на стол", "вносят торт", "собака под столом")
    for media_id, description in zip(ids, seen):
        conn.execute(
            """INSERT INTO media_understandings (media_id, memory_id, description, cast_labels)
               VALUES (%s, %s, %s, %s)""",
            (media_id, str(memory[0]), description, ["человек-1"]),
        )
    planning_over(monkeypatch, ids[:3], "gemini_plan_thoughts.json")
    montage_id = order(conn, memory, {"target_seconds": 30, "music_tracks": [],
                                      "story_text": "день рождения, всё вокруг торта"})
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True
    assert read_montage(conn, montage_id)[0] == "ready"

    directing = entries_of(conn, montage_id, "directing")
    assert (directing[0].media_id, directing[0].decision) == (None, None), (
        "мысль о плане в целом — первой записью и без кадра"
    )
    assert directing[0].thought.startswith("План на две части")
    assert [(one.media_id, one.decision, one.reason) for one in directing[1:]] == [
        (ids[0], "take", None), (ids[1], "take", None),
        (ids[2], "skip", director.LEFT_OUT), (ids[3], "skip", director.LEFT_OUT),
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
        "frames": 2, "of": 4, "parts": 2, "part_titles": ["Сборы", "Торт"]
    }


def test_the_parts_of_the_story_are_the_parts_the_video_has():
    """«3 части, 61 кадр» — про ролик: часть, ни один кадр которой не влез в заказанную длину, в
    итоге не считается, как не считаются в `frames` её кадры."""
    thoughts = [
        {"thought": "Три части: сборы, торт, прощание."},
        {"media_id": "a", "decision": "take", "story_part": 1, "story_part_title": "Сборы"},
        {"media_id": "b", "decision": "take", "story_part": 2, "story_part_title": "Торт"},
        {"media_id": "c", "decision": "skip", "reason": selection.NO_ROOM, "story_part": 3,
         "story_part_title": "Прощание"},
    ]

    assert montage._story_parts(thoughts) == {"parts": 2, "part_titles": ["Сборы", "Торт"]}


def test_service_characters_in_the_models_words_do_not_reach_the_journal(conn, memory):
    """Мысль модели увидят все участники воспоминания, а сказана она недоверенным ртом: перевод
    строки, управляющий символ, невидимая метка, разворачивающая текст, — всё это в рассказе о
    ролике не нужно, и до базы оно не доходит."""
    journal_ready(conn)
    montage_id = order(conn, memory, {})
    journal.begin(conn, montage_id, 1, 60)

    journal.entry(
        conn, montage_id, "directing", media_id=str(uuid.uuid4()), decision="take",
        thought="Бабушка\u0007 смеётся‮,\n\nсвет тёплый", story_part=2,
        story_part_title="Клятвы​ и объятия", labels=["смех\u0000", "​"],
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
        conn, montage_id, "directing", media_id=str(uuid.uuid4()), decision="take",
        thought="а" * 900, story_part=2, story_part_title="б" * 200,
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
        selection.Piece(media_id="sharp", sharpness=9.0, when=when, captured_at=when),
        selection.Piece(media_id="soft", sharpness=1.0, when=when,
                        captured_at=when + datetime.timedelta(seconds=1)),
    ]

    chosen, turned_down = selection.choose(burst, 30)

    assert [(piece.media_id, piece.series_size) for piece in chosen] == [("sharp", 2)]
    assert [piece.rejected for piece in turned_down] == [selection.DUPLICATE]
    assert dataclasses.replace(chosen[0], seconds=1.0).series_size == 2, (
        "и переживает любую перекройку куска по дороге"
    )
