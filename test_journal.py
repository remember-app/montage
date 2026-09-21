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
import journal
import media
import montage
import selection
import worker

# Фикстуры и рисованный ffmpeg-ом материал — `test_media.py`'s, как и везде в этом наборе.
from test_media import (  # noqa: F401
    add_item, conn, ffmpeg, fetch, memory, photo, probe, s3, video,
)
from test_montage import crushed, material, order, read_montage

Entry = collections.namedtuple(
    "Entry", "stage media_id thought decision reason series_size labels "
             "source_from_seconds source_to_seconds"
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
    """Реплики журнала в том порядке, в каком они легли, — `id` и есть курсор приложения."""
    return [
        Entry(*row)
        for row in conn.execute(
            """SELECT stage, media_id::text, thought, decision, reason, series_size, labels,
                      source_from_seconds, source_to_seconds
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
    assert stages["understanding"]["outcome"] == {"described": 0, "of": 5, "people": 0,
                                                  "animals": 0}
    assert entries_of(conn, montage_id, "understanding") == []

    # Режиссёрский этап: по файлу решение с причиной, у видео — какой кусок исходника взят.
    directing = entries_of(conn, montage_id, "directing")
    assert {one.media_id: (one.decision, one.reason) for one in directing} == {
        ids[1]: ("take", None), ids[2]: ("take", None), ids[3]: ("take", None),
        ids[0]: ("skip", "excluded"), dark: ("skip", "exposure"),
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
    модели и метки «что увидел». Итог — сколько кадров описано и скольких воспоминание на них
    узнало."""
    journal_ready(conn)
    memory_id, _ = memory
    montage_id = order(conn, memory, {})
    described, _ = add_item(conn, memory, "photo", status="ready")
    silent, _ = add_item(conn, memory, "photo", status="ready")
    conn.execute(
        """INSERT INTO media_understandings (media_id, memory_id, description, cast_labels)
           VALUES (%s, %s, %s, %s)""",
        (described, str(memory_id), "Бабушка смеётся, Лиза обнимает её",
         ["человек-1", "человек-2", "животное-1"]),
    )
    judged = [selection.Piece(media_id=described), selection.Piece(media_id=silent)]

    journal.begin(conn, montage_id, len(judged), 60)
    montage._what_was_seen(conn, montage_id, judged)

    assert [(one.media_id, one.thought, one.labels)
            for one in entries_of(conn, montage_id, "understanding")] == [
        (described, "Бабушка смеётся, Лиза обнимает её", ["человек-1", "человек-2", "животное-1"])
    ]
    assert stages_of(conn, montage_id)["understanding"]["outcome"] == {
        "described": 1, "of": 2, "people": 2, "animals": 1
    }


def test_a_series_is_one_entry_and_the_organizer_is_visible_in_the_decisions(conn, memory):
    """Серия одинаковых кадров — одна запись «взял 1 из 4» на том, кто из неё остался, а не три
    «мимо, дубль» рядом. Слово организатора видно с обеих сторон: `included` на том, что он велел
    взять, `excluded` на вычеркнутом."""
    journal_ready(conn)
    montage_id = order(conn, memory, {})
    of_a_burst = selection.Piece(media_id=str(uuid.uuid4()), kind="photo", series_size=4)
    asked_for = selection.Piece(media_id=str(uuid.uuid4()), kind="video", seconds=3.0, start=4.0,
                                duration=10.0)
    struck_out = selection.Piece(media_id=str(uuid.uuid4()), rejected=selection.EXCLUDED)
    swallowed = selection.Piece(media_id=str(uuid.uuid4()), rejected=selection.DUPLICATE)
    journal.begin(conn, montage_id, 4, 60)

    montage._write_choices(conn, montage_id, [of_a_burst, asked_for],
                           [struck_out, swallowed], {asked_for.media_id})

    written = {one.media_id: one for one in entries_of(conn, montage_id, "directing")}
    assert swallowed.media_id not in written, "серию рассказала одна запись выше"
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
