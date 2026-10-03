"""Журнал монтажа целиком: этапы по порядку, чтение вживую и что остаётся после падения.

Что именно пишется в каждую реплику — `test_journal_entries.py`; контракт —
`docs/jobs/montage-journal.md`.

Real everything, как и в `test_montage.py`: настоящий Postgres, настоящий бакет, настоящий ffmpeg.
Журнал для того и пишется, чтобы его читали вживую, поэтому здесь есть и **второе соединение** —
то самое, которым его читает api, пока ролик ещё собирается.

Warning: эти тесты чистят таблицу `jobs` той базы, на которую смотрит DATABASE_URL, и пишут в
бакет, на который смотрит R2_ENDPOINT.
"""

import uuid

import pytest

from montage import journal, storage, worker
from montage.video import assembly, layout, montage, selection

from material import crushed, jpeg, unopenable
from rows import (
    add_item,
    entries_of,
    journal_ready,
    layout_of,
    order,
    read_montage,
    ready_material,
    stages_of,
)

# --- весь рассказ целиком ----------------------------------------------------


def test_the_journal_tells_the_whole_assembly_stage_by_stage(conn, s3, memory, tmp_path):
    """Пять этапов с первой секунды и до готового ролика: каждый прошёл ожидание, работу и
    готово, у каждого записано, сколько он занял и чем кончился. Материал подобран так, чтобы
    рассказывать было о чём: три фотографии и клип, один снимок пересвечен в черноту (в ролик он
    идёт наравне с остальными), один вычеркнут организатором, один непрочитан.
    """
    journal_ready(conn)
    ids = ready_material(conn, s3, memory, tmp_path)
    dark, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(crushed(tmp_path / "crushed.jpg")), storage.client().bucket, key)
    broken, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(unopenable(tmp_path / "broken.jpg")), storage.client().bucket, key)
    montage_id = order(
        conn,
        memory,
        {"target_seconds": 30, "music_track_id": "warm", "exclude_media_ids": [ids[0]]},
    )
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True
    assert read_montage(conn, montage_id)[0] == "ready"

    stages = stages_of(conn, montage_id)
    assert list(stages) == list(journal.STAGES), "этапы идут в порядке пайплайна"
    for name, stage in stages.items():
        assert stage["state"] == "done", name
        assert stage["started_at"] <= stage["finished_at"], f"{name}: сколько занял"
        assert stage["eta"] == 0, f"{name}: готовому этапу нечего ждать"

    # Подготовка: счётчики заказа и запись только на файл, который взять нельзя. Пересвеченный
    # снимок — не брак, а обычный кадр: в журнале о нём нет ни слова (doc-03, «What it doesn't»).
    prep = stages["prep"]
    assert (prep["done"], prep["total"]) == (6, 6)
    assert prep["outcome"] == {
        "photos": 5,
        "videos": 1,
        "photos_done": 5,
        "videos_done": 1,
        "bytes": 0,
        "senders": {str(memory[1]): 6},
        "video_seconds": pytest.approx(3.0, abs=0.5),
    }
    assert [
        (one.media_id, one.decision, one.reason) for one in entries_of(conn, montage_id, "prep")
    ] == [(broken, "skip", "unreadable")]

    # Разбор кадров: описаний нет ни у одного файла — этап проходит и честно говорит «0 описано
    # сейчас», а счётчик его стоит на шести: столько кадров пришлось бы описать.
    assert stages["analysis"]["outcome"] == {
        "described": 0,
        "earlier": 0,
        "faces": 0,
        "people": 0,
        "animals": 0,
    }
    assert entries_of(conn, montage_id, "analysis") == []

    # Режиссёрский этап: по файлу решение с причиной, у видео — какой кусок исходника взят.
    # Ни вычеркнутого, ни непрочитанного: первый убран человеком и в журнале не упоминается вовсе,
    # второго назвала подготовка. Пересвеченный — равный кадр, «беру» наравне с остальными.
    directing = entries_of(conn, montage_id, "directing")
    assert {one.media_id: one.decision for one in directing} == {
        one: "take" for one in (*ids[1:], dark)
    }
    # Частей истории у отбора нет — он их не придумывает, — и итог этапа о них молчит: ролик резал
    # не режиссёр, а счётчики «взял / мимо» считает API по записям (контракт api, «directing»).
    assert stages["directing"]["outcome"] == {}
    clip = next(one for one in directing if one.media_id == ids[3])
    assert clip.source_to_seconds > clip.source_from_seconds >= 0, "кусок клипа — секундами"
    still = next(one for one in directing if one.media_id == ids[1])
    assert (still.source_from_seconds, still.source_to_seconds) == (None, None), (
        "у фотографии куска исходника нет — она идёт в ролик целиком"
    )

    # Музыка: подо что подогнаны куски.
    assert stages["music"]["outcome"]["track"] == ["warm"]
    assert stages["music"]["outcome"]["cuts"] >= 0

    # Склейка: сколько склеек и какая длина.
    assert (stages["assembly"]["done"], stages["assembly"]["total"]) == (4, 4)
    assert stages["assembly"]["outcome"]["cuts"] == 3, "четыре куска — три склейки"
    assert stages["assembly"]["outcome"]["seconds"] == pytest.approx(
        read_montage(conn, montage_id)[2], abs=0.2
    )


def test_a_frame_the_algorithm_disliked_reaches_the_video_all_the_same(conn, s3, memory, tmp_path):
    """Кадры, оставленные организатором при выборе, равны: снимок, пересвеченный в черноту, отбор
    назвал бы браком — и он стоит в ролике наравне с остальными. «Включить» для этого больше не
    нужно: решение человека уже было, он его оставил."""
    journal_ready(conn)
    ready_material(conn, s3, memory, tmp_path)
    dark, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(crushed(tmp_path / "crushed.jpg")), storage.client().bucket, key)
    montage_id = order(conn, memory, {"target_seconds": 30, "music_tracks": []})
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True
    assert read_montage(conn, montage_id)[0] == "ready"
    assert dark in {media_id for media_id, _, _ in layout_of(conn, montage_id)}, (
        "пересвеченный снимок в ролике, а не выброшен за брак"
    )
    assert not [one for one in entries_of(conn, montage_id) if one.reason == "exposure"], (
        "и в журнале о нём нет ни слова"
    )


def test_a_montage_without_music_says_so_where_the_music_would_be(conn, s3, memory, tmp_path):
    """Ролик с живым звуком: этап музыки всё равно проходит, и итог у него честный — дорожки нет,
    подгонять было не подо что, ни один кусок не сдвинут."""
    journal_ready(conn)
    montage_id = order(conn, memory, {"target_seconds": 10, "music_tracks": []})
    for index in range(3):
        _, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(jpeg(tmp_path / f"photo-{index}.jpg")), storage.client().bucket, key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    stages = stages_of(conn, montage_id)
    assert stages["music"]["state"] == "done"
    assert stages["music"]["outcome"] == {"track": [], "cuts": 0}


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

    Смотрим дважды: перед самой склейкой (к этому времени подготовка и режиссёр уже всё рассказали)
    и после каждого готового куска — прогресс склейки должен двигаться, а не появиться разом.
    """
    journal_ready(conn)
    ready_material(conn, s3, memory, tmp_path)
    _, key = add_item(conn, memory, "photo", status="ready")
    s3.upload_file(str(unopenable(tmp_path / "broken.jpg")), storage.client().bucket, key)
    montage_id = order(conn, memory, {"target_seconds": 30, "music_tracks": []})

    before, after = [], []
    assembling = assembly.assemble

    def watched(pieces, orientation, track, workdir, out, on_clip=None, **kwargs):
        def watching(done):
            on_clip(done)
            after.append(from_another_connection(montage_id))

        before.append(from_another_connection(montage_id))
        return assembling(pieces, orientation, track, workdir, out, on_clip=watching, **kwargs)

    monkeypatch.setattr(assembly, "assemble", watched)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    status, entries, stages = before[0]
    assert status == "processing", "смотрели, пока ролик ещё собирался"
    assert [one for one in entries if one.stage == "prep"], "подготовка уже рассказала о себе"
    assert [one for one in entries if one.decision == "take"], "и режиссёр тоже"
    assert stages["prep"]["state"] == "done" and stages["assembly"]["state"] == "running"

    assert [stages["assembly"]["done"] for _, _, stages in after] == [1, 2, 3, 4], (
        "прогресс склейки идёт по готовому куску, а не появляется разом в конце"
    )
    assert all(stages["assembly"]["eta"] is not None for _, _, stages in after), (
        "у идущего этапа есть «осталось ≈», посчитанное по тому, как он идёт"
    )


# --- повтор, падение и цена ошибки -------------------------------------------


def test_a_montage_assembled_again_starts_its_journal_from_a_clean_sheet(conn, memory):
    """«Повторить» упавшего ролика и ретрай задачи — это вторая попытка, а не продолжение первой:
    иначе на экране шли бы две подготовки подряд и решения по одному кадру дважды."""
    journal_ready(conn)
    montage_id = order(conn, memory, {})
    journal.begin(conn, montage_id, 5, 60)
    journal.start(conn, montage_id, "prep", total=5)
    journal.entry(
        conn,
        montage_id,
        "prep",
        media_id=str(uuid.uuid4()),
        decision="skip",
        reason=selection.UNREADABLE,
    )
    journal.finish(conn, montage_id, "prep", photos_done=4, videos_done=1)

    journal.begin(conn, montage_id, 2, 30)

    assert entries_of(conn, montage_id) == []
    stages = stages_of(conn, montage_id)
    assert len(stages) == len(journal.STAGES)
    assert all(
        (one["state"], one["done"], one["total"], one["outcome"], one["started_at"])
        == ("pending", 0, None, {}, None)
        for one in stages.values()
    )
    assert stages["prep"]["eta"] == journal.seconds_for("prep", 2, 30), (
        "и оценка времени пересчитана по новому материалу"
    )


def test_a_montage_assembled_again_does_not_show_the_last_attempts_layout(
    conn, s3, memory, tmp_path, monkeypatch
):
    """Раскладка живёт и умирает вместе с журналом попытки. API присоединяет места кадров к записям
    журнала по одному `montage_id`, без статуса ролика, — и повтор, оставивший прошлую раскладку до
    своей, показывал бы кадр там, где его поставил ролик, которого уже нет."""
    journal_ready(conn)
    ids = ready_material(conn, s3, memory, tmp_path)
    montage_id = order(conn, memory, {"target_seconds": 30, "music_tracks": []})
    # Прошлая попытка дошла до склейки и упала: её журнал и её раскладка лежат в базе.
    journal.begin(conn, montage_id, len(ids), 30)
    journal.entry(
        conn, montage_id, "prep", media_id=ids[0], decision="skip", reason=selection.UNREADABLE
    )
    conn.execute(
        """INSERT INTO montage_frames (montage_id, media_id, start_seconds, duration_seconds)
           VALUES (%s, %s, 42, 3)""",
        (montage_id, ids[0]),
    )

    before_the_new_one = []
    writing = layout.write

    def watched(connection, ref, pieces):
        if pieces:
            with worker.connect() as watcher:
                before_the_new_one.append(layout_of(watcher, montage_id))
        return writing(connection, ref, pieces)

    monkeypatch.setattr(layout, "write", watched)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True
    assert read_montage(conn, montage_id)[0] == "ready"

    assert before_the_new_one[0] == [], "до новой раскладки прошлой нет"
    after = layout_of(conn, montage_id)
    assert {media_id for media_id, _, _ in after} == set(ids), "а после сборки — новая"
    assert after[0][1] == 0 and all(start != 42 for _, start, _ in after)


def test_a_montage_that_fell_leaves_the_stage_it_fell_on_marked_failed(conn, memory):
    """Упавший ролик оставляет журнал как есть — до какого места дошёл, видно, — а этап, который
    шёл в этот момент, помечен упавшим. Сделанное до него остаётся сделанным."""
    journal_ready(conn)
    montage_id = order(conn, memory, {})
    journal.begin(conn, montage_id, 5, 60)
    journal.start(conn, montage_id, "prep", total=5)
    journal.finish(conn, montage_id, "prep", photos_done=5)
    journal.start(conn, montage_id, "analysis", total=5)

    montage.give_up(conn, montage_id, "не удалось собрать ролик")

    stages = stages_of(conn, montage_id)
    assert read_montage(conn, montage_id)[0] == "failed"
    assert [stages[name]["state"] for name in journal.STAGES] == [
        "done",
        "failed",
        "pending",
        "pending",
        "pending",
    ]
    assert stages["analysis"]["eta"] is None, "упавшему этапу ждать нечего"


def test_a_journal_that_cannot_be_written_does_not_cost_the_montage(conn):
    """Журнал — зрелище, а не суть: ролик, который собирается минутами, не выбрасывают из-за
    строки о нём. Здесь ролика нет вовсе, так что не пройдёт ни одна запись."""
    journal_ready(conn)
    gone = str(uuid.uuid4())

    journal.begin(conn, gone, 5, 60)
    journal.start(conn, gone, "prep", total=5)
    journal.progress(conn, gone, "prep", 1, photos_done=1)
    journal.entry(
        conn, gone, "prep", media_id=str(uuid.uuid4()), decision="skip", reason=selection.UNREADABLE
    )
    journal.finish(conn, gone, "prep", photos_done=4)
    journal.failed(conn, gone)
    journal.entry(conn, gone, "prep", nonsense="колонки с таким именем в журнале нет")

    assert stages_of(conn, gone) == {} and entries_of(conn, gone) == []
