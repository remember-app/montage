"""Журнал монтажа целиком: этапы по порядку, чтение вживую и что остаётся после падения.

Что именно пишется в каждую реплику — `test_journal_entries.py`; контракт —
`docs/jobs/montage-journal.md`.

Real everything, как и в `test_montage.py`: настоящий Postgres, настоящий бакет, настоящий ffmpeg.
Журнал для того и пишется, чтобы его читали вживую, поэтому здесь есть и **второе соединение** —
то самое, которым его читает api, пока ролик ещё собирается.

Warning: эти тесты чистят таблицу `jobs` той базы, на которую смотрит DATABASE_URL, и пишут в
бакет, на который смотрит R2_ENDPOINT.
"""

import pathlib
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

    # Музыка: подо что подогнаны куски и где в ролике стоят склейки — треки в порядке заказа, а
    # секунды те же, что на раскладке, только без первой (перед началом ролика склейки нет).
    music = stages["music"]["outcome"]
    assert music["tracks"] == ["warm"]
    assert music["cut_seconds"] == pytest.approx(
        [start for _media, start, _seconds in layout_of(conn, montage_id)[1:]], abs=0.2
    )

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
    """Ролик с живым звуком: этап музыки всё равно проходит, и итог у него честный — дорожки нет и
    подгонять было не подо что, а склейки стоят там, где встали по смыслу: приложение по ним и
    рисует ленту под роликом."""
    journal_ready(conn)
    montage_id = order(conn, memory, {"target_seconds": 10, "music_tracks": []})
    for index in range(3):
        _, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(str(jpeg(tmp_path / f"photo-{index}.jpg")), storage.client().bucket, key)
    worker.enqueue(conn, "montage", montage_id)

    assert worker.run_once(conn) is True

    stages = stages_of(conn, montage_id)
    assert stages["music"]["state"] == "done"
    assert stages["music"]["outcome"] == {
        "tracks": [],
        "cut_seconds": [round(start, 2) for _m, start, _s in layout_of(conn, montage_id)[1:]],
    }
    assert "bpm" not in stages["music"]["outcome"], "темпа без дорожки нет — и строки нет"


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

    montage.give_up(conn, montage_id, "не удалось собрать ролик", "internal")

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
    # Код причины — на упавшем этапе, и когда код не назван, подставляется `internal`. Именно так
    # очередь кончает попытки (`worker._give_up` зовёт `give_up` без кода), и именно этим звонком
    # пользуются шесть отказов из семи: молчаливый упавший этап без причины — это поломка.
    assert stages["analysis"]["outcome"]["stopped"] == "internal"


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
    journal.failed(conn, gone, "internal")
    journal.entry(conn, gone, "prep", nonsense="колонки с таким именем в журнале нет")

    assert stages_of(conn, gone) == {} and entries_of(conn, gone) == []


# --- подшаги сборки и причина остановки --------------------------------------


def test_the_assembly_says_where_it_is_by_substeps_and_a_percent_that_only_grows():
    """Процент и подшаги — единственное, чем сборка говорит о себе по ходу, и оба должны быть
    правдой в каждую секунду: процент не уезжает назад, все четыре подшага написаны всегда, и
    ровно один из них `running` — тот, что приложение рисует под процентом.

    Замолчать нельзя: список без идущего подшага рисуется пустым, и ролик выглядит сломанным при
    воркере, который молчит. На закрытом этапе идущего подшага нет вовсе, и это верно — ждать
    больше нечего, все галочки стоят, а процент ровно 100.
    """
    walked = (
        [journal.assembling("clips", done, 4) for done in range(5)]
        + [journal.assembling(step) for step in ("sound", "cover", "file")]
        + [journal.assembling(None)]
    )

    percents = [one["percent"] for one in walked]
    assert percents == sorted(percents), "процент сборки не уезжает назад"
    assert percents[0] == 0 and percents[-1] == 100, "от нуля и до ста"
    assert sum(journal.ASSEMBLY_WEIGHTS.values()) == 100, "подшаги делят процент нацело"

    for came in walked:
        assert [one["step"] for one in came["steps"]] == list(journal.ASSEMBLY_STEPS)
        assert len([one for one in came["steps"] if one["state"] == "running"]) in (0, 1)
    assert walked[-1]["steps"] == [
        {"step": code, "state": "done"} for code in journal.ASSEMBLY_STEPS
    ], "закрытый этап — все подшаги сделаны"


# --- причина остановки на каждом месте отказа --------------------------------


def _ordered(conn, s3, memory, tmp_path, monkeypatch, **params):
    """Заказ и очередь, как их оставляет api, — дальше отказ случается сам."""
    montage_id = order(conn, memory, params)
    worker.enqueue(conn, "montage", montage_id)
    return montage_id


def _ready_then_ordered(conn, s3, memory, tmp_path, monkeypatch, **params):
    """Годного материала хватает, поэтому отказ случается дальше по конвейеру, а не на нижнем
    пороге заказа."""
    ready_material(conn, s3, memory, tmp_path)
    return _ordered(conn, s3, memory, tmp_path, monkeypatch, **params)


def _unreadable_three(conn, s3, memory, tmp_path, monkeypatch, **params):
    """Три файла, которые ffmpeg не откроет: годных не остаётся ни одного."""
    for number in range(3):
        _item, key = add_item(conn, memory, "photo", status="ready")
        s3.upload_file(
            str(unopenable(tmp_path / f"broken-{number}.jpg")), storage.client().bucket, key
        )
    return _ordered(conn, s3, memory, tmp_path, monkeypatch, **params)


def _refuse_nothing_fits(conn, s3, memory, tmp_path, monkeypatch, **params):
    """Всё, что есть, на «исключить» — но довести до этого проще подменой выбора: `survived` такой
    заказ отбил бы раньше сам, а ветку «в ролик нечего ставить» надо всё-таки прогнать."""
    ready_material(conn, s3, memory, tmp_path)
    monkeypatch.setattr(selection, "choose", lambda *_: ([], []))
    return _ordered(conn, s3, memory, tmp_path, monkeypatch, target_seconds=30, **params)


def _refuse_out_of_attempts(conn, s3, memory, tmp_path, monkeypatch, **params):
    """Очередь исчерпала попытки: сборка падает на склейке, `worker` сам назовёт ролик упавшим."""
    ready_material(conn, s3, memory, tmp_path)
    montage_id = _ordered(conn, s3, memory, tmp_path, monkeypatch, target_seconds=30, **params)

    def falls(*_args, **_kwargs):
        raise RuntimeError("ffmpeg fell over")

    monkeypatch.setattr(assembly, "assemble", falls)
    conn.execute("UPDATE jobs SET attempts = %s WHERE type = 'montage'", (worker.MAX_ATTEMPTS - 1,))
    return montage_id


# Каждое место, где ролик признан негодным: как довести до него и что он должен сказать. `stage` —
# та стадия, на которой код обязан лечь, а `None` — отказ до журнала, когда этапов нет вовсе и
# причину видит человек в `error_text` под роликом.
REFUSALS = (
    pytest.param(_ordered, {"music_tracks": ["не-трек"]}, "internal", None, id="unknown-track"),
    pytest.param(_ordered, {"orientation": "квадрат"}, "internal", None, id="unknown-orientation"),
    pytest.param(_ordered, {}, "internal", None, id="no-material-at-all"),
    pytest.param(_unreadable_three, {}, "no_material", "prep", id="nothing-usable"),
    pytest.param(
        _ready_then_ordered,
        {"target_seconds": 30, "music_tracks": ["music/нет-такого"]},
        "track_unreadable",
        "music",
        id="own-track-unreadable",
    ),
    pytest.param(_refuse_nothing_fits, {}, "no_material", "directing", id="nothing-to-put-in"),
    pytest.param(_refuse_out_of_attempts, {}, "internal", "assembly", id="out-of-attempts"),
)


@pytest.mark.parametrize("refused, params, code, stage", REFUSALS)
def test_every_place_the_montage_is_refused_names_its_reason(
    conn, s3, memory, tmp_path, monkeypatch, refused, params, code, stage
):
    """Каждое место отказа называет причину **своим** кодом и кладёт её туда, где что-то было.

    Код у `give_up` обязателен, и забытый — падение задания, а не отказ. Но «код назван» ещё не
    значит «код проверен»: место, которого нет в тесте, выглядит покрытым ровно так же, как
    покрытое. Поэтому мест столько же, сколько строк в `REFUSALS`, и это проверяется отдельно
    (второй тест ниже) — иначе список протухнет молча.

    Куда ложится код — тоже часть проверки. Упавшей делается та стадия, **чья проверка упала**, и
    обязательно та, что в этот момент идёт: «годных кадров меньше трёх» спрашивает подготовка
    (до её закрытия), «свой трек не открылся» — музыка (после её начала), а не разбор, который к
    дорожке не при чём. Стадия, которая не началась, остаётся не начатой: подписывать упавшей то,
    что не ломалось, значило бы написать в журнале то, чего не было.
    """
    journal_ready(conn)
    montage_id = refused(conn, s3, memory, tmp_path, monkeypatch, **params)

    assert worker.run_once(conn) is True
    status, _key, _duration, error_text = read_montage(conn, montage_id)
    assert status == "failed", "отказ, а не падение задания"

    stages = stages_of(conn, montage_id)
    if stage is None:
        # До журнала: ложить причину некуда, и её видит человек в `error_text`.
        assert error_text, "причина названа словами под роликом"
        assert stages == {}, "журнала до отказа не было — и это верно"
        return

    fell = [name for name in journal.STAGES if stages[name]["state"] == "failed"]
    assert fell == [stage], f"причина лежит на {stage}, а не на другой стадии"
    assert stages[stage]["outcome"]["stopped"] == code
    for name in journal.STAGES:
        if name not in fell and stages[name]["state"] == "pending":
            assert "stopped" not in stages[name]["outcome"], "пустая стадия причины не получает"


def test_the_list_of_refusals_covers_every_place_in_the_source(conn):
    """Сколько мест отказа в исходнике — столько строк в `REFUSALS`. Без этой сверки список
    протухает молча: кто добавит восьмое место и не допишет строку, тот получит зелёный набор и
    место, выглядящее покрытым. Расхождение валит тест, пока строка не добавлена.
    """
    journal_ready(conn)
    root = pathlib.Path(montage.__file__).parents[1]  # `montage/video/montage.py` → `montage/`
    written = sum(
        1
        for name in ("video/montage.py", "worker.py")
        for line in (root / name).read_text().splitlines()
        if ("_give_up_for_good(" in line or "montages.give_up(" in line)
        and not line.strip().startswith("def ")
    )
    assert written == len(REFUSALS), (
        f"в исходнике мест отказа {written}, а в списке {len(REFUSALS)} — "
        "новое место не описано или место выкинули"
    )
