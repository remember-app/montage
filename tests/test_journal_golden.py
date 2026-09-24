"""Журнал монтажа насквозь: настоящая сборка целиком и «золотой» снимок того, что она записала.

Снимок — мост между сервисами. Его читает не Python: api ведёт его через свой маршрут журнала, а
приложение — через свои экраны, и оба должны видеть ровно то, что воркер пишет на самом деле, а не
то, что о нём когда-то предположил автор чужого теста. Поэтому снимок снимается с **настоящей**
сборки (настоящие Postgres, бакет и ffmpeg, режиссёр-нейросеть — записанным ответом из `fixtures/`,
как во всём наборе), а тест стережёт его: воркер начал писать журнал иначе — тест падает и
показывает разницу. Формат и способ перезаписать — README, «Золотой снимок журнала».

Warning: эти тесты чистят таблицу `jobs` той базы, на которую смотрит DATABASE_URL, и пишут в
бакет, на который смотрит R2_ENDPOINT.
"""

import dataclasses
import datetime
import difflib
import json
import os
import pathlib

import pytest

from montage import ffmpeg, journal, storage, worker
from montage.video import assembly

from material import crushed, draw, jpeg, unopenable
from recorded import planning_over
from rows import add_item, order, read_montage

GOLDEN = pathlib.Path(__file__).with_name("fixtures")
UPDATE = "MONTAGE_UPDATE_GOLDEN"

FORMAT = 1
EPOCH = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
FIRST_SHOT = datetime.datetime(2026, 6, 1, 18, 0, tzinfo=datetime.timezone.utc)


@pytest.fixture(autouse=True)
def nobody_to_call(monkeypatch):
    """Ни поставщика нейросети, ни api для пуша: что бы ни лежало в окружении разработчика, сюда
    оно не попадёт. Режиссёр там, где он нужен, — записанный ответ (`planning_over`)."""
    for name in (
        "AI_PROVIDER",
        "GEMINI_API_KEY",
        "OPENAI_API_KEY",
        "API_URL",
        "INTERNAL_WORKER_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def journal_ready(conn):
    for table in ("montage_stages", "montage_journal_entries", "montage_frames"):
        if conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0] is None:
            pytest.skip(f"{table} is missing — run the api's migrations first (see README.md)")


# --- материал -----------------------------------------------------------------


def flat(path):
    """Ровный серый кадр: открывается, освещён как надо, и в нём нечему быть резким — «смазан»."""
    draw(
        "-f",
        "lavfi",
        "-i",
        "color=c=0x808080:size=1200x800:duration=1",
        "-frames:v",
        "1",
        str(path),
    )
    return path


def soft(path):
    """Фотография, которая годится, но нарочно мягче клипа: в отборе по оценке клип идёт раньше
    неё, и «мимо, не влезло» достаётся фото, а не ему."""
    draw(
        "-f",
        "lavfi",
        "-i",
        "testsrc=size=1200x800:duration=1",
        "-vf",
        "gblur=sigma=8",
        "-frames:v",
        "1",
        str(path),
    )
    return path


def sharp_at_four(path, seconds=8):
    """Клип, в котором резкий только один кадр — на четвёртой секунде, всё остальное размыто. Лучший
    момент у него один и стоит не с краю, поэтому кусок, который взят из исходника, начинается не
    с нуля — и «беру 0:02–0:06» в журнале называет настоящий кусок, а не начало файла."""
    draw(
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size=640x360:rate=12:duration={seconds}",
        "-f",
        "lavfi",
        "-i",
        f"sine=frequency=440:duration={seconds}",
        "-vf",
        "gblur=sigma=6:enable='not(between(t,3.5,4.5))'",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        str(path),
    )
    return path


@dataclasses.dataclass
class File:
    label: str
    id: str
    kind: str
    at: float  # секунды от первого кадра до съёмки этого
    duration: float | None = None
    described: str | None = None
    cast: tuple = ()


VIDEOS = (sharp_at_four,)


def lay_out(conn, s3, memory, tmp_path, shots, first=1):
    """Материал воспоминания: по строке `shots` на файл, в порядке съёмки — он же порядок меток.

    Каждая строка: `(что за файл, секунда съёмки, описание, кто в кадре)`. «Что за файл» — функция,
    которая рисует его на диск (`jpeg`, `crushed`, …) или `None` — клип. Файл кладётся в бакет
    готовым (`ready`), дата съёмки ставится явно: порядок ролика и серии дедупа не должны зависеть
    от того, в какую миллисекунду вставилась строка.
    """
    files = []
    for number, (make, second, described, cast) in enumerate(shots, start=first):
        kind = "video" if make in VIDEOS else "photo"
        item_id, key = add_item(conn, memory, kind, status="ready")
        path = make(tmp_path / f"file-{number}.{'mp4' if kind == 'video' else 'jpg'}")
        s3.upload_file(str(path), storage.bucket(), key)
        duration = round(ffmpeg.duration(path), 1) if kind == "video" else None
        conn.execute(
            "UPDATE media_items SET captured_at = %s, duration_seconds = %s WHERE id = %s",
            (FIRST_SHOT + datetime.timedelta(seconds=second), duration, item_id),
        )
        if described:
            conn.execute(
                """INSERT INTO media_understandings (media_id, memory_id, description, cast_labels)
                   VALUES (%s, %s, %s, %s)""",
                (item_id, str(memory[0]), described, list(cast)),
            )
        files.append(File(f"f{number}", item_id, kind, second, duration, described, cast))
    return files


# --- снимок -------------------------------------------------------------------


def snapshot(conn, montage_id, files, params, scenario):
    """Всё, что ролик и его журнал оставили в базе, — JSON, который читают, не запуская Python.

    Метки `f1`, `f2`… вместо uuid файлов (по порядку съёмки), порядковые номера `n` вместо bigserial
    записей, а метки времени — секундами от условной эпохи по месту во времени: кто раньше кого
    легла, видно, сколько это заняло, — нет. Всё остальное — как в базе, столбец в столбец.
    """
    label = {one.id: one.label for one in files}
    marked = lambda value: label[str(value)] if value is not None else None  # noqa: E731

    status, duration = conn.execute(
        "SELECT status, duration_seconds FROM montages WHERE id = %s", (montage_id,)
    ).fetchone()
    stages = conn.execute(
        """SELECT stage, state, started_at, finished_at, eta_seconds, done_count, total_count,
                  outcome, updated_at
             FROM montage_stages WHERE montage_id = %s""",
        (montage_id,),
    ).fetchall()
    entries = conn.execute(
        """SELECT stage, media_id, thought, decision, reason, series_size, story_part,
                  story_part_title, source_from_seconds, source_to_seconds, labels,
                  focus_x, focus_y, focus_w, focus_h, created_at
             FROM montage_journal_entries WHERE montage_id = %s ORDER BY id""",
        (montage_id,),
    ).fetchall()
    layout = conn.execute(
        """SELECT media_id, start_seconds, duration_seconds FROM montage_frames
            WHERE montage_id = %s ORDER BY start_seconds, media_id""",
        (montage_id,),
    ).fetchall()

    entries = _understanding_in_file_order(
        entries, {one.id: number for number, one in enumerate(files)}
    )
    # Все метки времени одной сборки — на одну шкалу: порядок сохранён, значения — нет.
    moments = sorted(
        {
            at
            for at in (
                *(when for row in stages for when in (row[2], row[3], row[8]) if when),
                *(row[-1] for row in entries),
            )
        }
    )
    when = lambda at: (  # noqa: E731
        None
        if at is None
        else (EPOCH + datetime.timedelta(seconds=moments.index(at))).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    by_stage = {row[0]: row for row in stages}
    return {
        "format": FORMAT,
        "scenario": scenario,
        "montage": {
            "status": status,
            "duration_seconds": round(duration, 2) if duration is not None else None,
            "params": _params_by_label(params, label),
        },
        "files": [
            {
                "label": one.label,
                "kind": one.kind,
                "position": position,
                "captured_after_seconds": one.at,
                "duration_seconds": one.duration,
            }
            for position, one in enumerate(files, start=1)
        ],
        "stages": [
            {
                "stage": name,
                "state": row[1],
                "started_at": when(row[2]),
                "finished_at": when(row[3]),
                "eta_seconds": row[4],
                "done_count": row[5],
                "total_count": row[6],
                "outcome": row[7],
                "updated_at": when(row[8]),
            }
            for name in journal.STAGES
            for row in [by_stage[name]]
        ],
        "entries": [
            {
                "n": number,
                "stage": row[0],
                "media": marked(row[1]),
                "thought": row[2],
                "decision": row[3],
                "reason": row[4],
                "series_size": row[5],
                "story_part": row[6],
                "story_part_title": row[7],
                "source_from_seconds": row[8],
                "source_to_seconds": row[9],
                "labels": row[10],
                "focus": (
                    {"x": row[11], "y": row[12], "w": row[13], "h": row[14]}
                    if row[11] is not None
                    else None
                ),
                "created_at": when(row[-1]),
            }
            for number, row in enumerate(entries, start=1)
        ],
        # Место каждого кадра в ролике, по порядку в ролике.
        "layout": [
            {
                "media": marked(media_id),
                "start_seconds": round(start, 2) + 0.0,
                "duration_seconds": round(seconds, 2) + 0.0,
            }
            for media_id, start, seconds in layout
        ],
    }


def _params_by_label(params, label):
    """Заказ так, как его положил api, — только id файлов в нём переписаны на метки."""
    return {
        name: [label[one] for one in value] if name.endswith("_media_ids") else value
        for name, value in params.items()
    }


def _understanding_in_file_order(entries, position):
    """Записи «что на кадрах» — по файлам в порядке съёмки, а не в том, в каком их отдала база.

    Воркер читает готовые описания одним запросом без порядка (`story.what_was_seen`), и как
    они лягут друг за другом, решает физический порядок строк в таблице, а не сборка: у двух
    прогонов на одной базе он бывает разный. Между собой эти записи ничего не говорят (по записи
    на файл, одной пачкой), а снимок, шатающийся от порядка строк в куче, не сторож. Метки времени
    остаются на своих местах в пачке — порядок, в котором стоят записи, а не файлы.
    """
    seen = [index for index, row in enumerate(entries) if row[0] == "understanding"]
    ordered = sorted((entries[index] for index in seen), key=lambda row: position[str(row[1])])
    entries = list(entries)
    for index, row in zip(seen, ordered):
        entries[index] = (*row[:-1], entries[index][-1])
    return entries


# Длины в ролике меряются по готовому файлу: каждый кусок кодируется в кадры, а склейки сдвигаются
# под биты дорожки (librosa). У другой сборки ffmpeg или другой версии библиотеки они разойдутся на
# кадр-другой — это не смена того, как воркер пишет журнал. Расхождение больше — уже смена.
TOLERANCE_SECONDS = 0.2


def _within_a_frame_or_two(kept, made):
    """Закоммиченный снимок, у которого измеренные длины подтянуты к свежим, если те не дальше
    `TOLERANCE_SECONDS`: всё остальное сравнивается слово в слово, а эти числа — с допуском."""

    def near(old, new, key):
        if (
            None not in (old.get(key), new.get(key))
            and abs(old[key] - new[key]) <= TOLERANCE_SECONDS
        ):
            old[key] = new[key]

    near(kept["montage"], made["montage"], "duration_seconds")
    for old, new in zip(kept["stages"], made["stages"]):
        if old["stage"] == new["stage"] == "assembly":
            near(old["outcome"], new["outcome"], "seconds")
    if len(kept["layout"]) == len(made["layout"]):
        for old, new in zip(kept["layout"], made["layout"]):
            for key in ("start_seconds", "duration_seconds"):
                near(old, new, key)
    return kept


def compare_with_golden(name, made):
    """Свежий снимок против закоммиченного: расхождение падает и показывает разницу построчно.

    Перезаписать — только явным действием (`MONTAGE_UPDATE_GOLDEN=1`), и тест в этот раз не
    зелёный, а пропущенный: зелёным «прошёл» стать может только сравнение, а не перезапись.
    """
    __tracebackhide__ = True
    path = GOLDEN / name
    wrote = json.dumps(made, ensure_ascii=False, indent=2) + "\n"
    if os.environ.get(UPDATE):
        path.write_text(wrote, encoding="utf-8")
        pytest.skip(f"{name} переписан ({UPDATE} стоит в окружении) — смотри git diff")
    if not path.exists():
        pytest.fail(
            f"нет снимка {path.name}: снять его — {UPDATE}=1 pytest {pathlib.Path(__file__).name}"
        )
    kept = path.read_text(encoding="utf-8")
    kept = (
        json.dumps(_within_a_frame_or_two(json.loads(kept), made), ensure_ascii=False, indent=2)
        + "\n"
    )
    if kept != wrote:
        difference = "".join(
            difflib.unified_diff(
                kept.splitlines(keepends=True),
                wrote.splitlines(keepends=True),
                f"{path.name} (закоммичен)",
                f"{path.name} (свежая сборка)",
                n=2,
            )
        )
        pytest.fail(
            f"воркер пишет журнал иначе, чем в снимке {path.name}. Если так и задумано — "
            f"{UPDATE}=1 перезапишет снимок, но api и приложение читают его тоже.\n{difference}"
        )


def assemble(conn, memory, files, params):
    """Заказ и одна настоящая сборка через очередь, как её берёт воркер."""
    montage_id = order(conn, memory, params)
    worker.enqueue(conn, "montage", montage_id)
    assert worker.run_once(conn) is True
    assert read_montage(conn, montage_id)[0] == "ready"
    return montage_id


# --- сборки -------------------------------------------------------------------


def test_the_journal_of_a_montage_directed_by_the_model_is_the_recorded_one(
    conn, s3, memory, tmp_path, monkeypatch
):
    """Ролик, который режиссировала нейросеть (записанным ответом), с историей и всем, что при этом
    случается с кадрами: взят по плану — с мыслью, частью и рамкой; взят «из серии»; взят по слову
    организатора вопреки отсеву; остался за бортом со словами модели и без слов; не влез в
    заказанную длину; вычеркнут организатором; смазан, тёмен и не открылся на отсеве.
    """
    files = lay_out(
        conn,
        s3,
        memory,
        tmp_path,
        [
            (jpeg, 0, "накрывают на стол", ("человек-1", "человек-2")),
            (jpeg, 60, "вносят торт", ("человек-1", "человек-3")),
            (sharp_at_four, 120, "чокаются, все смеются", ("человек-2", "человек-3")),
            (jpeg, 180, "задувают свечи", ("человек-3",)),
            (jpeg, 181, "задувают свечи, ближе", ("человек-3",)),
            (jpeg, 240, "собака под столом", ("животное-1",)),
            (jpeg, 300, "пустая тарелка", ()),
            (jpeg, 360, "салют во дворе", ()),
            (jpeg, 420, "гость на пороге", ("человек-4",)),
            (crushed, 480, "тёмный кадр после тоста", ("человек-2",)),
            (crushed, 540, None, ()),
            (unopenable, 600, None, ()),
            (flat, 660, None, ()),
        ],
    )
    planning_over(monkeypatch, [one.id for one in files], "gemini_plan_journal.json")
    params = {
        "target_seconds": 15,
        "music_tracks": [],
        "story_text": "день рождения, всё вокруг торта",
        "exclude_media_ids": [files[8].id],
        "include_media_ids": [files[9].id],
    }

    montage_id = assemble(conn, memory, files, params)

    compare_with_golden(
        "journal_director.json", snapshot(conn, montage_id, files, params, "director")
    )


def test_the_journal_of_a_montage_cut_by_the_algorithm_is_the_recorded_one(
    conn, s3, memory, tmp_path
):
    """Тот же ролик там, где нейросети нет вовсе (так собирается каждый ролик, пока нет ключа):
    решения
    ставит отсев — «взял 1 из 3» на серии, «мимо» с кодом на каждом, кого он не взял, — и дорожка
    ритма с настоящей музыкой полки. Описания кадров при этом есть: этап «разбираем» их
    пересказывает.
    """
    files = lay_out(
        conn,
        s3,
        memory,
        tmp_path,
        [
            (soft, 0, "Бабушка смеётся", ("человек-1",)),
            (soft, 1, "Бабушка смеётся, Лиза рядом", ("человек-1", "человек-2")),
            (soft, 2, "Лиза обнимает бабушку", ("человек-1", "человек-2")),
            (soft, 60, "стол", ()),
            (sharp_at_four, 120, "тост", ("человек-2", "животное-1")),
            (soft, 180, None, ()),
            (soft, 240, None, ()),
            (soft, 300, "чужой человек в кадре", ("человек-3",)),
            (crushed, 360, "тёмный кадр", ()),
            (crushed, 420, None, ()),
            (unopenable, 480, None, ()),
            (flat, 540, None, ()),
        ],
    )
    params = {
        "target_seconds": 10,
        "music_tracks": ["warm"],
        "exclude_media_ids": [files[7].id],
        "include_media_ids": [files[8].id],
    }

    montage_id = assemble(conn, memory, files, params)

    compare_with_golden(
        "journal_algorithm.json", snapshot(conn, montage_id, files, params, "algorithm")
    )


# --- с чистого листа ----------------------------------------------------------


def test_a_montage_that_fell_and_was_taken_again_says_only_what_its_second_attempt_did(
    conn, s3, memory, tmp_path, monkeypatch
):
    """Ретрай той же задачи на настоящей сборке: первая попытка дошла до склейки и упала, и её
    журнал,
    этапы и раскладка лежат в базе; пока ролик ждал, в воспоминание добавили кадр — вторая попытка
    собирает уже другой материал. После неё в базе ровно то, что записал бы ролик, собранный
    впервые из этого же материала: ни записей первой попытки, ни двух отсевов подряд, ни этапа,
    застрявшего идущим.

    Куски по отдельности стережёт `test_journal.py`: журнал — `journal.begin`, раскладку — сборка
    поверх прошлой раскладки, которую до новой не видно. Здесь сверено целое, и у второй попытки, и
    у чистого заказа рядом — одной и той же рукой.
    """
    shots = [
        (jpeg, 60, "стол", ("человек-1",)),
        (jpeg, 120, "торт", ("человек-1", "человек-2")),
        (crushed, 180, None, ()),
        (jpeg, 240, None, ()),
    ]
    files = lay_out(conn, s3, memory, tmp_path, shots)
    params = {"target_seconds": 30, "music_tracks": []}
    montage_id = order(conn, memory, params)

    assembling = assembly.assemble

    def falls_once(*args, **kwargs):
        monkeypatch.setattr(assembly, "assemble", assembling)
        raise RuntimeError("ffmpeg fell over")

    monkeypatch.setattr(assembly, "assemble", falls_once)
    worker.enqueue(conn, "montage", montage_id)
    assert worker.run_once(conn) is True  # упала: задача ушла на повтор, ролик не `ready`

    def count(table):
        return conn.execute(
            f"SELECT count(*) FROM {table} WHERE montage_id = %s", (montage_id,)
        ).fetchone()[0]

    assert count("montage_journal_entries") > 0 and count("montage_frames") > 0, (
        "первая попытка оставила журнал и раскладку"
    )
    assert read_montage(conn, montage_id)[0] == "processing"

    # Пока ролик ждал повтора, в воспоминание добавили кадр — раньше всех остальных по времени
    # съёмки, так что каждый кадр ролика теперь стоит не там, где стоял.
    files = (
        lay_out(conn, s3, memory, tmp_path, [(jpeg, -60, "гости", ())], first=len(files) + 1)
        + files
    )
    files.sort(key=lambda one: one.at)
    files = [
        dataclasses.replace(one, label=f"f{number}") for number, one in enumerate(files, start=1)
    ]
    conn.execute("UPDATE jobs SET run_after = now() WHERE type = 'montage'")
    assert worker.run_once(conn) is True
    assert read_montage(conn, montage_id)[0] == "ready"

    retried = snapshot(conn, montage_id, files, params, "retry")
    fresh_id = assemble(conn, memory, files, params)
    fresh = snapshot(conn, fresh_id, files, params, "retry")

    assert retried == fresh, "вторая попытка записала не то, что записал бы чистый ролик"
    assert {row["media"] for row in retried["layout"]} == {"f1", "f2", "f3", "f5"}
    assert retried["stages"][-1]["state"] == "done"
