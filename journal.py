"""Журнал монтажа: что воркер рассказывает о сборке, пока она идёт.

Пять этапов ролика и реплики на каждом — «что за кадр держу в руках, беру или прохожу мимо и
почему» (doc-03, «Монтаж в прямом эфире»). Рассказ лежит вместе с роликом и читается и через
полгода; экраны — `design/screens/MontageLive.html`, `MontageLiveStage.html`, `MontageLiveDone.html`.

Схему завёл API (его миграция `130.do.montage-journal.sql`), контракт колонок — в его README,
раздел «Журнал монтажа»; воркер сюда только пишет, читает и отдаёт приложению API. Этот файл —
тонкий писатель и ничего больше: **что** записать, решает оркестратор (`montage.py`) по тому, что
вернули модули, а отсев, режиссёр, ритм и сборка о журнале не знают вовсе — иначе замена любого из
них тащила бы за собой рассказ о нём.

Слов для человека здесь нет: в записи идут коды и числа (`skip`, `blur`, «1 из 4»), а фразу из них
строит приложение на своём языке. Единственное исключение — `thought`, мысль модели её же словами;
её пишет режиссёр-нейросеть, и место для неё — общий `entry` ниже.

**Живьём, а не пачкой в конце.** Приложение опрашивает журнал раз в секунду-две, пока ролик
собирается, поэтому каждая запись уходит отдельной командой и ложится в базу сразу: соединение
воркера в автокоммите (`worker.connect`), и никакой транзакции вокруг сборки нет. Отсюда же и
размер записей — запись на отсеянный файл и одно небольшое `UPDATE` на готовый кусок, а не строка
на каждое движение.

**Журнал — зрелище, а не суть.** Ролик собирается минутами, и ни одна из этих записей не стоит
того, чтобы его уронить: всё, что пишется отсюда, ошибку проглатывает и оставляет строку в логе
(`quiet`).
"""

import functools
import json
import logging

# Ручка оценки времени: сколько секунд этап тратит на один файл материала и на одну секунду
# готового ролика. Числа — «на глаз», и подстраивать их по реальным прогонам надо здесь: больше
# оценка времени нигде не считается.
#
# Сумма по всем пяти этапам держится равной тому, чем API обещает «соберём за N минут» на итоге
# заказа (`estimateMinutes`, README API, `POST …/montages/preview`): **3 секунды на кадр и 1,5
# секунды на секунду готового ролика**. Обещание перед «Начать монтаж» и «осталось ≈» на экране
# монтажа считаются по одним и тем же числам, иначе ролик, обещанный за 15 минут, шёл бы на экране
# к сорока. Двигать их надо вместе — сумма пинится тестом (`test_journal.py`).
ESTIMATE = {
    "screening": (1.5, 0.0),      # скачать каждый файл и просмотреть его кадр за кадром
    "understanding": (0.8, 0.0),  # без модели — чтение готовых описаний; с ней это её ответ
    "directing": (0.5, 0.0),      # отсев считает мгновенно, режиссёр-нейросеть — один запрос
    "rhythm": (0.2, 0.1),         # разбор дорожки librosa: по длине ролика, а не по материалу
    "assembly": (0.0, 1.4),       # нормализация кусков и кросс-фейды — вся остальная минута
}

# Этапы в порядке пайплайна — тот же, в котором их рисует экран, и тот же, что в контракте.
STAGES = tuple(ESTIMATE)

# Колонки записи, которые заполняет воркер (README API, «Запись журнала»). Здесь же место для
# режиссёра-нейросети: мысль, часть истории с названием, рамка «куда смотрит» и метки «что увидел»
# кладутся тем же `entry`, новых функций для них не нужно.
COLUMNS = (
    "media_id", "thought", "decision", "reason", "series_size", "story_part", "story_part_title",
    "source_from_seconds", "source_to_seconds", "labels", "focus_x", "focus_y", "focus_w", "focus_h",
)

# Потолки из контракта. Стоят `CHECK`-ами в базе, поэтому запись, вышедшую за них, база отвергнет
# целиком — обрезаем здесь, чтобы длинное описание стоило журналу хвоста фразы, а не всей реплики.
THOUGHT_CHARACTERS = 500
REASON_CHARACTERS = 40
TITLE_CHARACTERS = 80
LABELS_COUNT, LABELS_CHARACTERS = 12, 240

log = logging.getLogger("montage.journal")


def quiet(write):
    """Запись, которая не легла, стоит строки в логе и ничего больше — сборка идёт дальше.

    Тем же швом, что и раскладка кадров (`montage._write_layout`): рассказ о ролике не может
    стоить ролика. Соединение воркера в автокоммите, так что неудавшаяся команда не оставляет за
    собой оборванной транзакции — следующая запись ляжет как ни в чём не бывало.

    Публичный, потому что журнал пишет и оркестратор: то, что он собирает для журнала сам (чтение
    описаний, например), падать на ролик не должно ровно так же.
    """
    @functools.wraps(write)
    def written(*args, **kwargs):
        try:
            return write(*args, **kwargs)
        except Exception as exc:
            log.warning("journal: %s did not land: %s", write.__name__, exc)
    return written


def seconds_for(stage, files, target_seconds):
    """Оценка этапа в секундах — по числу файлов материала и заказанной длине ролика."""
    per_file, per_second = ESTIMATE[stage]
    return round(per_file * files + per_second * target_seconds)


@quiet
def begin(conn, ref, files, target_seconds):
    """Пять этапов ролика с оценкой времени на каждом — и чистый лист.

    Зовётся в начале сборки, до первого скачанного файла: список этапов на экране строится по этим
    строкам, и до них рассказывать нечего. Повтор того же ролика — ретрай задачи или «повторить»
    упавшего — начинает журнал заново: старые реплики удаляются, этапы возвращаются в ожидание.
    Иначе вторая попытка дописывала бы свои решения к решениям первой, и на экране шли бы два
    отсева подряд.

    Этапы не удаляются, а переписываются на месте: между `DELETE` и `INSERT` читающий API увидел бы
    ролик вовсе без журнала, то есть «этот ролик собран до того, как журнал появился».
    """
    conn.execute("DELETE FROM montage_journal_entries WHERE montage_id = %s", (ref,))
    with conn.cursor() as cursor:
        cursor.executemany(
            """INSERT INTO montage_stages (montage_id, stage, eta_seconds) VALUES (%s, %s, %s)
               ON CONFLICT (montage_id, stage) DO UPDATE
                 SET state = 'pending', started_at = null, finished_at = null,
                     eta_seconds = excluded.eta_seconds, done_count = 0, total_count = null,
                     outcome = '{}', updated_at = now()""",
            [(ref, stage, seconds_for(stage, files, target_seconds)) for stage in STAGES],
        )


@quiet
def start(conn, ref, stage, total=None):
    """Этап пошёл: из ожидания в работу, и сколько ему считать (`total_count`)."""
    conn.execute(
        """UPDATE montage_stages
              SET state = 'running', started_at = now(), finished_at = null,
                  total_count = coalesce(%s, total_count), updated_at = now()
            WHERE montage_id = %s AND stage = %s""",
        (total, ref, stage),
    )


@quiet
def progress(conn, ref, stage, done, total=None):
    """«61 из 212 разложено» — и сколько этапу осталось по факту.

    Оценка идущего этапа считается не по константам, а по тому, как он идёт на самом деле:
    потраченное время на сделанное, умноженное на оставшееся. Первая оценка (`begin`) — догадка по
    числу файлов, эта — уже измерение, и на медленной машине «осталось ≈» уезжает вместе с ней.
    """
    conn.execute(
        """UPDATE montage_stages
              SET done_count = %(done)s,
                  total_count = coalesce(%(total)s, total_count),
                  eta_seconds = CASE
                      WHEN %(done)s > 0 AND started_at IS NOT NULL
                           AND coalesce(%(total)s, total_count) >= %(done)s
                      THEN greatest(round(extract(epoch FROM now() - started_at) / %(done)s
                                          * (coalesce(%(total)s, total_count) - %(done)s)), 0)
                      ELSE eta_seconds END,
                  updated_at = now()
            WHERE montage_id = %(ref)s AND stage = %(stage)s""",
        {"done": done, "total": total, "ref": ref, "stage": stage},
    )


@quiet
def finish(conn, ref, stage, **outcome):
    """Этап кончился, и вот его итог — «212 из 214 годятся», «3 части, 61 кадр».

    Ключи итога — на этап свои (README API, «Этапы»), фразу из них строит приложение; пустое
    значение не пишется вовсе — отсутствующий ключ экран просто не рисует. Сколько этап занял,
    считается по `started_at` и `finished_at`, отдельной колонки для этого нет.
    """
    conn.execute(
        """UPDATE montage_stages
              SET state = 'done', finished_at = now(), eta_seconds = 0,
                  done_count = coalesce(total_count, done_count),
                  outcome = %s::jsonb, updated_at = now()
            WHERE montage_id = %s AND stage = %s""",
        (json.dumps({name: value for name, value in outcome.items() if value is not None}), ref,
         stage),
    )


@quiet
def failed(conn, ref):
    """Ролик не собрался: этап, который шёл, так и помечен — упавшим.

    Какой именно, спрашивать не надо: идущий этап всегда один. Остальные остаются как есть —
    сделанное сделано, и упавший ролик показывает, до какого места он дошёл.
    """
    conn.execute(
        """UPDATE montage_stages
              SET state = 'failed', finished_at = now(), eta_seconds = null, updated_at = now()
            WHERE montage_id = %s AND state = 'running'""",
        (ref,),
    )


@quiet
def entry(conn, ref, stage, **fields):
    """Одна реплика в журнал: про кадр или про план в целом.

    Поля — колонки контракта (`COLUMNS`), все необязательные: `media_id`, `decision` (`take` /
    `skip`), `reason` кодом, `series_size` для «взял 1 из N», кусок исходника секундами,
    `thought` с частью истории, рамкой и метками у режиссёра-нейросети. Незаданные не пишутся.
    """
    fields = _fit(fields)
    unknown = set(fields) - set(COLUMNS)
    if unknown:
        raise ValueError(f"the journal has no column(s) {sorted(unknown)}")
    # Имена колонок — свои же, из `COLUMNS`: в запрос подставляется список констант этого файла, а
    # значения идут параметрами, как везде.
    columns = [name for name in COLUMNS if fields.get(name) is not None]
    named = "".join(f", {one}" for one in columns)
    conn.execute(
        f"""INSERT INTO montage_journal_entries (montage_id, stage{named})
            VALUES (%s, %s{', %s' * len(columns)})""",
        (ref, stage, *(fields[name] for name in columns)),
    )


def _fit(fields):
    """Те же поля, обрезанные по потолкам контракта.

    Длинное приходит от модели — описание кадра, название части, метки, — и база держит потолки
    `CHECK`-ами: без обрезки реплика не легла бы вовсе. Лучше хвост фразы, чем пустое место в
    рассказе.
    """
    fitted = dict(fields)
    for name, characters in (("thought", THOUGHT_CHARACTERS), ("reason", REASON_CHARACTERS),
                             ("story_part_title", TITLE_CHARACTERS)):
        if fitted.get(name) is not None:
            fitted[name] = str(fitted[name])[:characters]
    if fitted.get("labels") is not None:
        labels = [str(one).strip() for one in fitted["labels"]]
        labels = [one for one in labels if one][:LABELS_COUNT]
        while labels and len(",".join(labels)) > LABELS_CHARACTERS:
            labels.pop()
        fitted["labels"] = labels or None
    return fitted
