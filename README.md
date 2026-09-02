# montage

Фоновый обработчик Remember. Отдельный процесс на Python, наружу не смотрит: в цикле берёт задачи
из таблицы `jobs` в Postgres, выполняет и почти всё время спит. Единственный способ дать ему работу —
вставить строку в `jobs` (это делает API).

Сейчас здесь только скелет очереди: одна фиктивная задача `dummy`. Обработчики `media_preview`
(m-1) и `montage` (m-2) придут со своими фичами, вместе с `ffmpeg`, OpenCV и ONNX.

## Что нужно

- Python 3.11+
- Postgres 16

## Postgres

Канонная локальная база — та же, что у API:

```sh
cd ../api && docker compose up -d
```

Если её ещё нет (или нужна отдельная), сгодится любой Postgres 16:

```sh
docker run --rm -d --name remember-pg \
  -e POSTGRES_USER=remember -e POSTGRES_PASSWORD=remember -e POSTGRES_DB=remember \
  -p 5432:5432 postgres:16
```

## Запуск

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export DATABASE_URL='postgres://remember:remember@localhost:5432/remember'

python worker.py init            # создать таблицу jobs
python worker.py run             # цикл разбора очереди, Ctrl-C для остановки
```

`init` можно звать сколько угодно раз: `schema.sql` идемпотентен (`CREATE TABLE IF NOT EXISTS`),
на чистой базе накатывается с нуля.

## Фиктивная задача

В другом терминале (с тем же `DATABASE_URL`):

```sh
python worker.py enqueue dummy         # воркер её возьмёт и поставит done
python worker.py enqueue dummy fail    # ref=fail — упадёт и уйдёт в повтор
```

Второй вариант и есть проверка повторов: воркер отметит попытку, запишет ошибку и отложит задачу
(30 с, потом 60 с), а после третьей попытки поставит `failed`.

## Тесты

```sh
DATABASE_URL='postgres://remember:remember@localhost:5432/remember' python -m pytest
```

Тесты живые: им нужен настоящий Postgres — тот же локальный, что выше. **Каждый прогон чистит
таблицу `jobs`** в базе из `DATABASE_URL`, так что не направляйте их на что-то ценное.

## Как устроена очередь

Одна таблица `jobs` (`schema.sql`) на все типы задач:

| столбец | что в нём |
|---|---|
| `type` | какой обработчик звать |
| `ref` | id объекта, о котором задача |
| `status` | `queued` → `running` → `done` / `failed` |
| `attempts` | сколько попыток уже было |
| `run_after` | раньше этого момента задача не берётся |
| `last_error` | текст последней ошибки |

Цикл (`worker.py`): одним запросом берёт одну задачу со `status='queued' AND run_after <= now()`
через `FOR UPDATE SKIP LOCKED` и сразу метит `running` — второй воркер такую строку пропустит.
Успех → `done`. Исключение → `attempts + 1`, `last_error`, и либо `queued` с
`run_after = now() + 30с × 2^(попытка−1)`, либо `failed` после третьей попытки. Пустая очередь →
сон 5 секунд.

Без Redis и брокеров: один воркер, одна таблица. Это осознанный потолок — см. «Очередь задач» в
`architecture.md` корневого репозитория.
