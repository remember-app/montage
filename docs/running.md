# Что нужно

- Python 3.11+
- Postgres 16
- `ffmpeg` и `ffprobe` в `PATH` — ими делаются превью
- Объектное хранилище: локально MinIO из `../api/docker-compose.yml`, в проде Cloudflare R2

# Postgres и хранилище

Канонная локальная база и бакет — те же, что у API, одной командой:

```sh
cd ../api && docker compose up -d
```

Если её ещё нет (или нужна отдельная), сгодится любой Postgres 16:

```sh
docker run --rm -d --name remember-pg \
  -e POSTGRES_USER=remember -e POSTGRES_PASSWORD=remember -e POSTGRES_DB=remember \
  -p 5432:5432 postgres:16
```

# Запуск

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env             # база и локальное хранилище
set -a && . ./.env && set +a     # внести в окружение: воркер читает только его

python -m montage init           # создать таблицы воркера
python -m montage tracks         # выложить музыку в хранилище (jobs/montage.md, «Треки»)
python -m montage run            # цикл разбора очереди, Ctrl-C для остановки
```

Своего чтения `.env` у воркера нет — переменные приходят из окружения, и на хостинге их задаёт
сам сервис. Имена хранилища — те же четыре, что читает API, так что вместо своего `.env` можно
внести его `../api/.env` целиком.

`init` можно звать сколько угодно раз: `schema.sql` идемпотентен (`CREATE TABLE IF NOT EXISTS`),
на чистой базе накатывается с нуля.

Необязательный `R2_REGION` (по умолчанию `auto`) — на случай хранилища, которому не всё равно,
чем подписан запрос.

`API_URL` и `INTERNAL_WORKER_TOKEN` нужны только готовому ролику — ими воркер стучится в API,
чтобы тот разослал пуш. Не заданы — сборка всё равно проходит, в логе остаётся предупреждение.

# Фиктивная задача

В другом терминале (с тем же окружением):

```sh
python -m montage enqueue dummy        # воркер её возьмёт и поставит done
python -m montage enqueue dummy fail   # ref=fail — упадёт и уйдёт в повтор
```

Второй вариант и есть проверка повторов: воркер отметит попытку, запишет ошибку и отложит задачу
(30 с, потом 60 с), а после третьей попытки поставит `failed`.

