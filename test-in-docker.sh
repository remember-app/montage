#!/bin/sh
# Тесты воркера в контейнере (Python и ffmpeg внутри) против Postgres и MinIO, уже поднятых
# на хосте: постоянного Postgres на 5432 и MinIO из `cd ../api && docker compose up -d --wait`.
# Аргументы уходят в pytest:
#
#     ./test-in-docker.sh                 # весь набор
#     ./test-in-docker.sh -k preview      # часть
#
# База и бакет — свои на прогон (`<база>_<метка>_test` и `<имя>-<метка>`): два прогона сразу —
# два ворктри, две сессии — не читают строки и объекты друг друга. Базу заводит и сносит этот
# скрипт, бакет — набор после себя (`tests/conftest.py`).
#
# Миграции api накатываются в ту же базу до сборки образа: тесты живые и берут настоящие таблицы
# (`media_items`, `frame_comments` и прочие), а без них падают не своей ошибкой, а
# `relation "..." does not exist` — виноватым в этом выглядит кто угодно, кроме ненакатанной базы.
# Не накатились — прогон останавливается до сборки образа, с указанием базы и команды.
#
# Остальные переменные — те же значения, что в .env.example, только `localhost` заменён на хост из
# контейнера. Откуда база и как зовётся метка — `tests/test-env.sh`. Рабочая копия подключена
# только на чтение: писать тестам некуда, кроме своих временных файлов.
set -e
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
. tests/test-env.sh

trap test_env_drop EXIT
test_env_create
test_env_migrate

docker build --target test -t montage-tests .

# Не exec: сносить базу прогона надо после того, как контейнер вышел, а exec съел бы trap.
docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  -v "$here:/app:ro" \
  -e DATABASE_URL="postgres://postgres:postgres@host.docker.internal:5432/$TEST_DB_NAME" \
  -e R2_ENDPOINT="http://host.docker.internal:9000" \
  -e R2_BUCKET="$R2_BUCKET" \
  -e R2_ACCESS_KEY_ID="remember" \
  -e R2_SECRET_ACCESS_KEY="remember-secret" \
  -e TEST_RUN_TAG="$TEST_RUN_TAG" \
  montage-tests python -m pytest -p no:cacheprovider "$@"
