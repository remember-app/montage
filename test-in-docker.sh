#!/bin/sh
# Тесты воркера в контейнере (Python и ffmpeg внутри) против Postgres и MinIO, уже поднятых
# на хосте: `cd ../api && docker compose up -d --wait`. Аргументы уходят в pytest:
#
#     ./test-in-docker.sh                 # весь набор
#     ./test-in-docker.sh -k preview      # часть
#
# Переменные — те же значения, что в .env.example, только `localhost` заменён на хост из
# контейнера. Рабочая копия подключена только на чтение: писать тестам некуда, кроме своих
# временных файлов.
set -e
cd "$(dirname "$0")"

docker build --target test -t montage-tests .

exec docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  -v "$PWD:/app:ro" \
  -e DATABASE_URL="postgres://remember:remember@host.docker.internal:5432/remember" \
  -e R2_ENDPOINT="http://host.docker.internal:9000" \
  -e R2_BUCKET="remember" \
  -e R2_ACCESS_KEY_ID="remember" \
  -e R2_SECRET_ACCESS_KEY="remember-secret" \
  montage-tests python -m pytest -p no:cacheprovider "$@"
