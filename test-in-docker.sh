#!/bin/sh
# Тесты воркера в контейнере (Python и ffmpeg внутри) против Postgres и MinIO, уже поднятых
# на хосте: постоянного Postgres на 5432 и MinIO из `cd ../api && docker compose up -d --wait`.
# Аргументы уходят в pytest:
#
#     ./test-in-docker.sh                 # весь набор
#     ./test-in-docker.sh -k preview      # часть
#
# Переменные — те же значения, что в .env.example, только `localhost` заменён на хост из
# контейнера, а база — тестовая `remember_test`, та же, что у `make test-montage`: прогон чистит
# `jobs`, и стенду этого не надо. Рабочая копия подключена только на чтение: писать тестам некуда,
# кроме своих временных файлов.
set -e
cd "$(dirname "$0")"

docker build --target test -t montage-tests .

exec docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  -v "$PWD:/app:ro" \
  -e DATABASE_URL="postgres://postgres:postgres@host.docker.internal:5432/remember_test" \
  -e R2_ENDPOINT="http://host.docker.internal:9000" \
  -e R2_BUCKET="remember" \
  -e R2_ACCESS_KEY_ID="remember" \
  -e R2_SECRET_ACCESS_KEY="remember-secret" \
  montage-tests python -m pytest -p no:cacheprovider "$@"
