#!/bin/sh
# Тесты воркера в контейнере (Python и ffmpeg внутри) против Postgres и MinIO, уже поднятых
# на хосте: постоянного Postgres на 5432 и MinIO из `cd ../api && docker compose up -d --wait`.
# Аргументы уходят в pytest:
#
#     ./test-in-docker.sh                 # весь набор
#     ./test-in-docker.sh -k preview      # часть
#
# Перед прогоном накатываются миграции api в ту же базу: тесты живые и берут настоящие таблицы
# (`media_items`, `frame_comments` и прочие), а без них падают не своей ошибкой, а
# `relation "..." does not exist` — виноватым в этом выглядит кто угодно, кроме ненакатанной базы.
# Не накатились — прогон останавливается до сборки образа, с указанием базы и команды.
#
# Переменные — те же значения, что в .env.example, только `localhost` заменён на хост из
# контейнера, а база — тестовая `remember_test`, та же, что у `make test-montage`: прогон чистит
# `jobs`, и стенду этого не надо. `TEST_DB_NAME` переопределяется тем, кому общая база мешает
# (она общая на всех прогоняющих, и чистят её тоже все). Рабочая копия подключена только на
# чтение: писать тестам некуда, кроме своих временных файлов.
set -e
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"

find_api() {
  d=$1
  while [ "$d" != / ]; do
    [ -d "$d/api/migrations" ] && { echo "$d/api"; return 0; }
    d=$(dirname "$d")
  done
  return 1
}

# Миграции живут в api — соседе по чекауту продукта. Рядом он не всегда: изолированный worktree
# воркера лежит на три уровня глубже, и `../api` от него уводит не туда, поэтому ищем вверх.
# `|| true` — иначе set -e убьёт скрипт молча, на самом неинтересном месте из всех.
API_DIR=${API_DIR:-$(find_api "$here" || true)}
TEST_DB_NAME=${TEST_DB_NAME:-remember_test}
HOST_DB_URL="postgres://postgres:postgres@localhost:5432/$TEST_DB_NAME"

# С хостом по localhost, из контейнера — по host.docker.internal: одна база, два адреса к ней.
die() {
  cat >&2 <<EOF

!! $1
   База тестов: $HOST_DB_URL — Postgres на 5432, постоянный контейнер.
   Накатывает её миграции api make test-montage из корня чекаута продукта: он же заводит базу,
   если её нет. Руками то же, чем пользуется скрипт:
     cd ${API_DIR:-<каталог api>} && DATABASE_URL=$HOST_DB_URL npm run migrate

EOF
  exit 1
}

[ -d "$API_DIR" ] || die "каталог api с миграциями не найден (${API_DIR:-нигде вверх от $here})."
echo "накатываю миграции api в $TEST_DB_NAME"
(cd "$API_DIR" && DATABASE_URL="$HOST_DB_URL" npm run migrate) ||
  die "миграции api в базу $TEST_DB_NAME не накатились (что именно — строкой выше)."

docker build --target test -t montage-tests .

exec docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  -v "$here:/app:ro" \
  -e DATABASE_URL="postgres://postgres:postgres@host.docker.internal:5432/$TEST_DB_NAME" \
  -e R2_ENDPOINT="http://host.docker.internal:9000" \
  -e R2_BUCKET="remember" \
  -e R2_ACCESS_KEY_ID="remember" \
  -e R2_SECRET_ACCESS_KEY="remember-secret" \
  montage-tests python -m pytest -p no:cacheprovider "$@"