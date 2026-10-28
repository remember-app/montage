#!/bin/sh
# Проверка `test-in-docker.sh`: прогон обязан завести себе базу с меткой, накатить в неё миграции
# api и снести после, а если накатывать нечем — сказать словами, что базе нужно и какой командой
# это лечится. Раньше он обещал и не держал: тест падал на `relation "frame_comments" does not
# exist`, и виноватым выглядел кто угодно, кроме ненакатанной базы. Без этой проверки скрипт
# можно сломать обратно, и первым это заметит тот, кто запустит тесты следующим, и не ты.
#
#     tests/test-in-docker-check.sh
#
# Живой Postgres стенда и docker, как у самого скрипта. База проверки — своя, с меткой; общую
# `remember_test` и бакет `remember` она не трогает. Скрипт проверяется как есть: ни одна его
# настройка не подменяется, кроме основы имени базы (своя) и каталога api (во втором случае его
# нет намеренно). База остаётся после прогона только чтобы её можно было осмотреть (TEST_KEEP_DB=1)
# и сносится здесь же.
set -e
here=$(cd "$(dirname "$0")/.." && pwd)
BASE=montage_in_docker_check
TAG=check$$
DB="${BASE}_${TAG}_test"
err=$(mktemp)
pg=$(docker ps --format '{{.Names}} {{.Ports}}' | awk '$2 ~ /(^|:)5432->/ {print $1; exit}')

[ -n "$pg" ] || { echo "нет контейнера Postgres на 5432 — стенд не поднят"; exit 1; }
# WITH (FORCE): прогон мог оставить соединение, и DROP без неё спотыкается о чужой кэш.
drop_db() { docker exec "$pg" psql -U postgres -d postgres -qc "DROP DATABASE IF EXISTS $DB WITH (FORCE)" >/dev/null; }
trap 'drop_db; rm -f "$err"' EXIT

# Базы прогона ещё нет — и скрипт обязан завести её сам, а не ждать, что её заведёт кто-то другой.
# Довести её до свежей версии миграций и есть его обещание.
echo "== базы прогона нет: скрипт обязан завести $DB и накатить в неё миграции api"
drop_db
TEST_DATABASE_URL="postgres://postgres:postgres@localhost:5432/$BASE" \
TEST_RUN_TAG="$TAG" TEST_KEEP_DB=1 \
  "$here/test-in-docker.sh" --collect-only -q >/dev/null
got=$(docker exec "$pg" psql -U postgres -d "$DB" -tAc 'SELECT max(version) FROM schema_migrations' 2>/dev/null || true)
# Свежая версия — из файлов api, а не из базы: иначе проверка повторяла бы то же, что проверяет.
d=$here
while [ "$d" != / ] && [ ! -d "$d/api/migrations" ]; do d=$(dirname "$d"); done
[ "$d" != / ] || { echo "!! не нашёл каталог api с миграциями — проверять нечем"; exit 1; }
freshest=$(ls "$d/api/migrations"/[0-9]*.do.*.sql | sed 's|.*/||; s|\.do\..*||' | sort -n | tail -1)
[ "$got" = "$freshest" ] ||
  { echo "!! в базе $DB применена версия ${got:-пусто — таблицы schema_migrations нет}, а свежая в api — $freshest"; exit 1; }
echo "ok: $DB доехала до $freshest"

# Накатывать нечем — не молча и не прогоном тестов, а остановкой с адресом.
echo "== api на месте нет: скрипт обязан остановиться и сказать, что делать"
if TEST_DATABASE_URL="postgres://postgres:postgres@localhost:5432/$BASE" \
  TEST_RUN_TAG="$TAG" API_DIR=/nonexistent \
  "$here/test-in-docker.sh" >/dev/null 2>"$err"; then
  echo "!! скрипт пошёл дальше, не заметив, что накатывать нечем"
  exit 1
fi
for about in "$DB" 'make test-montage'; do
  grep -q "$about" "$err" || { echo "!! в сообщении нет: $about"; cat "$err"; exit 1; }
done
echo "ok: сказано"

echo "== обе проверки прошли"
