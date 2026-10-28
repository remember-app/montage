# Окружение одного прогона: своя база и свой бакет. Подключается из test.sh и test-in-docker.sh —
# разница между ними только в том, чем запускается pytest.
#
# Имя базы — как у api: `<база>_<метка>_test`, метка — четыре случайных байта. Бакет —
# `<имя>-<метка>`. Метка уезжает в TEST_RUN_TAG, и `tests/test_environment.py` проверяет, что оба
# имени её несут: без неё два прогона сразу — два ворктри, две сессии — снова садятся на одну базу
# и один бакет, очередь отдаёт соседу чужую задачу, а сосед сносит объекты под ногами.
#
# База берётся из TEST_DATABASE_URL, если он задан, иначе из DATABASE_URL; `_test` с конца
# снимается, чтобы `remember` и `remember_test` звали одну и ту же базу прогона. Переопределить
# метку — TEST_RUN_TAG; оставить базу после прогона (это делает tests/test-in-docker-check.sh) —
# TEST_KEEP_DB=1.

TEST_RUN_TAG=${TEST_RUN_TAG:-$(od -An -N4 -tx1 /dev/urandom | tr -d ' \n' | tr 'A-F' 'a-f')}
base=${TEST_DATABASE_URL:-${DATABASE_URL:-postgres://postgres:postgres@localhost:5432/remember_test}}
base=${base%_test}
TEST_DB_URL="${base}_${TEST_RUN_TAG}_test"
TEST_DB_NAME=${TEST_DB_URL##*/}
SERVER_DB_URL=${TEST_DB_URL%/*}/postgres
R2_BUCKET="${R2_BUCKET:-remember}-${TEST_RUN_TAG}"
export TEST_RUN_TAG TEST_DB_URL TEST_DB_NAME R2_BUCKET

# SQL в служебную базу `postgres`: psql с машины, а если его нет — psql в контейнере Postgres, на
# котором стоит стенд (его ищут по порту, как tests/test-in-docker-check.sh).
test_env_sql() {
  if command -v psql >/dev/null 2>&1; then
    psql "$SERVER_DB_URL" -v ON_ERROR_STOP=1 -qtAc "$1"
  else
    pg=$(docker ps --format '{{.Names}} {{.Ports}}' | awk '$2 ~ /(^|:)5432->/ {print $1; exit}')
    [ -n "$pg" ] || {
      echo "!! ни psql, ни контейнера Postgres на 5432 — базу прогона не завести" >&2
      exit 1
    }
    docker exec "$pg" psql -U postgres -d postgres -v ON_ERROR_STOP=1 -qtAc "$1"
  fi
}

test_env_create() {
  [ -z "$(test_env_sql "SELECT 1 FROM pg_database WHERE datname = '$TEST_DB_NAME'")" ] || return 0
  test_env_sql "CREATE DATABASE \"$TEST_DB_NAME\"" >/dev/null
}

test_env_drop() {
  [ -z "${TEST_KEEP_DB:-}" ] || return 0
  test_env_sql "DROP DATABASE IF EXISTS \"$TEST_DB_NAME\" WITH (FORCE)" >/dev/null || true
}

find_api() {
  d=$1
  while [ "$d" != / ]; do
    [ -d "$d/api/migrations" ] && { echo "$d/api"; return 0; }
    d=$(dirname "$d")
  done
  return 1
}

die() {
  cat >&2 <<EOF

!! $1
   База прогона: $TEST_DB_URL — Postgres на 5432, постоянный контейнер.
   Заводит её и накатывает в неё миграции api сам прогон: make test-montage из корня чекаута
   продукта или ./test.sh / ./test-in-docker.sh отсюда. Руками то же, чем пользуется прогон:
     cd ${API_DIR:-<каталог api>} && DATABASE_URL=$TEST_DB_URL npm run migrate

EOF
  exit 1
}

# Миграции живут в api — соседе по чекауту продукта. Рядом он не всегда: изолированный worktree
# воркера лежит на три уровня глубже, и `../api` от него уводит не туда, поэтому ищем вверх.
# `|| true` — иначе set -e убьёт прогон молча, на самом неинтересном месте из всех.
test_env_migrate() {
  API_DIR=${API_DIR:-$(find_api "$here" || true)}
  [ -d "$API_DIR" ] || die "каталог api с миграциями не найден (${API_DIR:-нигде вверх от $here})."
  (cd "$API_DIR" && DATABASE_URL="$TEST_DB_URL" npm run migrate) ||
    die "миграции api в базу $TEST_DB_NAME не накатились (что именно — строкой выше)."
}
