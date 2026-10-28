#!/bin/sh
# Тесты воркера нативно — на своей базе и со своим бакетом, стенд не трогают. Аргументы уходят в
# pytest:
#
#     ./test.sh                 # весь набор
#     ./test.sh -k preview      # часть
#
# База заводится под прогон (имя с меткой перед `_test`) и сносится после; бакет — так же, его
# убирает `tests/conftest.py`. Откуда база и как зовётся метка — `tests/test-env.sh`.
# `make test-montage` из корня чекаута продукта зовёт этот же скрипт.
set -e
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
. tests/test-env.sh

PYTHON=${PYTHON:-.venv/bin/python}
[ -x "$PYTHON" ] || {
  echo "!! нет $PYTHON — поставь зависимости: make install" >&2
  exit 1
}

trap test_env_drop EXIT
test_env_create
test_env_migrate
DATABASE_URL="$TEST_DB_URL" "$PYTHON" -m pytest "$@"
