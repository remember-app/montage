"""Свой прогон — своя база и свой бакет, и это проверяется.

Два прогона сразу, например два ворктри одного репозитория, раньше садились на общую
`remember_test` и один бакет: очередь отдавала соседу чужую задачу, сосед сносил объекты под
ногами, и проверки мигали на одном и том же коммите — то красное, то зелёное.

`test.sh` и `test-in-docker.sh` кладут в имена базы и бакета метку прогона (`TEST_RUN_TAG`).
Отсюда и проверка: убери метку из имени — красное. Спрашиваем базу, а не переменную окружения:
отвечает соединение, которым прогон на самом деле работает.
"""

import os

from montage import storage, worker


def test_the_run_has_a_database_and_a_bucket_of_its_own():
    tag = os.environ["TEST_RUN_TAG"]
    with worker.connect() as connection:
        name = connection.execute("SELECT current_database()").fetchone()[0]
    assert name.endswith(f"_{tag}_test"), (
        f"{name} — не своя база: в имени нет метки прогона {tag}, сосед снова войдёт в неё"
    )
    bucket = storage.client().bucket
    assert bucket.endswith(f"-{tag}"), (
        f"{bucket} — не свой бакет: в имени нет метки прогона {tag}, сосед снова его снесёт"
    )
