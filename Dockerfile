# Две стадии на одной основе. Последняя, без имени цели, — боевой образ воркера: его собирает CI.
# `test` — среда для test-in-docker.sh. Общая `base` значит, что тесты гоняются на тех же ffmpeg,
# libheif и librosa, что работают в кластере.
FROM python:3.12-slim-trixie AS base

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# pytest и ruff стоят в общем списке, в боевой образ они не едут.
RUN grep -vE '^(pytest|ruff)\b' requirements.txt > /tmp/runtime.txt \
    && pip install --no-cache-dir -r /tmp/runtime.txt \
    && rm /tmp/runtime.txt

# Код сюда не копируется — test-in-docker.sh подключает рабочую копию томом, так что образ
# пересобирается только при смене requirements.txt.
FROM base AS test
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "-m", "pytest"]

FROM base AS runtime
COPY schema.sql .
COPY montage montage
RUN useradd --system --uid 10001 --user-group --no-create-home montage
# librosa компилирует разбор ритма через numba и кладёт кэш рядом со своими файлами, а они root'а.
# Без каталога на запись разбор падает, и ролик молча режется ровными кусками мимо битов.
ENV NUMBA_CACHE_DIR=/tmp/numba
# Числом, а не именем: иначе runAsNonRoot в кластере не сможет проверить, что это не root.
USER 10001:10001
# Python под PID 1 не отвечает на SIGTERM, и под ждал бы SIGKILL; SIGINT для `run` — это Ctrl-C.
STOPSIGNAL SIGINT
CMD ["python", "-m", "montage", "run"]
