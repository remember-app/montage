FROM python:3.12-slim-trixie

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN grep -vE '^ruff\b' requirements.txt > /tmp/runtime.txt \
    && pip install --no-cache-dir -r /tmp/runtime.txt \
    && rm /tmp/runtime.txt

COPY src src
RUN useradd --system --uid 10001 --user-group --no-create-home worker
ENV NUMBA_CACHE_DIR=/tmp/numba
USER 10001:10001
CMD ["python", "-m", "src", "run"]
