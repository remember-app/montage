# Среда для прогона тестов воркера: Python и заведомо рабочий ffmpeg внутри, что бы ни было
# установлено на хосте. Сам код сюда не копируется — `test-in-docker.sh` подключает рабочую
# копию томом, так что образ пересобирается только при смене requirements.txt.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-m", "pytest"]
