import json
import logging
import urllib.request
import uuid

from src import config

log = logging.getLogger("movie.notify")

TIMEOUT_SECONDS = 10


def _post(montage_id: uuid.UUID, action: str, body: dict) -> None:
    url = config.api_url()
    token = config.worker_token()
    if not url or not token:
        log.warning("API_URL/INTERNAL_WORKER_TOKEN unset: montage %s is not announced", montage_id)
        return
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/internal/montages/{montage_id}/{action}",
        data=json.dumps(body).encode(),
        method="POST",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            response.read()
    except Exception as exc:
        log.warning("could not announce montage %s as %s: %s", montage_id, action, exc)


def ready(montage_id: uuid.UUID) -> None:
    _post(montage_id, "ready", {})


def failed(montage_id: uuid.UUID, error: str) -> None:
    _post(montage_id, "failed", {"error": error[:2000]})
