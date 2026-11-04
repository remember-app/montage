import os


def env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def required(name: str) -> str:
    value = env(name)
    if value is None:
        raise RuntimeError(f"{name} is not set")
    return value


def positive(name: str, fallback: int) -> int:
    value = env(name)
    if value is None:
        return fallback
    number = int(value)
    if number <= 0:
        raise RuntimeError(f"{name} must be a positive integer")
    return number


def database_url() -> str:
    return env("POSTGRES_URL") or required("DATABASE_URL")


def s3_endpoint() -> str:
    return required("S3_ENDPOINT").rstrip("/")


def s3_bucket() -> str:
    return required("S3_BUCKET")


def s3_region() -> str:
    return env("S3_REGION") or "auto"


def s3_access_key_id() -> str:
    return required("S3_ACCESS_KEY_ID")


def s3_secret_access_key() -> str:
    return required("S3_SECRET_ACCESS_KEY")


def api_url() -> str | None:
    return env("API_URL")


def worker_token() -> str | None:
    return env("INTERNAL_WORKER_TOKEN")
