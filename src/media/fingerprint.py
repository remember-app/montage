"""Отпечаток содержимого файла: тот же SHA-256 в шестнадцатеричном виде, что считает телефон."""

import hashlib
import pathlib

CHUNK = 1024 * 1024


def sha256_of(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()
