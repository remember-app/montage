import sys

from src.worker import run


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command == "run":
        run()
        return
    if command == "backfill":
        from src.media import backfill

        backfill.run()
        return
    raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":
    main()
