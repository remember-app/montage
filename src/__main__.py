import sys

from src.worker import run


def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    if command != "run":
        raise SystemExit(f"unknown command {command!r}")
    run()


if __name__ == "__main__":
    main()
