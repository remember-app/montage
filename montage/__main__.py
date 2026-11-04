import contextlib
import logging
import sys

from montage import worker
from montage.video import library

log = logging.getLogger("montage")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    command, *args = sys.argv[1:] or ["run"]
    if command == "tracks":
        if args:
            sys.exit(__doc__)
        try:
            published = library.publish_tracks()
        except KeyError as unset:
            sys.exit(f"{unset.args[0]} is not set — see README.md")
        except Exception as exc:
            sys.exit(f"the music library was not published: {exc}")
        print(
            f"the music library is in the bucket: {len(published)} of {len(library.TRACKS)} "
            f"uploaded, the rest were already there"
        )
        sys.exit()
    with worker.connect() as conn:
        if command == "init" and not args:
            worker.init_schema(conn)
            print("the worker's tables are ready")
        elif command == "enqueue" and 1 <= len(args) <= 2:
            print("queued job", worker.enqueue(conn, *args))
        elif command == "run" and not args:
            with contextlib.suppress(KeyboardInterrupt):
                worker.run(conn)
            log.info("stopped")
        else:
            sys.exit(__doc__)


if __name__ == "__main__":
    main()
