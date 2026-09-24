"""Команды воркера, которые запускают руками:

    python -m montage init                  # create the worker's own tables (idempotent)
    python -m montage tracks                # the music library into the bucket (idempotent)
    python -m montage enqueue <type> [ref]  # put one job in the queue
    python -m montage run                   # the loop, Ctrl-C to stop

Needs DATABASE_URL — except `tracks`, which needs the four storage variables. See README.md.
"""

import logging
import sys

from montage import worker
from montage.video import library

log = logging.getLogger("montage")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    command, *args = sys.argv[1:] or ["run"]
    # Outside the connection below on purpose: the library is files and a bucket, and there is no
    # reason for the one command with no database in it to refuse to run without DATABASE_URL.
    if command == "tracks":
        if args:
            sys.exit(__doc__)  # here too, or a typo would ask for a database to say «usage»
        try:
            published = library.publish_tracks()
        except KeyError as unset:
            # A storage variable missed on the deploy — `storage.client()` reads the four of them
            # straight out of the environment. One line saying which, the same answer `connect`
            # gives for DATABASE_URL, rather than a `KeyError` out of the middle of boto3.
            sys.exit(f"{unset.args[0]} is not set — see README.md")
        except Exception as exc:
            # Storage that cannot be reached, the wrong bucket, a key that may not write: what it
            # said, in one line rather than in a traceback. Whatever landed before the failure
            # stays there and the command is idempotent, so the answer to any of these is to fix
            # the cause and run it again.
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
            try:
                worker.run(conn)
            except KeyboardInterrupt:
                log.info("stopped")
        else:
            sys.exit(__doc__)


if __name__ == "__main__":
    main()
