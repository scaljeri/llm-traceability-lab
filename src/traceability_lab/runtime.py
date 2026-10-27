"""One backend process per database; the execution coordinator is process-local."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import Lock

_held = {}
_guard = Lock()


@contextmanager
def database_process_lock(database: Path):
    path = database.resolve()
    with _guard:
        if path not in _held:
            path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(str(path) + ".process-lock", timeout=0, check_same_thread=False)
            try:
                connection.execute("BEGIN EXCLUSIVE")
            except sqlite3.OperationalError as error:
                connection.close()
                raise RuntimeError(
                    "Another backend process is using this database. Start with one worker."
                ) from error
            _held[path] = [connection, 0]
        _held[path][1] += 1
    try:
        yield
    finally:
        with _guard:
            _held[path][1] -= 1
            if not _held[path][1]:
                _held.pop(path)[0].close()
