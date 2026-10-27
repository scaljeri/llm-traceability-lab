"""The process-local execution coordinator must never serve a shared multi-worker DB."""

import subprocess
import sys

from traceability_lab.runtime import database_process_lock


def test_second_process_is_refused_and_lock_releases(tmp_path):
    database = tmp_path / "lab.sqlite3"
    script = 'from pathlib import Path; from traceability_lab.runtime import database_process_lock\nwith database_process_lock(Path(__import__("sys").argv[1])): pass'
    with database_process_lock(database):
        # Multiple test clients in this process can share the same application DB.
        with database_process_lock(database):
            blocked = subprocess.run(
                [sys.executable, "-c", script, str(database)], capture_output=True, text=True
            )
            assert blocked.returncode != 0
            assert "Start with one worker" in blocked.stderr
    allowed = subprocess.run([sys.executable, "-c", script, str(database)], capture_output=True, text=True)
    assert allowed.returncode == 0, allowed.stderr
