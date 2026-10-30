"""
A lock file that keeps a YARP progress pass and yarp-restart from changing the
same run directory at the same time.

A progress pass loads STATUS.json and the pickles at its start and writes them
back at its end, so anything another process changes in between is silently
overwritten. Both take this lock first. It is a plain file created with
O_EXCL, so it works across login nodes sharing the run directory; it records
who holds it so a lock left behind by a killed process can be inspected and
removed by hand.
"""
import json
import os
import socket
import time
from contextlib import contextmanager
from pathlib import Path

LOCK_NAME = ".yarp_run.lock"


class RunLockedError(RuntimeError):
    """The run directory is locked by another YARP process."""


def lock_path(work_dir) -> Path:
    return Path(work_dir) / LOCK_NAME


@contextmanager
def run_lock(work_dir, owner: str):
    """
    Hold the run directory's lock for the duration of the block.

    Raises RunLockedError, naming the current holder, if the lock is taken.
    The lock is removed when the block ends, including on an exception; only a
    process killed outright (or a crashed node) leaves it behind.
    """
    path = lock_path(work_dir)
    info = {"owner": owner, "host": socket.gethostname(), "pid": os.getpid(),
            "since": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        try:
            holder = path.read_text().strip()
        except OSError:
            holder = "unknown"
        raise RunLockedError(
            f"{path} is held by another YARP process: {holder}. If that process is no longer "
            f"running (check the host and pid), delete the lock file and try again."
        )
    with os.fdopen(fd, "w") as f:
        json.dump(info, f)
    try:
        yield
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
