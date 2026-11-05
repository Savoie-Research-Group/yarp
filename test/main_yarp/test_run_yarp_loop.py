"""
Tests for yarp/run_yarp_loop.py, the loop that runs a progress pass every few
minutes and stops when a pass reports the run quiescent.
"""
import sys
from unittest.mock import MagicMock

import pytest

from yarp import run_yarp_loop

MARKER = "No active or pending tasks remain; YARP has reached a quiescent state."


class _StopLoop(Exception):
    """Raised from the mocked sleep to end the loop after a set number of passes."""


def _run_loop(tmp_path, mocker, pass_outputs):
    """
    Run the loop with each progress pass mocked to append the next string of
    `pass_outputs` to yarp_loop.out. Returns how many passes ran before the loop
    stopped (on the marker, or when the passes run out).
    """
    outputs = iter(pass_outputs)
    passes = []

    def fake_run(cmd, stdout, stderr, check):
        passes.append(cmd)
        stdout.write(next(outputs))
        stdout.flush()
        return MagicMock(returncode=0)

    def fake_sleep(_):
        if len(passes) >= len(pass_outputs):
            raise _StopLoop

    mocker.patch.object(run_yarp_loop.subprocess, "run", side_effect=fake_run)
    mocker.patch.object(run_yarp_loop.time, "sleep", side_effect=fake_sleep)
    mocker.patch.object(sys, "argv", ["yarp-loop", "-w", str(tmp_path), "-i", "1", "-d", "60"])
    try:
        run_yarp_loop.main()
    except _StopLoop:
        pass
    return len(passes)


class TestStopMarkerIsReadFromThisLoopOnly:
    """
    The loop stops when a pass prints the quiescent marker. A run restarted with
    yarp-restart already has that marker in yarp_loop.out from when it finished
    before; only output written by the current loop may stop it.
    """

    def test_old_marker_from_a_previous_loop_does_not_stop_a_new_one(self, tmp_path, mocker):
        """Three passes with pending work after a restart: the loop keeps going through all three."""
        (tmp_path / "yarp_loop.out").write_text(f"earlier run\n{MARKER}\nShutdown marker found\n")

        assert _run_loop(tmp_path, mocker, ["pass 1\n", "pass 2\n", "pass 3\n"]) == 3

    def test_marker_from_this_loop_still_stops_it(self, tmp_path, mocker):
        """The second pass reports quiescence: the loop stops there, not after the third."""
        (tmp_path / "yarp_loop.out").write_text(f"earlier run\n{MARKER}\n")

        assert _run_loop(tmp_path, mocker, ["pass 1\n", f"pass 2\n{MARKER}\n", "pass 3\n"]) == 2
