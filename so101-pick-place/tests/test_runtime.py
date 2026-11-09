"""Kit fast shutdown must never turn an execution error into a successful CLI."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from so101_lab.runtime import run_simulator


@pytest.mark.parametrize(
    "error,exit_code,write_status",
    [
        (None, 0, True),
        ("Camera unavailable", 0, True),
        (None, 0, False),
        (None, 139, True),
    ],
)
def test_simulator_completion(monkeypatch, error, exit_code, write_status):
    def child(command, env):
        if write_status:
            Path(env["SO101_SIM_CHILD_STATUS"]).write_text(json.dumps({"error": error}))
        return SimpleNamespace(returncode=exit_code)

    monkeypatch.setattr("so101_lab.runtime.subprocess.run", child)
    args = SimpleNamespace(command="check-scene", video=False)
    if error is None and exit_code == 0 and write_status:
        assert run_simulator(args, ["check-scene"]) == 0
    else:
        with pytest.raises(RuntimeError):
            run_simulator(args, ["check-scene"])
