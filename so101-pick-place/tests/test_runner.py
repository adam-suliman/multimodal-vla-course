import argparse
import json
import numpy as np
import pytest
from so101_lab import runner
from so101_lab.config import load_config, EVAL_SEEDS


class FakeEnvironment:
    ik = None
    initial_position = np.array([0.2, 0, 0.735])
    last_action = np.zeros(6, np.float32)
    last_samples = []

    def reset(self, seed):
        self.seed = seed
        return {
            "state": np.zeros(6, np.float32),
            "images": {
                "top": np.zeros((4, 4, 3), np.uint8),
                "wrist": np.zeros((4, 4, 3), np.uint8),
            },
        }

    def step(self, action):
        return self.reset(self.seed), True

    def result(self):
        return {"seed": self.seed, "success": False}


class FakePolicy:
    def __init__(self, *args, **kwargs):
        self.metadata = {}

    def reset(self, *args):
        pass

    def action(self, obs):
        return np.zeros(6, np.float32)

    def close(self):
        pass


def args(tmp_path):
    return argparse.Namespace(
        output=tmp_path / "run",
        command="evaluate",
        policy="smolvla",
        gpu=0,
        data=tmp_path / "data",
        policy_python="unused",
        policy_device="cpu",
    )


def setup(monkeypatch):
    from so101_lab import policy

    monkeypatch.setattr(policy, "PolicyClient", FakePolicy)
    monkeypatch.setattr(
        runner,
        "GPUMonitor",
        type("Monitor", (), {"__init__": lambda *a: None, "close": lambda s: None}),
    )


def test_zero_sr_keeps_all_twenty_seeds(tmp_path, monkeypatch):
    setup(monkeypatch)
    runner.run(FakeEnvironment(), load_config(), args(tmp_path), {})
    result = json.loads((tmp_path / "run/run.json").read_text())
    assert result["complete"] and result["success_rate"] == 0
    assert result["denominator"] == 20
    assert [r["seed"] for r in result["episodes"]] == list(EVAL_SEEDS)
    assert len(list((tmp_path / "run/failures").glob("seed-*"))) == 20


def test_execution_error_does_not_become_valid_zero_sr(tmp_path, monkeypatch):
    setup(monkeypatch)
    env = FakeEnvironment()

    def broken(action):
        raise RuntimeError("Simulator unavailable")

    env.step = broken
    with pytest.raises(RuntimeError, match="execution errors"):
        runner.run(env, load_config(), args(tmp_path), {})
    result = json.loads((tmp_path / "run/run.json").read_text())
    assert not result["complete"] and result["success_rate"] is None
    assert len(result["episodes"]) == 20 and all(
        r.get("error") for r in result["episodes"]
    )
