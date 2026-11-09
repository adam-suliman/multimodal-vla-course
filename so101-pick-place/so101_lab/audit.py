"""Recompute each episode result from physics samples, without Isaac Sim."""

import json
from pathlib import Path
import numpy as np
from .config import EVAL_SEEDS, PROTOCOL, fingerprint
from .metrics import SuccessTracker


def audit(run, vram_budget_mib=None):
    run = Path(run)
    metadata = json.loads((run / "run.json").read_text())
    cfg = metadata["config"]
    if vram_budget_mib is not None:
        peak = metadata.get("gpu_memory", {}).get("sampled_peak_mib")
        if peak is None or not 0 < peak <= vram_budget_mib:
            raise AssertionError(
                f"Missing GPU memory measurement or budget exceeded: {peak} MiB"
            )
    assert fingerprint(metadata["protocol"]) == fingerprint(PROTOCOL), (
        "Protocol changed"
    )
    assert fingerprint(cfg) == metadata["config_sha256"], (
        "Configuration fingerprint changed"
    )
    if metadata["mode"] == "evaluate":
        assert metadata["seeds"] == list(EVAL_SEEDS), "Evaluation seeds changed"
    support = cfg["table"]["center"][2] + cfg["table"]["size"][2] / 2
    seen, successes = [], 0
    for path in sorted(run.glob("*/seed-*/result.json")):
        result = json.loads(path.read_text())
        if result.get("error"):
            raise AssertionError(f"Execution error in seed {result['seed']}")
        seen.append(result["seed"])
        tracker = SuccessTracker(
            result["initial_position"][2], tuple(cfg["target"]["center_xy"]), support
        )
        ticks = 0
        for line in (path.parent / "physics.jsonl").read_text().splitlines():
            sample = json.loads(line)
            tick, claimed = sample.pop("tick"), sample.pop("success")
            ticks += 1
            assert tick == ticks, "Missing physics sample"
            for key in ("position", "rotation", "linear_velocity", "angular_velocity"):
                sample[key] = np.array(sample[key])
            assert bool(tracker.update(**sample, dt=1 / cfg["physics_hz"])) == claimed
        assert 0 < ticks <= round(PROTOCOL["time_limit_s"] * cfg["physics_hz"])
        assert tracker.success == result["success"]
        assert abs(tracker.max_lift - result["max_lift_m"]) < 1e-8
        successes += int(tracker.success)
    assert sorted(seen) == sorted(metadata["seeds"]), "Missing or duplicate episodes"
    assert metadata["successes"] == successes
    assert metadata["complete"]
    return {
        "passed": True,
        "successes": successes,
        "episodes": len(seen),
        "success_rate": successes / len(seen),
    }
