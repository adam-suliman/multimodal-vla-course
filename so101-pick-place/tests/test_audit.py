import json
import numpy as np
import pytest
from so101_lab.audit import audit
from so101_lab.config import PROTOCOL, load_config, fingerprint


def archive(tmp_path):
    cfg = load_config()
    episode = tmp_path / "successes/seed-10000"
    episode.mkdir(parents=True)
    meta = {
        "mode": "collect",
        "config": cfg,
        "config_sha256": fingerprint(cfg),
        "protocol": PROTOCOL,
        "seeds": [10000],
        "successes": 1,
        "complete": True,
        "gpu_memory": {"sampled_peak_mib": 4000},
    }
    (tmp_path / "run.json").write_text(json.dumps(meta))
    (episode / "result.json").write_text(
        json.dumps(
            {
                "seed": 10000,
                "success": True,
                "initial_position": [0.23, 0.1, 0.735],
                "max_lift_m": 0.03,
            }
        )
    )
    samples = []
    for tick in range(1, 122):
        samples.append(
            {
                "tick": tick,
                "success": tick == 121,
                "position": [0.23, 0.1, 0.765 if tick == 1 else 0.735],
                "rotation": np.eye(3).tolist(),
                "linear_velocity": [0, 0, 0],
                "angular_velocity": [0, 0, 0],
                "finger_contacts": [tick == 1] * 2,
                "gripper_open": tick != 1,
            }
        )
    path = episode / "physics.jsonl"
    path.write_text("\n".join(map(json.dumps, samples)) + "\n")
    return path, samples


def test_audit_recomputes_and_rejects_altered_height(tmp_path):
    path, samples = archive(tmp_path)
    assert audit(tmp_path, 12288)["success_rate"] == 1
    samples[0]["position"][2] = 0.75
    path.write_text("\n".join(map(json.dumps, samples)) + "\n")
    with pytest.raises(AssertionError):
        audit(tmp_path)


def test_audit_enforces_memory_evidence(tmp_path):
    archive(tmp_path)
    with pytest.raises(AssertionError, match="budget exceeded"):
        audit(tmp_path, 3000)
