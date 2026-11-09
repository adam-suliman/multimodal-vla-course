"""Lossless episode archive, independent of the machine-learning environment."""

import json
from pathlib import Path
import numpy as np
from PIL import Image


class EpisodeWriter:
    def __init__(self, directory, seed, cfg):
        self.path = Path(directory) / f"seed-{seed}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.rows = []
        self.cfg, self.seed = cfg, seed
        self.trace = (self.path / "physics.jsonl").open("w")
        for camera in ("top", "wrist"):
            (self.path / camera).mkdir()

    def add(self, observation, executed_action, done, samples=()):
        index = len(self.rows)
        for name, image in observation["images"].items():
            Image.fromarray(image).save(self.path / name / f"{index:06d}.png")
        self.rows.append(
            (observation["state"].copy(), executed_action.copy(), bool(done))
        )
        for sample in samples:
            self.trace.write(json.dumps(sample) + "\n")
        self.trace.flush()

    def finish(self, result):
        self.trace.close()
        np.savez_compressed(
            self.path / "trajectory.npz",
            state=np.array([r[0] for r in self.rows], dtype=np.float32).reshape(-1, 6),
            action=np.array([r[1] for r in self.rows], dtype=np.float32).reshape(-1, 6),
            timestamp=np.arange(len(self.rows), dtype=np.float64)
            / self.cfg["control_hz"],
            done=np.array([r[2] for r in self.rows], dtype=bool),
        )
        record = {
            **result,
            "task": self.cfg["task"],
            "fps": self.cfg["control_hz"],
            "frames": len(self.rows),
            "state_units": "radians",
            "action_units": "absolute joint targets, radians",
            "observation_timing": "before action; done refers to the resulting transition",
        }
        (self.path / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        return self.path
