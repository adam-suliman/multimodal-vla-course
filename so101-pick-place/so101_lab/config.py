"""Configuration and the frozen evaluation protocol (SI units)."""

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
EVAL_SEEDS = (
    101,
    211,
    307,
    401,
    503,
    601,
    701,
    809,
    907,
    1009,
    1103,
    1201,
    1301,
    1409,
    1511,
    1601,
    1709,
    1801,
    1901,
    2003,
)
PROTOCOL = {
    "version": 1,
    "seeds": EVAL_SEEDS,
    "time_limit_s": 30.0,
    "lift_m": 0.02,
    "stable_s": 1.0,
    "linear_speed_m_s": 0.01,
    "angular_speed_rad_s": 0.1,
    "support_tolerance_m": 0.003,
}


def load_config(path=None):
    cfg = json.loads(Path(path or ROOT / "configs/default.json").read_text())
    if cfg["physics_hz"] % cfg["control_hz"] or cfg["control_hz"] != 10:
        raise ValueError("Control rate must be 10 Hz and divide the physics rate.")
    if cfg["cube"]["size"] != 0.03 or cfg["target"]["size"] != [0.1, 0.1]:
        raise ValueError("This protocol requires a 3 cm cube and a 10 x 10 cm target.")
    for lo, hi in zip(cfg["cube"]["xy_min"], cfg["cube"]["xy_max"]):
        if lo >= hi:
            raise ValueError("Cube sampling bounds must have positive area.")
    if cfg["physics_hz"] < 120 or cfg["robot"]["max_speed_rad_s"] <= 0:
        raise ValueError(
            "Use at least 120 Hz physics and a positive joint speed limit."
        )
    if len(cfg["robot"]["home_rad"]) != 6:
        raise ValueError("Robot home must contain six joint positions.")
    color = cfg["robot"].get("printed_color", [1.0, 1.0, 1.0])
    if len(color) != 3 or not all(0 <= value <= 1 for value in color):
        raise ValueError("Robot printed_color must contain three RGB values in [0, 1].")
    cfg["robot"]["printed_color"] = color
    return copy.deepcopy(cfg)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
