"""Verify the numeric adapter against every frame of the pinned public dataset."""

import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request
import numpy as np
import pyarrow.parquet as pq
from so101_lab.config import ROOT, load_config
from so101_lab.assets import fetch, robot_urdf
from so101_lab.kinematics import Kinematics


def main():
    source = json.loads((ROOT / "sources.lock.json").read_text())["training_data"]
    url = f"https://huggingface.co/datasets/{source['repository']}/resolve/{source['revision']}/data/chunk-000/file-000.parquet"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "data.parquet"
        with urllib.request.urlopen(url, timeout=120) as response:
            path.write_bytes(response.read())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != source["sample_sha256"]:
            raise RuntimeError("Training data checksum mismatch.")
        table = pq.read_table(path)
    cfg = load_config()
    result = {
        "rows": len(table),
        "source_sha256": digest,
        "revision": source["revision"],
        "max_absolute_errors": {},
    }
    for key in ("observation.state", "action"):
        expected = np.array(table[key].to_pylist())
        radians = np.array(table[key + ".radian_urdf0"].to_pylist())
        actual = (
            radians * np.array(cfg["adapter"]["scale_per_radian"])
            + cfg["adapter"]["offset"]
        )
        error = abs(actual - expected).max(axis=0)
        if error.max() >= 1e-5:
            raise AssertionError(f"Calibration mismatch for {key}: {error}")
        result["max_absolute_errors"][key] = error.tolist()
    with tempfile.TemporaryDirectory() as tmp:
        fetch(tmp, ("robot",))
        cfg["robot"]["base_position"] = [0, 0, 0]
        cfg["robot"]["tool_offset"] = [0, 0, 0]
        ik = Kinematics(robot_urdf(tmp), cfg)
        q = np.array(table["observation.state.radian_urdf0"].to_pylist())
        xyz = np.array(table["observation.ee_pos.robot_xyzrpy"].to_pylist())[:, :3]
        fk = np.array([ik.tool(row)[:3, 3] for row in q])
        error = float(np.linalg.norm(fk - xyz, axis=1).max())
        if error > 1e-6:
            raise AssertionError(f"URDF frame convention mismatch: {error} m")
        result["max_fk_error_m"] = error
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
