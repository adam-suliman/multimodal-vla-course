import json
import pytest
from so101_lab import assets
from so101_lab.config import EVAL_SEEDS


def test_frozen_seeds_are_unique_and_disjoint():
    assert len(EVAL_SEEDS) == len(set(EVAL_SEEDS)) == 20
    assert max(EVAL_SEEDS) < 10000


def test_offline_download_detects_corruption(tmp_path, monkeypatch):
    origin = tmp_path / "origin"
    origin.write_bytes(b"public asset")
    entry = {
        "path": "mesh.stl",
        "url": origin.as_uri(),
        "sha256": assets.sha256(origin),
        "size": origin.stat().st_size,
    }
    (tmp_path / "sources.lock.json").write_text(
        json.dumps({"robot": {"files": [entry]}})
    )
    monkeypatch.setattr(assets, "ROOT", tmp_path)
    assets.fetch(tmp_path / "data", ("robot",))
    cached = tmp_path / "data/robot/mesh.stl"
    assert cached.read_bytes() == origin.read_bytes()
    cached.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="corrupt"):
        assets.fetch(tmp_path / "data", ("robot",), offline=True)
    assets.fetch(tmp_path / "data", ("robot",))
    assert cached.read_bytes() == origin.read_bytes()
