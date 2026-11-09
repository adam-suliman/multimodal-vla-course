import json
import numpy as np
import pytest
from so101_lab.config import load_config
from so101_lab.recording import EpisodeWriter


def make_run(root):
    cfg = load_config()
    cfg["cameras"]["resolution"] = [16, 16]
    (root / "successes").mkdir(parents=True)
    (root / "run.json").write_text(json.dumps({"mode": "collect", "config": cfg}))
    for seed, length in ((10000, 3), (10001, 5)):
        writer = EpisodeWriter(root / "successes", seed, cfg)
        for i in range(length):
            rgb = np.full((16, 16, 3), i * 20, dtype=np.uint8)
            obs = {
                "state": np.arange(6, dtype=np.float32) * 0.1,
                "images": {"top": rgb, "wrist": 255 - rgb},
            }
            writer.add(obs, np.ones(6, dtype=np.float32) * i / 20, i == length - 1)
        writer.finish({"seed": seed, "success": True})
    return cfg


def test_archive_pairs_observation_with_executed_action(tmp_path):
    make_run(tmp_path)
    with np.load(tmp_path / "successes/seed-10001/trajectory.npz") as t:
        assert t["state"].shape == (5, 6)
        assert t["done"].tolist() == [False] * 4 + [True]
        np.testing.assert_allclose(t["timestamp"], np.arange(5) / 10)
        np.testing.assert_allclose(t["action"][-1], 0.2)


def test_lerobot_real_roundtrip(tmp_path):
    pytest.importorskip("lerobot")
    from so101_lab.dataset import export

    cfg = make_run(tmp_path / "run")
    result = export(tmp_path / "run", tmp_path / "dataset")
    assert result == {"passed": True, "episodes": 2, "frames": 8, "fps": 10}
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("local/so101-pick-place", root=tmp_path / "dataset")
    np.testing.assert_allclose(
        ds[7]["action"].numpy(),
        0.2 * np.array(cfg["adapter"]["scale_per_radian"]) + cfg["adapter"]["offset"],
        atol=2e-6,
    )
    assert float(ds[0]["observation.images.top"].sum()) == 0
    assert float(ds[0]["observation.images.left_wrist"].mean()) == 1


def test_eval_export_rejected(tmp_path):
    from so101_lab import dataset

    pytest.importorskip("lerobot")
    make_run(tmp_path / "run")
    p = tmp_path / "run/run.json"
    meta = json.loads(p.read_text())
    meta["mode"] = "evaluate"
    p.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="evaluation"):
        dataset.export(tmp_path / "run", tmp_path / "dataset")


def test_diagnostic_videos_decode(tmp_path):
    av = pytest.importorskip("av")
    from so101_lab.dataset import videos

    make_run(tmp_path)
    videos(tmp_path)
    lengths = []
    for path in tmp_path.glob("successes/*/diagnostic.mp4"):
        with av.open(str(path)) as stream:
            lengths.append(len(list(stream.decode(video=0))))
    assert sorted(lengths) == [3, 5]
