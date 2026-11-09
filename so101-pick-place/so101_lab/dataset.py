"""LeRobot v3 export and read-back validation in the policy environment."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
from .config import JOINTS, EVAL_SEEDS


def export(run, destination, repo_id="local/so101-pick-place"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    run, destination = Path(run), Path(destination)
    metadata = json.loads((run / "run.json").read_text())
    if metadata["mode"] != "collect":
        raise ValueError(
            "Training export accepts collection runs only, never evaluation runs."
        )
    episodes = []
    for path in sorted((run / "successes").glob("seed-*/result.json")):
        result = json.loads(path.read_text())
        if result["seed"] in EVAL_SEEDS or not result["success"] or result.get("error"):
            raise ValueError(f"Invalid training episode: {path}")
        episodes.append((path.parent, result))
    if not episodes:
        raise ValueError(
            "No successful demonstrations to export; diagnostics remain in failures/."
        )
    cfg = metadata["config"]
    width, height = cfg["cameras"]["resolution"]
    names = [f"{name}.pos" for name in JOINTS]
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": names},
        "action": {"dtype": "float32", "shape": (6,), "names": names},
        "next.done": {"dtype": "bool", "shape": (1,), "names": None},
    }
    for camera in ("top", "left_wrist"):
        features[f"observation.images.{camera}"] = {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        }
    # Use the checkpoint's calibration for trainable columns; SI archive remains alongside the run.
    scale = np.array(cfg["adapter"]["scale_per_radian"], dtype=np.float32)
    offset = np.array(cfg["adapter"]["offset"], dtype=np.float32)
    ds = LeRobotDataset.create(
        repo_id,
        fps=cfg["control_hz"],
        root=destination,
        robot_type="so101_follower",
        features=features,
        use_videos=False,
        image_writer_threads=2,
    )
    source_map = []
    try:
        for episode, result in episodes:
            with np.load(episode / "trajectory.npz", allow_pickle=False) as data:
                count = len(data["state"])
                if not count or not data["done"][-1]:
                    raise ValueError(f"Incomplete successful episode: {episode}")
                for i in range(count):
                    if abs(float(data["timestamp"][i]) - i / cfg["control_hz"]) > 1e-8:
                        raise ValueError(
                            f"Invalid source timestamp in {episode}, frame {i}"
                        )
                    # LeRobot 0.4.4 generates timestamp=i/fps; explicit timestamps
                    # are rejected by its frame validator despite add_frame's API.
                    frame = {
                        "observation.state": data["state"][i] * scale + offset,
                        "action": data["action"][i] * scale + offset,
                        "next.done": np.array([data["done"][i]], dtype=bool),
                        "task": result["task"],
                    }
                    for source, target in (("top", "top"), ("wrist", "left_wrist")):
                        with Image.open(episode / source / f"{i:06d}.png") as image:
                            frame[f"observation.images.{target}"] = np.array(
                                image.convert("RGB")
                            )
                    ds.add_frame(frame)
                ds.save_episode()
                source_map.append(
                    {
                        "episode_index": len(source_map),
                        "seed": result["seed"],
                        "frames": count,
                    }
                )
    finally:
        ds.stop_image_writer()
        ds.finalize()
    (destination / "course_provenance.json").write_text(
        json.dumps(
            {
                "source_run": metadata,
                "episodes": source_map,
                "calibration": cfg["adapter"],
                "units": "checkpoint calibrated joint units",
            },
            indent=2,
        )
        + "\n"
    )
    return check(destination, repo_id)


def check(root, repo_id="local/so101-pick-place"):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=Path(root), download_videos=False)
    source = json.loads((Path(root) / "course_provenance.json").read_text())
    expected_frames = sum(e["frames"] for e in source["episodes"])
    if ds.num_frames != expected_frames or ds.num_episodes != len(source["episodes"]):
        raise AssertionError(
            "LeRobot frame/episode counts do not match the source archive."
        )
    offset = 0
    for ep in source["episodes"]:
        for i in range(ep["frames"]):
            frame = ds[offset + i]
            assert int(frame["episode_index"]) == ep["episode_index"]
            assert int(frame["frame_index"]) == i
            assert abs(float(frame["timestamp"]) - i / ds.fps) < 1e-4
            assert bool(frame["next.done"]) == (i == ep["frames"] - 1)
            assert frame["task"]
            for key in ("observation.state", "action"):
                assert frame[key].shape == (6,) and frame[key].isfinite().all()
            for camera in ("top", "left_wrist"):
                rgb = frame[f"observation.images.{camera}"]
                assert rgb.ndim == 3 and rgb.shape[0] == 3 and rgb.isfinite().all()
                assert rgb.min() >= 0 and rgb.max() <= 1
        offset += ep["frames"]
    result = {
        "passed": True,
        "episodes": ds.num_episodes,
        "frames": ds.num_frames,
        "fps": ds.fps,
    }
    (Path(root) / "readback.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def videos(run):
    import av

    for result_path in sorted(Path(run).glob("*/seed-*/result.json")):
        result = json.loads(result_path.read_text())
        if not result["frames"]:
            continue
        folder = result_path.parent
        with av.open(str(folder / "diagnostic.mp4"), "w") as output:
            stream = output.add_stream("libx264", rate=result["fps"])
            with Image.open(folder / "top/000000.png") as image:
                stream.width, stream.height = image.width * 2, image.height
            stream.pix_fmt = "yuv420p"
            for i in range(result["frames"]):
                images = []
                for name in ("top", "wrist"):
                    with Image.open(folder / name / f"{i:06d}.png") as im:
                        images.append(np.array(im.convert("RGB")))
                frame = av.VideoFrame.from_ndarray(
                    np.concatenate(images, axis=1), format="rgb24"
                )
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
