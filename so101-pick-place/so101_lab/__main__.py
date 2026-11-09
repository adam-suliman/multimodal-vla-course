"""One CLI shared by the README, notebook, simulator, and data tools."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from .config import ROOT, EVAL_SEEDS, load_config


def parser():
    p = argparse.ArgumentParser(description="SO-101 pick-and-place course laboratory")
    subs = p.add_subparsers(dest="command", required=True)
    for command in (
        "doctor",
        "fetch",
        "convert",
        "scene",
        "check-scene",
        "check-kinematics",
        "collect",
        "evaluate",
        "policy-smoke",
    ):
        s = subs.add_parser(command)
        s.add_argument("--config", type=Path, default=ROOT / "configs/default.json")
        s.add_argument("--data", type=Path, default=ROOT / "data")
        s.add_argument(
            "--gpu",
            type=int,
            default=0,
            help="GPU index visible inside this process/container",
        )
        s.add_argument(
            "--gui",
            action="store_true",
            help="Open an Isaac Sim window (headless is the default)",
        )
        if command == "fetch":
            s.add_argument("--only", choices=("robot", "policy", "all"), default="all")
            s.add_argument("--offline", action="store_true")
        if command in ("scene", "check-scene", "collect", "evaluate"):
            s.add_argument(
                "--output",
                type=Path,
                required=True,
                help="New output directory; existing results are never overwritten",
            )
            s.add_argument("--video", action="store_true")
            s.add_argument("--policy-python", default="/opt/policy/bin/python")
        if command in ("collect", "evaluate"):
            s.add_argument("--policy", choices=("expert", "smolvla"), default="expert")
            s.add_argument(
                "--policy-device",
                default=None,
                help="Defaults to the simulator GPU; use cpu for diagnostics",
            )
        if command in ("scene", "collect"):
            s.add_argument(
                "--seed", type=int, default=10000 if command == "collect" else 0
            )
            s.add_argument(
                "--episodes", type=int, default=10 if command == "collect" else 1
            )
        if command == "scene":
            s.add_argument("--seconds", type=float, default=5)
        if command == "policy-smoke":
            s.add_argument("--device", default="cuda:0")
    for command in ("export", "check-dataset", "video", "audit"):
        s = subs.add_parser(command)
        s.add_argument("--run", type=Path, required=command != "check-dataset")
        if command == "audit":
            s.add_argument("--vram-budget-mib", type=float, default=None)
        if command in ("export", "check-dataset"):
            s.add_argument("--dataset", type=Path, required=True)
            s.add_argument("--repo-id", default="local/so101-pick-place")
    subs.add_parser("test")
    return p


def doctor(args):
    report = {
        "python": sys.version,
        "packages": {},
        "gpu": None,
        "errors": [],
        "warnings": [],
    }
    for name in ("isaacsim", "numpy", "scipy", "torch", "lerobot", "transformers"):
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        report["gpu"] = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(args.gpu),
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        report["errors"].append(str(exc))
    if report["gpu"]:
        import csv

        driver = list(csv.reader(report["gpu"].splitlines()))[1][1].strip()
        minimum = json.loads((ROOT / "sources.lock.json").read_text())[
            "base_requirements"
        ]["minimum_linux_driver"]
        report["minimum_image_driver"] = minimum
        if tuple(map(int, driver.split("."))) < tuple(map(int, minimum.split("."))):
            report["warnings"].append(
                f"Driver {driver} predates the image reference {minimum}. "
                "The separate wheel runtime must pass check-scene on this driver."
            )
    report["asset_lock"] = "sources.lock.json"
    report["next_check"] = (
        "Run check-scene to verify rendering, URDF conversion and deterministic resets."
    )
    print(json.dumps(report, indent=2))
    return bool(report["errors"])


def check_kinematics(args, cfg):
    import numpy as np
    from .kinematics import Kinematics
    from .expert import Expert
    from .assets import robot_urdf

    ik = Kinematics(robot_urdf(args.data), cfg)
    support = cfg["table"]["center"][2] + cfg["table"]["size"][2] / 2
    worst_jump = 0.0
    max_duration = 0.0
    for seed in EVAL_SEEDS:
        xy = np.random.default_rng(seed).uniform(
            cfg["cube"]["xy_min"], cfg["cube"]["xy_max"]
        )
        expert = Expert(ik, cfg)
        previous = np.array(cfg["robot"]["home_rad"])
        expert.reset(np.r_[xy, support + 0.015], previous)
        for step in range(300):
            q = expert.action({})
            worst_jump = max(worst_jump, float(abs(q - previous).max()))
            previous = q
            if expert.stage == "settle" and expert.elapsed >= 3.0:
                max_duration = max(max_duration, (step + 1) / cfg["control_hz"])
                break
        else:
            raise RuntimeError(f"Expert trajectory exceeded 30 seconds for seed {seed}")
        print(f"Seed {seed}: IK trajectory reachable", flush=True)
    result = {
        "seeds": 20,
        "max_requested_joint_step_rad": worst_jump,
        "max_trajectory_duration_s": max_duration,
        "physics_validated": False,
    }
    print(json.dumps(result, indent=2))


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "doctor":
        return doctor(args)
    if args.command == "test":
        import pytest

        return pytest.main([str(ROOT / "tests"), "-q"])
    if args.command in ("export", "check-dataset", "video"):
        from . import dataset

        if args.command == "export":
            print(
                json.dumps(
                    dataset.export(args.run, args.dataset, args.repo_id), indent=2
                )
            )
        elif args.command == "check-dataset":
            print(json.dumps(dataset.check(args.dataset, args.repo_id), indent=2))
        else:
            dataset.videos(args.run)
        return 0
    if args.command == "audit":
        from .audit import audit

        print(json.dumps(audit(args.run, args.vram_budget_mib), indent=2))
        return 0
    cfg = load_config(args.config)
    from .assets import fetch

    if args.command == "fetch":
        groups = {
            "all": ("robot", "policy", "vlm"),
            "robot": ("robot",),
            "policy": ("policy", "vlm"),
        }[args.only]
        fetch(args.data, groups, offline=args.offline)
        print("All requested files match sources.lock.json.")
        return 0
    if args.command == "check-kinematics":
        fetch(args.data, ("robot",), offline=True)
        check_kinematics(args, cfg)
        return 0
    if args.command == "policy-smoke":
        import numpy as np
        from .policy_worker import SmolPolicy

        policy = SmolPolicy(args.data, cfg["adapter"], args.device)
        image = np.full((256, 256, 3), 127, dtype=np.uint8)
        result = policy.action(
            np.array(cfg["robot"]["home_rad"], dtype=np.float32),
            image,
            image,
            cfg["task"],
        )
        print(
            json.dumps(
                {
                    "finite_action": result.tolist(),
                    "device": args.device,
                    "task_success_test": False,
                },
                indent=2,
            )
        )
        return 0
    if hasattr(args, "episodes") and args.episodes <= 0:
        raise ValueError("Episode count must be positive.")
    if args.command == "collect" and args.seed < 10000:
        raise ValueError("Demonstration seeds must start at 10000.")
    if hasattr(args, "output") and args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    if not os.environ.get("SO101_SIM_CHILD_STATUS"):
        from .runtime import run_simulator

        return run_simulator(args, list(sys.argv[1:] if argv is None else argv))
    lock = fetch(args.data, ("robot",), offline=True)
    # Import SimulationApp before all Omniverse/Isaac Sim APIs.
    from isaacsim import SimulationApp
    from .runtime import experience

    # Our CLI is already parsed; do not pass its --config/--output options to Kit.
    sys.argv = sys.argv[:1]
    app = SimulationApp(
        {
            "headless": not args.gui,
            "active_gpu": args.gpu,
            "physics_gpu": args.gpu,
            "multi_gpu": False,
            "fast_shutdown": True,
            "renderer": "RayTracedLighting",
            "width": 640,
            "height": 480,
            "anti_aliasing": 0,
            "disable_viewport_updates": not args.gui,
        },
        experience=experience(args.data),
    )
    env = None
    failure = None
    try:
        from .environment import Environment, convert_robot

        if args.command == "convert":
            print(convert_robot(args.data))
            return 0
        env = Environment(cfg, args.data)
        if args.command in ("collect", "evaluate"):
            from .runner import run

            run(env, cfg, args, lock)
        else:
            import numpy as np
            from PIL import Image

            args.output.mkdir(parents=True)
            seed = 12345 if args.command == "check-scene" else args.seed
            obs = env.reset(seed)
            first_position, first_state = (
                env.initial_position.copy(),
                obs["state"].copy(),
            )
            for name, rgb in obs["images"].items():
                if rgb.std() < 2:
                    raise RuntimeError(
                        f"Camera {name} returned a nearly uniform frame."
                    )
                Image.fromarray(rgb).save(args.output / f"{name}.png")
            if args.command == "check-scene":
                from .metrics import cube_corners

                cube_points = cube_corners(
                    first_position, np.eye(3), cfg["cube"]["size"]
                )
                target_xy = np.array(cfg["target"]["center_xy"])
                half = np.array(cfg["target"]["size"]) / 2
                target_points = np.array(
                    [
                        [*(target_xy + half * [x, y]), env.support_z]
                        for x in (-1, 1)
                        for y in (-1, 1)
                    ]
                )
                pixels = env.cameras["top"].get_image_coords_from_world_points(
                    np.concatenate((cube_points, target_points))
                )
                if not (
                    np.isfinite(pixels).all()
                    and (pixels >= 0).all()
                    and (pixels < np.array(cfg["cameras"]["resolution"])).all()
                ):
                    raise RuntimeError(
                        "Top camera must frame the whole cube and target."
                    )
                env.reset(seed + 1)
                repeat = env.reset(seed)
                np.testing.assert_allclose(
                    first_position, env.initial_position, atol=1e-5, rtol=0
                )
                np.testing.assert_allclose(
                    first_state, repeat["state"], atol=1e-4, rtol=0
                )
                np.testing.assert_allclose(
                    repeat["state"],
                    cfg["robot"]["home_rad"],
                    atol=0.05,
                    rtol=0,
                    err_msg="Joint drives did not hold the configured home pose.",
                )
                before = env.world.current_time
                action = repeat["state"].copy()
                action[0] += 0.1
                moved, _ = env.step(action)
                elapsed = env.world.current_time - before
                np.testing.assert_allclose(
                    elapsed, 1 / cfg["control_hz"], atol=1e-7, rtol=0
                )
                if moved["state"][0] - repeat["state"][0] < 0.01:
                    raise RuntimeError("The shoulder drive did not follow its target.")
                if (
                    np.count_nonzero(
                        np.abs(
                            moved["images"]["top"].astype(float)
                            - repeat["images"]["top"].astype(float)
                        )
                        > 10
                    )
                    < 20
                ):
                    raise RuntimeError(
                        "Camera frame did not update after joint motion."
                    )
                result = {
                    "reset_repeatable": True,
                    "cameras": list(obs["images"]),
                    "control_step_sim_time_s": elapsed,
                    "capture_advances_physics": False,
                    "joint_drive_and_render_motion": True,
                    "top_cube_and_target_in_frame": True,
                    "camera_intrinsics": {
                        name: cam.get_intrinsics_matrix().tolist()
                        for name, cam in env.cameras.items()
                    },
                    "initial_position": first_position.tolist(),
                    "config": cfg,
                }
                (args.output / "checks.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                print("Both RGB cameras and repeated seeded reset passed.")
            else:
                for _ in range(min(int(args.seconds * cfg["control_hz"]), 300)):
                    _, done = env.step(obs["state"])
                    if done:
                        break
    except BaseException as exc:
        failure = exc
        print(f"Simulation command failed: {exc}", file=sys.stderr, flush=True)
    finally:
        if env is not None:
            env.close()
        Path(os.environ["SO101_SIM_CHILD_STATUS"]).write_text(
            json.dumps(
                {
                    "error": None
                    if failure is None
                    else f"{type(failure).__name__}: {failure}"
                }
            )
        )
        sys.stdout.flush()
        sys.stderr.flush()
        app.close()
    if failure is not None:
        raise failure
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
