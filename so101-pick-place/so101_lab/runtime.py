"""Small Isaac application, matching the successfully rendered B1K diagnostic."""

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile


def run_simulator(args, argv):
    """Preserve CLI errors across Kit's default fast process shutdown.

    Kit 107 can crash during Python teardown with graceful shutdown enabled.
    The simulator writes its result before fast shutdown; the parent requires
    both that result and a clean process exit, and then creates diagnostic videos.
    """
    with tempfile.TemporaryDirectory(prefix="so101-sim-") as directory:
        status = Path(directory) / "status.json"
        child_env = os.environ.copy()
        child_env["SO101_SIM_CHILD_STATUS"] = str(status)
        child = subprocess.run(
            [sys.executable, "-u", "-m", "so101_lab", *argv], env=child_env
        )
        report = json.loads(status.read_text()) if status.exists() else None
    if (
        getattr(args, "video", False)
        and args.command in ("collect", "evaluate")
        and (args.output / "run.json").exists()
    ):
        video_env = os.environ.copy()
        video_env.pop("LD_LIBRARY_PATH", None)
        video_env.pop("PYTHONHOME", None)
        subprocess.run(
            [args.policy_python, "-m", "so101_lab", "video", "--run", str(args.output)],
            check=True,
            env=video_env,
        )
    if child.returncode != 0 or report is None:
        raise RuntimeError(
            f"Simulator exited without a clean completion (exit {child.returncode})."
        )
    if report["error"] is not None:
        raise RuntimeError(report["error"])
    return 0


def experience(data_dir):
    # Call after importing isaacsim, before starting SimulationApp.
    import isaacsim

    root = Path(isaacsim.__file__).resolve().parent
    if (
        not (root / "exts/isaacsim.core.api").is_dir()
        and Path("/isaac-sim/exts").is_dir()
    ):
        root = Path("/isaac-sim")
    dependencies = (
        "isaacsim.core.api",
        "isaacsim.sensors.camera",
        "isaacsim.simulation_app",
        "isaacsim.core.version",
        "omni.kit.loop-isaac",
        "omni.hydra.rtx",
        "omni.kit.viewport.window",
        "omni.physx.tensors",
        "omni.replicator.core",
        "omni.syntheticdata",
    )
    text = (
        '[package]\ntitle = "SO-101 laboratory"\nversion = "1.0.0"\n\n[dependencies]\n'
    )
    text += "".join(f'"{name}" = {{}}\n' for name in dependencies)
    text += """
[settings.app]
name = "SO-101 Isaac Sim 5.1"
version = "5.1.0"
vulkan = true
settings.persistent = false
settings.fabricDefaultStageFrameHistoryCount = 3
extensions.registryEnabled = false
runLoops.main.rateLimitEnabled = false

[settings.app.exts.folders]
'++' = [
"""
    text += "".join(
        f'    "{root / name}",\n' for name in ("exts", "extscache", "extsDeprecated")
    )
    text += "]\n"
    target = Path(data_dir).resolve() / "runtime/so101.kit"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return str(target)
