"""Episode orchestration with a fixed denominator and an auditable run manifest."""

import json
import hashlib
import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import threading
import time
import traceback
from .config import ROOT, PROTOCOL, EVAL_SEEDS, fingerprint
from .expert import Expert
from .recording import EpisodeWriter


class GPUMonitor:
    """Sample total device usage, including both processes and external GPU clients."""

    def __init__(self, path, gpu=0):
        self.path, self.gpu = Path(path), gpu
        self.peak_mib = None
        self.total_mib = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        with self.path.open("w") as out:
            while not self.stop.is_set():
                try:
                    value = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "-i",
                            str(self.gpu),
                            "--query-gpu=memory.used,memory.total,name,driver_version",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        timeout=5,
                    ).strip()
                    used, total = [float(x.strip()) for x in value.split(",")[:2]]
                    self.peak_mib = max(self.peak_mib or 0, used)
                    self.total_mib = total
                    out.write(json.dumps({"time": time.time(), "sample": value}) + "\n")
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    out.write(
                        json.dumps({"time": time.time(), "error": str(exc)}) + "\n"
                    )
                out.flush()
                self.stop.wait(0.5)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=6)


def run(env, cfg, args, lock):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    seeds = (
        list(EVAL_SEEDS)
        if args.command == "evaluate"
        else list(range(args.seed, args.seed + args.episodes))
    )
    if args.command == "collect" and (
        min(seeds) < 10000 or set(seeds) & set(EVAL_SEEDS)
    ):
        raise ValueError(
            "Demonstration seeds must start at 10000 and must not overlap evaluation seeds."
        )
    meta = {
        "mode": args.command,
        "policy": args.policy,
        "seeds": seeds,
        "config": cfg,
        "config_sha256": fingerprint(cfg),
        "protocol": PROTOCOL,
        "sources": lock,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "complete": False,
    }
    meta["code_sha256"] = fingerprint(
        {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "so101_lab").glob("*.py"))
        }
    )
    meta["runtime_locks"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("requirements-simulator.lock", "requirements-policy.lock")
    }
    meta["simulator_packages"] = {}
    for name in ("isaacsim", "torch", "numpy"):
        try:
            meta["simulator_packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    (output / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    for name in ("pending", "successes", "failures"):
        (output / name).mkdir()
    monitor = GPUMonitor(output / "gpu-memory.jsonl", args.gpu)
    policy = None
    results = []
    fatal = None
    try:
        if args.policy == "smolvla":
            from .policy import PolicyClient

            policy = PolicyClient(
                args.data,
                output / "config.json",
                output / "policy.log",
                python=args.policy_python,
                device=args.policy_device or f"cuda:{args.gpu}",
            )
            meta["policy_runtime"] = policy.metadata
        else:
            policy = Expert(env.ik, cfg)
        for seed in seeds:
            writer = EpisodeWriter(output / "pending", seed, cfg)
            try:
                obs = env.reset(seed)
                if args.policy == "expert":
                    policy.reset(env.initial_position, obs["state"])
                else:
                    policy.reset(seed)
                done = False
                while not done:
                    action = policy.action(obs)
                    next_obs, done = env.step(action)
                    writer.add(obs, env.last_action, done, env.last_samples)
                    obs = next_obs
                result = env.result()
            except Exception as exc:
                result = {
                    "seed": seed,
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            directory = writer.finish(result)
            shutil.move(
                str(directory),
                str(
                    output
                    / ("successes" if result["success"] else "failures")
                    / directory.name
                ),
            )
            results.append(result)
            (output / "episodes.json").write_text(json.dumps(results, indent=2) + "\n")
            print(
                f"Seed {seed}: {'success' if result['success'] else 'failure'}"
                + (f" ({result['error']})" if result.get("error") else ""),
                flush=True,
            )
    except BaseException as exc:
        fatal = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if policy is not None and args.policy == "smolvla":
            policy.close()
        monitor.close()
        meta["gpu_memory"] = {
            "sampled_peak_mib": getattr(monitor, "peak_mib", None),
            "total_mib": getattr(monitor, "total_mib", None),
            "sample_interval_s": 0.5,
            "includes_other_processes": True,
        }
        successes = sum(r["success"] for r in results)
        complete = (
            len(results) == len(seeds)
            and not any(r.get("error") for r in results)
            and fatal is None
        )
        meta.update(
            complete=complete,
            fatal_error=fatal,
            episodes=results,
            successes=successes,
            denominator=len(seeds),
            success_rate=successes / len(seeds) if complete else None,
            expert_acceptance=bool(
                complete
                and args.command == "evaluate"
                and args.policy == "expert"
                and successes == 20
                and all(r.get("bilateral_contact_seen") for r in results)
            ),
        )
        (output / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: meta[k]
                for k in (
                    "complete",
                    "successes",
                    "denominator",
                    "success_rate",
                    "expert_acceptance",
                )
            }
        )
    )
    if not meta["complete"]:
        raise RuntimeError(
            f"Run has execution errors. No valid SR reported. See {output}/run.json"
        )
    if (
        args.command == "evaluate"
        and args.policy == "expert"
        and not meta["expert_acceptance"]
    ):
        raise RuntimeError(
            "Expert acceptance requires 20/20 successes and bilateral contact evidence."
        )
    return output
