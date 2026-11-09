"""Client for isolated SmolVLA inference; never forwards privileged simulator data."""

import os
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from .config import ROOT
from .protocol import send, receive


class PolicyClient:
    def __init__(
        self,
        data_dir,
        config_path,
        log_path,
        python="/opt/policy/bin/python",
        device="cuda:0",
    ):
        self.directory = tempfile.TemporaryDirectory(prefix="so101-policy-")
        adapter_path = Path(self.directory.name) / "adapter.json"
        config = json.loads(Path(config_path).read_text())
        adapter_path.write_text(
            json.dumps(
                {key: config["adapter"][key] for key in ("scale_per_radian", "offset")}
            )
        )
        self.log = open(log_path, "w")
        self.sock = None
        path = str(Path(self.directory.name) / "inference.sock")
        env = os.environ.copy()
        for key in ("PYTHONHOME", "LD_LIBRARY_PATH"):
            env.pop(key, None)
        env.update(
            PYTHONPATH=str(ROOT),
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            TOKENIZERS_PARALLELISM="false",
        )
        self.process = subprocess.Popen(
            [
                python,
                "-m",
                "so101_lab.policy_worker",
                "--socket",
                path,
                "--data",
                str(Path(data_dir).resolve()),
                "--adapter",
                str(adapter_path),
                "--device",
                device,
            ],
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 240
            while not Path(path).exists():
                if self.process.poll() is not None:
                    raise RuntimeError(f"Policy initialization failed; see {log_path}")
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Policy initialization timed out; see {log_path}"
                    )
                time.sleep(0.1)
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(240)
            self.sock.connect(path)
            self.metadata = {k: v.tolist() for k, v in receive(self.sock).items()}
        except BaseException:
            self.close()
            raise

    def reset(self, seed):
        send(self.sock, command="reset", seed=seed)
        receive(self.sock)

    def action(self, observation):
        # Explicit allowlist: the expert/evaluator cannot add object poses to this message.
        send(
            self.sock,
            command="action",
            state=observation["state"],
            task=observation["task"],
            top=observation["images"]["top"],
            wrist=observation["images"]["wrist"],
        )
        response = receive(self.sock)
        if "error" in response:
            raise RuntimeError(str(response["error"].item()))
        return response["action"]

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.log.close()
        self.directory.cleanup()
