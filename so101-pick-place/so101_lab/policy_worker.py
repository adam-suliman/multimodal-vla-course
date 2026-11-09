"""Run in the locked LeRobot environment, never in Isaac Sim's Python process."""

import argparse
import json
from pathlib import Path
import socket
import numpy as np
from .assets import fetch
from .protocol import receive, send


class SmolPolicy:
    def __init__(self, data_dir, adapter, device):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies.factory import make_pre_post_processors

        fetch(data_dir, ("policy", "vlm"), offline=True)
        data_dir = Path(data_dir).resolve()
        path = data_dir / "policy"
        config = PreTrainedConfig.from_pretrained(path, local_files_only=True)
        config.device = device
        config.vlm_model_name = str(data_dir / "vlm")
        # All VLM weights are already in the fine-tuned policy. Avoid a second download.
        config.load_vlm_weights = False
        self.policy = SmolVLAPolicy.from_pretrained(
            path, config=config, local_files_only=True, strict=True
        )
        self.pre, self.post = make_pre_post_processors(
            config,
            pretrained_path=str(path),
            preprocessor_overrides={
                "device_processor": {"device": device},
                "tokenizer_processor": {"tokenizer_name": str(data_dir / "vlm")},
            },
        )
        self.scale = np.array(adapter["scale_per_radian"], dtype=np.float32)
        self.offset = np.array(adapter["offset"], dtype=np.float32)
        self.device = device
        self.torch = torch
        self.reset(0)

    def reset(self, seed):
        self.torch.manual_seed(seed)
        if self.device.startswith("cuda"):
            self.torch.cuda.manual_seed_all(seed)
            self.torch.cuda.reset_peak_memory_stats()
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    def action(self, state, top, wrist, task):
        torch = self.torch
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ValueError("Policy state must be six finite joint angles.")
        batch = {
            "observation.state": torch.from_numpy(state * self.scale + self.offset),
            "task": str(task),
        }
        for name, rgb in (("top", top), ("left_wrist", wrist)):
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError("Policy cameras must be uint8 HWC RGB images.")
            batch[f"observation.images.{name}"] = (
                torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
            )
        with torch.inference_mode():
            action = self.post(self.policy.select_action(self.pre(batch)))
        action = action.squeeze(0).cpu().numpy()
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("Policy returned an invalid action.")
        return ((action - self.offset) / self.scale).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    policy = SmolPolicy(
        args.data, json.loads(Path(args.adapter).read_text()), args.device
    )
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(args.socket)
        server.listen(1)
        conn, _ = server.accept()
        with conn:
            send(
                conn,
                device=args.device,
                chunk_size=policy.policy.config.chunk_size,
                action_steps=policy.policy.config.n_action_steps,
                control_hz=10,
            )
            while True:
                try:
                    request = receive(conn)
                except EOFError:
                    break
                try:
                    command = str(request["command"].item())
                    if command == "reset":
                        policy.reset(int(request["seed"]))
                        send(conn, ready=True)
                    elif command == "action":
                        result = policy.action(
                            request["state"],
                            request["top"],
                            request["wrist"],
                            request["task"].item(),
                        )
                        send(conn, action=result)
                    else:
                        raise ValueError(f"Unknown policy command: {command}")
                except Exception as exc:
                    send(conn, error=f"{type(exc).__name__}: {exc}")
                    raise


if __name__ == "__main__":
    main()
