"""Run Docker with one GPU's existing device nodes and read-only driver libraries.

An explicit fallback for a host whose NVIDIA runtime hook fails while the selected
GPU works. No privileged container, driver installation or daemon edit is used.
All normal Docker arguments follow this script's GPU index argument.
"""

import glob
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def arguments(gpu):
    if not gpu.isdigit():
        raise ValueError("Direct GPU mode requires a numeric host GPU index.")
    query = ET.fromstring(
        subprocess.check_output(
            ["nvidia-smi", "-i", gpu, "-q", "-x"],
            text=True,
        )
    )
    minor, driver = query.findtext("gpu/minor_number"), query.findtext("driver_version")
    if minor is None or not minor.isdigit() or not driver:
        raise RuntimeError(
            "nvidia-smi did not identify the selected GPU device and driver."
        )
    args = ["--runtime=runc", "-e", "NVIDIA_VISIBLE_DEVICES=void"]
    for name in (f"nvidia{minor}", "nvidiactl", "nvidia-uvm", "nvidia-uvm-tools"):
        path = Path("/dev") / name
        if not path.exists():
            raise FileNotFoundError(path)
        args.extend(["--device", str(path)])
    # Map SONAMEs as well as versioned files. The kernel driver stays on the host.
    directory = Path("/usr/lib/x86_64-linux-gnu")
    mounted = set()
    for pattern in (
        "libcuda.so*",
        "libcudadebugger.so*",
        "libnvidia-*.so*",
        "libGLX_nvidia.so*",
        "libEGL_nvidia.so*",
        "libnvcuvid.so*",
        "libnvoptix.so*",
    ):
        for name in sorted(glob.glob(str(directory / pattern))):
            path = Path(name)
            if "container" in path.name or path.name in mounted:
                continue
            resolved = path.resolve()
            # Avoid stale libraries from another driver branch.
            if (
                resolved.name.endswith(tuple("0123456789"))
                and driver not in resolved.name
            ):
                if not any(part in path.name for part in ("egl-gbm", "egl-wayland")):
                    continue
            args.extend(
                [
                    "--mount",
                    f"type=bind,src={resolved},dst={directory / path.name},readonly",
                ]
            )
            mounted.add(path.name)
    if "libcuda.so.1" not in mounted or "libGLX_nvidia.so.0" not in mounted:
        raise RuntimeError(
            "Could not locate the selected driver's CUDA/Vulkan libraries."
        )
    for source, target in (
        ("/usr/bin/nvidia-smi", "/usr/bin/nvidia-smi"),
        (
            "/usr/share/vulkan/icd.d/nvidia_icd.json",
            "/etc/vulkan/icd.d/nvidia_icd.json",
        ),
        (
            "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
            "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
        ),
    ):
        if Path(source).is_file():
            args.extend(["--mount", f"type=bind,src={source},dst={target},readonly"])
    args.extend(["-e", "VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json"])
    print(
        f"Using host GPU {gpu}, existing driver {driver}, direct device mounts.",
        file=sys.stderr,
    )
    return args


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("Usage: direct_gpu.py HOST_GPU [docker run arguments...]")
    os.execvp("docker", ["docker", "run", *arguments(sys.argv[1]), *sys.argv[2:]])
