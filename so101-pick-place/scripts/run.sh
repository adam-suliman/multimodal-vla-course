#!/usr/bin/env bash
# Run from any directory. All writable content is inside this module's data/ and runs/.
set -euo pipefail
module_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${ACCEPT_EULA:-}" != Y ]]; then
    echo 'Set ACCEPT_EULA=Y after reading the NVIDIA Isaac Sim license linked in README.md.' >&2
    exit 1
fi
mkdir -p "$module_dir/data/cache/kit" "$module_dir/runs"
docker_command=(docker run --runtime=nvidia --gpus "device=${GPU_INDEX:-0}")
if [[ "${SO101_GPU_MODE:-toolkit}" == direct ]]; then
    docker_command=(python3 "$module_dir/scripts/direct_gpu.py" "${GPU_INDEX:-0}")
fi
"${docker_command[@]}" --rm --shm-size=2g \
    -e HTTPS_PROXY -e HTTP_PROXY -e NO_PROXY \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT="${PRIVACY_CONSENT:-N}" \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e XDG_CACHE_HOME=/opt/so101/data/cache \
    -v "$module_dir/data:/opt/so101/data" \
    -v "$module_dir/data/cache/kit:/opt/sim/lib/python3.11/site-packages/isaacsim/kit/cache" \
    -v "$module_dir/runs:/opt/so101/runs" \
    "${SO101_IMAGE:-so101-course:0.1}" "$@"
