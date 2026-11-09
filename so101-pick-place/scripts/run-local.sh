#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ "${ACCEPT_EULA:-}" != Y ]]; then
    echo 'Set ACCEPT_EULA=Y after reading the NVIDIA license linked in README.md.' >&2
    exit 1
fi
if [[ ! -f .runtime/sim/pyvenv.cfg || ! -f .runtime/policy/pyvenv.cfg ]]; then
    echo 'Run scripts/setup-local.sh first.' >&2
    exit 1
fi
mkdir -p data runs .runtime/cache
docker_command=(docker run --runtime=nvidia --gpus "device=${GPU_INDEX:-0}")
if [[ "${SO101_GPU_MODE:-toolkit}" == direct ]]; then
    docker_command=(python3 scripts/direct_gpu.py "${GPU_INDEX:-0}")
fi
exec "${docker_command[@]}" --rm --user 0 --shm-size=2g \
    -e HTTPS_PROXY -e HTTP_PROXY -e NO_PROXY \
    -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e OMNI_KIT_ALLOW_ROOT=1 \
    -e NVIDIA_DRIVER_CAPABILITIES=all -e PRIVACY_CONSENT="${PRIVACY_CONSENT:-N}" \
    -e XDG_CACHE_HOME=/opt/so101/.runtime/cache -e PYTHONPATH=/opt/so101 \
    -e TOKENIZERS_PARALLELISM=false \
    -v "$PWD:/opt/so101" -v "$PWD/.runtime/sim:/opt/sim" \
    -v "$PWD/.runtime/policy:/opt/policy" \
    --workdir /opt/so101 --entrypoint /opt/so101/scripts/entrypoint.sh \
    nvcr.io/nvidia/isaac-sim:5.1.0@sha256:93b0f99635ab126fb5b33298d513c11520f119f0ee60ff8414ccef67ea977829 "$@"
