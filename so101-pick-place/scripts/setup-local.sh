#!/usr/bin/env bash
# The same public wheels as Dockerfile, persisted in module-local environments.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p .runtime/wheels .runtime/sim .runtime/policy .runtime/cache
docker build --network="${SO101_BUILD_NETWORK:-default}" \
    --build-arg HTTPS_PROXY --build-arg HTTP_PROXY --build-arg NO_PROXY \
    --target runtime-installer -t so101-installer:0.1 .
docker run --rm --runtime=runc -e NVIDIA_VISIBLE_DEVICES=void --network host \
    -e HTTPS_PROXY -e HTTP_PROXY -e NO_PROXY \
    -e UV_CACHE_DIR=/opt/so101/.runtime/cache/uv \
    -v "$PWD:/opt/so101" --entrypoint /bin/bash so101-installer:0.1 -c '
set -euo pipefail
/opt/installer/bin/python scripts/download_wheels.py --directory .runtime/wheels \
    requirements-simulator.lock requirements-policy.lock requirements-build.lock
for environment in sim policy; do
    /opt/installer/bin/uv venv --allow-existing --python /isaac-sim/kit/python/bin/python3 ".runtime/$environment"
    requirements=simulator
    if [[ "$environment" == policy ]]; then requirements=policy; fi
    /opt/installer/bin/uv pip sync --offline --no-index --find-links .runtime/wheels \
        --require-hashes --build-constraints requirements-build.lock \
        --python ".runtime/$environment/bin/python" "requirements-$requirements.lock"
    /opt/installer/bin/uv pip check --python ".runtime/$environment/bin/python"
done
'
printf 'Local environments are ready. Run ACCEPT_EULA=Y ./scripts/run-local.sh doctor\n'
