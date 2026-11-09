# Public Linux/amd64 Isaac Sim 5.1.0 manifest, verified against NGC.
FROM nvcr.io/nvidia/isaac-sim:5.1.0@sha256:93b0f99635ab126fb5b33298d513c11520f119f0ee60ff8414ccef67ea977829 AS runtime-installer
USER root
WORKDIR /opt/so101
# evdev (a transitive LeRobot dependency) is distributed as C source.
# Freeze the compiler and its transitive OS dependencies using Ubuntu's snapshot.
COPY ubuntu-build.sources /tmp/so101.sources
RUN apt-get -o Dir::Etc::sourcelist=/tmp/so101.sources -o Dir::Etc::sourceparts=- update && \
    DEBIAN_FRONTEND=noninteractive apt-get -o Dir::Etc::sourcelist=/tmp/so101.sources \
    -o Dir::Etc::sourceparts=- install -y --no-install-recommends build-essential=12.10ubuntu1
COPY requirements-bootstrap.lock /opt/so101/
RUN /isaac-sim/python.sh -m venv /opt/installer && \
    /opt/installer/bin/python -m pip install --no-cache-dir --require-hashes -r requirements-bootstrap.lock

FROM runtime-installer AS environments
COPY requirements-policy.lock requirements-simulator.lock requirements-build.lock /opt/so101/
COPY scripts/download_wheels.py /opt/so101/scripts/download_wheels.py
RUN /opt/installer/bin/python scripts/download_wheels.py --directory /tmp/wheels \
    requirements-simulator.lock requirements-policy.lock requirements-build.lock && \
    /opt/installer/bin/uv venv --python /isaac-sim/kit/python/bin/python3 /opt/sim && \
    /opt/installer/bin/uv venv --python /isaac-sim/kit/python/bin/python3 /opt/policy && \
    /opt/installer/bin/uv pip sync --no-cache --offline --no-index --find-links /tmp/wheels \
    --python /opt/sim/bin/python --require-hashes \
    --build-constraints requirements-build.lock requirements-simulator.lock && \
    /opt/installer/bin/uv pip sync --no-cache --offline --no-index --find-links /tmp/wheels \
    --python /opt/policy/bin/python --require-hashes \
    --build-constraints requirements-build.lock requirements-policy.lock && \
    /opt/installer/bin/uv pip check --python /opt/sim/bin/python && \
    /opt/installer/bin/uv pip check --python /opt/policy/bin/python && rm -rf /tmp/wheels

# Reuse the installed environment layer directly. Copying both environments into
# another stage duplicates roughly 22 GB during a build on a Docker overlay store.
# The snapshot-pinned compiler remains in the image; neither environment shares
# Python packages with the other or with the base image's simulator.
FROM environments AS runtime
COPY . /opt/so101/
ENV PYTHONPATH=/opt/so101 OMNI_KIT_ALLOW_ROOT=1 OMNI_KIT_ACCEPT_EULA=YES TOKENIZERS_PARALLELISM=false
ENTRYPOINT ["/opt/so101/scripts/entrypoint.sh"]
CMD ["doctor"]
