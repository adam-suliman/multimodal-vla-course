# SO-101 pick-and-place laboratory

A self-contained course module for a contact-driven SO-101 arm in Isaac Sim 5.1.0,
with an IK expert, LeRobot demonstration export, and an isolated SmolVLA process.
All source assets and model files come from immutable public URLs. No developer
container, home directory, cache, ROS installation, physical robot, or Hugging Face
account is required.

**Measured with the original yellow robot on RTX 3060 GPU 0, driver 535.230.02:** the expert passed **20/20**
fixed seeds; SmolVLA scored **0/20** without execution errors. Simultaneous
simulator/inference memory peaked at **3,348 MiB** (sampled). Physical LeRobot
export/read-back and 32 software tests passed. Both installation paths and the quickstart from a clean local Git clone passed.
A published checkout on an empty Docker image store remains a release check. See
[VALIDATION.md](VALIDATION.md) for evidence and limits.

The current scene uses white printed robot parts and dark servos, matching the
course robots. `robot.printed_color` in `configs/default.json` controls the RGB
plastic color. This scene override preserves the pinned upstream asset and its
checksums. The results above predate the color change; the fixed 20-seed evaluation has not been repeated for the white
robot. The white scene passed camera/reset/joint-motion
checks: [top view](validation/white-scene-top.png),
[wrist view](validation/white-scene-wrist.png). Newly recorded videos show the
[expert](validation/expert-pilot.mp4) succeeding in 18.5 s and
[SmolVLA](validation/smolvla-example.mp4) timing out after 30 s on the same
demonstration seed (10000), both with the white robot. These single episodes are
separate from the historical benchmark.

## Requirements and licenses

Use Linux x86-64 (Ubuntu 22.04 or 24.04), Docker Engine, NVIDIA Container Toolkit,
an NVIDIA RTX GPU, at least 32 GB RAM, and approximately 90 GB free disk space for
images, downloads, build intermediates, and initial recordings. Larger datasets
need additional storage. Internet access is required for the initial build and
asset download. The base image is about 7.6 GB compressed / 15.1 GB unpacked; the complete course
image is approximately 38.7 GB. Allow tens of
minutes for a first build and several minutes for a first cold RTX startup.

The course validation target is **RTX 3060 12 GB, host GPU 0**. NVIDIA's published
Isaac Sim 5.1 reference requirements specify RTX 4080 / 16 GB and Linux driver
580.65.06. The image declares a 570.169 reference minimum. An older driver is
reported as a warning and must pass the actual scene checks; installation never
changes the host driver. The smaller course target needs an actual joint
simulator/inference memory test. Two 256×256 cameras and one environment are the default. CPU physics
still requires an RTX GPU for the cameras. H100/A100 do not provide that renderer.
See [NVIDIA's requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html).

Before running, read the [Isaac Sim license](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/common/licenses.html)
and the [container terms](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim).
`ACCEPT_EULA=Y` expresses your acceptance; the wrapper requires you to set it.
`PRIVACY_CONSENT` defaults to `N`. SO-101 assets and the policy have separate
Apache-2.0 licenses; the VLM configuration/tokenizer is also Apache-2.0. See
[PROVENANCE.md](PROVENANCE.md). The Dockerfile does not embed model weights or CAD
files: the fetch command downloads them to your own data directory.

Install Docker following [Docker's Ubuntu instructions](https://docs.docker.com/engine/install/ubuntu/).
Install NVIDIA Container Toolkit following
[NVIDIA's installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
then configure its Docker runtime as described there. Check the host first:

```bash
nvidia-smi -i 0
# After installing the toolkit:
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

The last two commands change host configuration and restart Docker; coordinate
with other users of a shared machine. They are installation instructions, not
commands performed by the module.

## Fresh-clone quickstart

Run these commands in a terminal on the GPU host. No notebook is needed.

```bash
git clone https://github.com/wingrune/multimodal-vla-course.git
cd multimodal-vla-course/so101-pick-place

docker build --pull --no-cache -t so101-course:0.1 .
export ACCEPT_EULA=Y
export GPU_INDEX=0
./scripts/run.sh doctor
./scripts/run.sh fetch
./scripts/run.sh convert
./scripts/run.sh check-kinematics
./scripts/run.sh check-scene --output runs/scene-check
```

Use a repository revision containing this module. Until the changes are published,
these commands describe the intended published checkout; a clone of an older
revision will not contain this directory.

`fetch` verifies SHA-256 and size for every file in [sources.lock.json](sources.lock.json).
The robot URDF and all referenced meshes are pinned to a Git commit. The model,
its normalization statistics, and the VLM configuration/tokenizer have separate
pinned Hugging Face revisions. A corrupted cache is re-downloaded. Use
`fetch --offline` to verify without network access. `convert` imports the URDF
with Isaac Sim's built-in importer, a fixed base and convex
collision decomposition. The conversion copy drops the geometry-free TCP marker
so its zero-inertia dummy link does not become a rigid body. It hashes the
resulting USD bundle. A changed source revision or conversion recipe invalidates this derived cache.

Inspect `runs/scene-check/top.png` and `wrist.png`. `check-scene` tests both RGB
cameras, an A–B–A seeded reset, top-camera coverage of the cube and target, and
joint tracking, changing frames after movement, and the actual physics/control
timing. Camera focal length and sensor aperture are
expressed in metres in the configuration. The renderer uses a common 100× scale
for both optical lengths, preserving the pinhole field of view while avoiding
clamping at very small focal lengths; depth of field is disabled. See the
[USD camera units](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/cameras.html). Run a short static scene with:

```bash
./scripts/run.sh scene --seed 0 --seconds 5 --output runs/preview
```

The wrapper selects one physical GPU with Docker; **inside the container it is
GPU 0**. For another host GPU, change `GPU_INDEX`, leaving CLI `--gpu 0` and
`--policy-device cuda:0`. By default everything is headless. `--gui` is available
for a manually launched container with a working X11/Wayland display and the
required display mounts; the standard wrapper deliberately targets headless use.

## Separate wheel installation and an existing host driver

The alternative setup reproduces the B1K installation method: a Python 3.11
virtual environment, official NVIDIA 5.1.0.0 wheels including extension caches,
and PyTorch CUDA 12.6. It uses the same public base image for Linux libraries and
Python, and keeps two independent environments under this module's `.runtime/`.
It does not mount or depend on any B1K checkout, environment or local image.
Launch these environments with `run-local.sh`; their Python interpreter belongs
to the container, so they are not host Python environments to activate directly.
Both environments have complete hash locks. See NVIDIA's
[Python installation documentation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html).

```bash
./scripts/setup-local.sh
export ACCEPT_EULA=Y GPU_INDEX=0
./scripts/run-local.sh doctor
./scripts/run-local.sh fetch
./scripts/run-local.sh check-scene --output runs/local-scene-check
SO101_RUNNER=./scripts/run-local.sh ./scripts/acceptance.sh
```

The downloader resumes large wheels with HTTP byte ranges and verifies every file
against the committed lock. `.runtime/wheels` is optional download storage; an
empty directory works. Failed downloads can be resumed by repeating setup. On
hosts affected by the NGC BuildKit issue, use
`DOCKER_BUILDKIT=0 SO101_BUILD_NETWORK=host ./scripts/setup-local.sh`.

If the NVIDIA Container Toolkit hook fails with an NVML error before Python
starts, but `nvidia-smi -i 0` and the selected GPU still work, both launchers support
an explicit fallback:

```bash
export SO101_GPU_MODE=direct
./scripts/run-local.sh check-scene --output runs/direct-scene-check
```

This exposes only the selected GPU device nodes and mounts the **already installed**
host CUDA/Vulkan driver libraries read-only into a container using `runc`.
It does not install a driver, disable Isaac's driver checks, change the Docker
daemon, or require `--privileged`. It currently targets Linux x86-64 systems with
NVIDIA libraries in `/usr/lib/x86_64-linux-gnu`. A working CUDA tensor alone does
not certify rendering; require `check-scene` and the task acceptance checks.
The small Kit application and synchronous Replicator capture are shared by both
installation paths. Capture uses zero timeline advancement so the physics
samples retain their fixed-rate meaning.

## Expert, demonstrations, and LeRobot

```bash
./scripts/run.sh evaluate --policy expert --output runs/expert-eval --video
./scripts/run.sh audit --run runs/expert-eval
./scripts/run.sh collect --seed 10000 --episodes 10 --output runs/demos --video
./scripts/run.sh export --run runs/demos --dataset runs/lerobot-demos
./scripts/run.sh check-dataset --dataset runs/lerobot-demos
```

The expert follows Cartesian approach, descent, finger closure, lift, transfer,
lowering, release, retreat and settling phases. Bounded URDF IK solves five arm
joints; the sixth controls the moving jaw. Position drives, Coulomb friction,
gravity and finger contact carry the cube. There is no attachment joint, suction,
parenting, or object teleportation during an episode. `reset` is the only place
that sets cube poses directly.

The table, 3 cm red cube, fixed 10×10 cm blue target, joint gains, speed/effort
limits, sampling bounds, camera poses and task text are configured in
[configs/default.json](configs/default.json). Geometry uses metres, seconds and
radians. The small sampling rectangle is kinematically checked; its physical
reachability remains an acceptance test. The expert can see the cube pose.
SmolVLA only receives two RGB images, six measured joint angles and task text.

Every episode first records lossless PNG images, the pre-action state, the actual
rate-limited joint target, timestamp, transition termination, and physics samples.
Successful collection episodes go into `successes/`; failures go into `failures/`
with the same diagnostics. Training export rejects evaluation runs and evaluation
seeds. Demonstration seeds start at 10000. Export uses the real `LeRobotDataset`
writer, saves episode/task metadata, and reads every frame back with the real
loader. Images are stored losslessly in the dataset; MP4s are optional diagnostics.
The original archive remains available in SI units. Exported state/actions use the
checkpoint calibration described below, with provenance alongside the dataset.

## Evaluate the pretrained policy

```bash
./scripts/run.sh policy-smoke
./scripts/run.sh evaluate --policy smolvla --output runs/smolvla-eval --video
./scripts/run.sh audit --run runs/smolvla-eval
```

The checkpoint is
[`Cache-SCA/smolvla_IsaacLab-SO101_pick_place_baseCaP_100epi_50ep`](https://huggingface.co/Cache-SCA/smolvla_IsaacLab-SO101_pick_place_baseCaP_100epi_50ep/tree/6c048c0a0574d88566cc83170341e5fd4306dae5).
Its training data came from a different IsaacLab scene. This module measures
transfer to the course scene. A zero success rate is a valid result; there is no
minimum SmolVLA SR and no fine-tuning or physical-robot calibration in this version.
The model-only smoke test uses synthetic images to check loading and finite
actions; it is not evidence of task success.

Simulator and policy run in separate processes and Python environments. The
simulator uses NVIDIA Isaac Sim 5.1.0.0 wheels in an isolated Python 3.11 environment,
with NumPy 1.26 and PyTorch 2.7.0 CUDA 12.6. The policy has a complete
hash-locked LeRobot 0.4.4 / Transformers 4.57.1 / PyTorch 2.7.1 CUDA 12.6 environment
with NumPy 2. They exchange bounded, pickle-free messages over a private Unix
socket. No service port is published. Inference uses only verified local files.
The simulator pauses during inference: 30 simulation seconds need not equal 30
wall-clock seconds. `gpu-memory.jsonl` samples total selected-device memory while
both processes are active; close unrelated GPU jobs before measuring memory.

The adapter follows the checkpoint contract:

| Property | Convention |
| --- | --- |
| Joint order | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper |
| Environment action | six absolute joint-position targets, radians |
| Checkpoint state/action | per-joint affine calibrated units, **not degrees** |
| Conversion | `checkpoint = scale_per_radian * radians + offset`; inverse for actions |
| Gripper | angular sixth joint; same affine calibration, not a metre opening |
| Camera mapping | wrist → `left_wrist` → `camera1`; top → `camera2` |
| Third camera | absent; native SmolVLA skips it with `empty_cameras=0` |
| RGB | uint8 HWC → float32 CHW in [0,1], then native resizing/padding |
| Normalization | checkpoint preprocessor and postprocessor statistics |
| Rate | 10 Hz control, 120 Hz CPU PhysX |
| Chunking | checkpoint's 50 actions, queue reset before each episode |

The affine conversion is verified against both state and action columns of all
34,264 training frames, using their paired `radian_urdf0` columns. This establishes
the recorded numeric convention; it does not establish a calibration for any real
arm. The current public upstream collection-code URL returns 404, so the empirical
check is retained in [validation/calibration.json](validation/calibration.json).
FK also matches the recorded TCP positions across all frames within 3.4e-8 m,
confirming the public URDF joint zero convention. The source dataset revision and
its checksum are pinned. To reproduce:

```bash
# Inside the policy environment, or in a container with this entrypoint override:
docker run --rm --entrypoint /opt/policy/bin/python so101-course:0.1 scripts/check_calibration.py
```

## Fixed evaluation protocol

The 20 evaluation seeds are committed in `so101_lab/config.py`. Each reset samples
the cube's XY position uniformly within the configured rectangle. Cube orientation
and target remain fixed. Each episode has a 30-second simulation-time limit.
A success requires all of the following in this order:

1. The cube centre rises at least 0.02 m above its settled initial height.
2. The gripper opens and both finger-contact measurements clear.
3. All eight oriented cube corners lie within the target's XY boundaries; the
   lowest corner rests within 3 mm of the table surface.
4. Linear speed is at most 0.01 m/s and angular speed at most 0.1 rad/s for one
   uninterrupted second. Any violation resets the dwell timer.

The evaluator runs on every physics substep, including during a policy chunk.
It rejects a small lift, holding over the target, partial containment and brief
crossings. The expert gate additionally requires **20/20** and bilateral finger
contact evidence in every episode. Seeds are never removed after failure. A run
with infrastructure errors has `complete=false` and `success_rate=null`, rather
than inventing policy failures or silently shrinking the denominator.

Each run saves seeds, configuration and its hash, source revisions, code hash,
per-episode results, success count, denominator, SR, and memory samples. `audit`
recomputes the metric from saved physical state. Reusing an output directory is an
error. After modifying the scene, report results under the new configuration hash;
do not present a tuned seed subset as the fixed benchmark.

## Notebook and student projects

Open [notebooks/pick_and_place.ipynb](notebooks/pick_and_place.ipynb) in Jupyter on
the GPU host. It calls the exact same CLI using `scripts/run.sh`, displays the two
camera images, runs collection/export/evaluation, and reads the saved results.
The notebook is not a Google Colab environment: the Docker daemon and RTX renderer
must be on the machine running its kernel. For the local wheel installation, set
`SO101_RUNNER=scripts/run-local.sh` before starting Jupyter.

For a project, copy `configs/default.json` to `configs/project.json`, edit camera
poses, table layout, robot drives or task text, rebuild the image, and pass
`--config configs/project.json` to simulator commands. The benchmark CLI enforces
the 3 cm cube, 10×10 cm target and 10 Hz checkpoint contract. If you introduce a
new task or robot, define a new evaluation protocol and adapter as well. Repeat
`check-kinematics`, `check-scene`, expert evaluation and dataset read-back after
changes. Robot mounting, camera alignment and control calibration matter even
when a model produces numerically valid actions.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Docker says permission denied | Follow Docker's post-install permission instructions; do not mount developer environments. |
| NVML error / unknown GPU error before Python starts | Check `nvidia-smi -i 0`. If that GPU works, try the explicit `SO101_GPU_MODE=direct` fallback above; this preserves the installed driver. |
| BuildKit gets 403 from NGC | Try `docker pull` of the exact digest from the Dockerfile, then `DOCKER_BUILDKIT=0 docker build --pull --no-cache -t so101-course:0.1 .` on Docker versions supporting the legacy builder. Registry/network access must work. |
| TLS/decryption/download error | Retry on a reliable connection; hashes remain mandatory. `--network=host` may help a misconfigured Docker bridge. Do not disable certificate or checksum verification. |
| Disk full | Provide space in Docker's data root and this checkout. Never prune other users' images or caches as part of setup. |
| CUDA/Vulkan initialization fails | Check the NVIDIA compatibility checker and driver version. An RTX GPU and working graphics libraries are required even for headless CPU physics. |
| `No adjacent samples found for interpolation` | Isaac 5.1 emits this during capture without advancing physics; NVIDIA confirms it is a logging issue ([upstream #403](https://github.com/isaac-sim/IsaacSim/issues/403)). Require the timing and moving-frame checks to pass. |
| Black/unavailable camera | Run `check-scene`, inspect the full simulator log, camera clipping planes and camera pose. |
| Policy returns invalid values | Inspect `policy.log`, the model revision, processor files and joint calibration. Invalid actions abort that episode. |
| Out of GPU memory | Close unrelated jobs, retain one environment and small cameras, inspect `gpu-memory.jsonl`. `--policy-device cpu` is diagnostic only and must be reported as a changed runtime. |
| No exported episodes | All collected demonstrations failed; inspect `failures/` and improve the expert/scene before collecting again. |
| `ERROR` with no SR | Read `run.json` and `episodes.json`. Execution errors are not a valid zero-SR evaluation. |

## Verification commands

```bash
# No GPU needed for these container commands:
docker run --rm so101-course:0.1 test
# GPU acceptance, after the build and fetch:
./scripts/acceptance.sh
```

`acceptance.sh` uses new output directories, checks scene/reset, requires the
expert's 20/20 gate, exports separate-seed demonstrations, reads the dataset,
evaluates SmolVLA on all 20 seeds, and audits both runs. Before release, repeat the
README on a fresh machine or isolated Docker daemon with an empty image store and
a fresh checkout/data directory. `--no-cache` rebuilds layers; it alone does not
prove the base image was downloaded into an empty store. Record the resulting
logs and memory measurements in the validation report.
