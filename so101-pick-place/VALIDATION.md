# Validation record

Updated: 2026-11-09 (Europe/Moscow). The local wheel runtime and physical task
checks pass on **RTX 3060, GPU 0, 12,288 MiB, driver 535.230.02**. The driver was
not changed. Both installation paths and the clean local checkout passed. A published GitHub
checkout on an empty Docker image store remains unverified.

## White robot appearance check

The printed plastic now defaults to white; servo housings remain dark. The
composed scene overrides only the imported plastic shader, preserving the
checksum-verified asset, collision shapes and dynamics. On 2026-11-09,
`check-scene --output runs/white-scene-check` exited successfully on GPU 0 with
unchanged driver 535.230.02. Both views were visually inspected. Repeated seeded
resets, joint motion, image updates and control timing also passed. See
[scene check](validation/white-scene.json), [top view](validation/white-scene-top.png)
and [wrist view](validation/white-scene-wrist.png).

The local Docker image was updated by copying the changed module into the
previously validated image `12966d8fe4e6a97e23c26bb4409cb9df36c1fa7c61c092bf553e48983ad15116`;
its installed dependencies were reused. The repository Dockerfile still builds
from the pinned public NVIDIA base. This check does not rerun the 20-seed policy
evaluation; the baseline task results below use yellow plastic.

## White robot demonstration videos

On 2026-11-09 both bundled MP4 files were replaced with newly simulated white-robot
episodes. Both use demonstration seed 10000 and the same initial cube position.
Each video shows the top camera on the left and wrist camera on the right at
512×256, 10 fps. Every encoded frame was decoded successfully, the white appearance
was visually checked, and both physics-trace audits passed.

| Policy | Result | Video and evidence |
| --- | --- | --- |
| IK expert | Success in 18.5 s; 185 frames; bilateral contact and 5.88 cm lift | [Video](validation/expert-pilot.mp4), [manifest](validation/white-expert-video.json), [audit](validation/white-expert-video-audit.json) |
| SmolVLA | No success within 30 s; 300 frames; no required lift; no execution error | [Video](validation/smolvla-example.mp4), [manifest](validation/white-smolvla-video.json), [audit](validation/white-smolvla-video-audit.json) |

[Video provenance](validation/white-videos.json) includes commands, source runs,
image identity, checksums and decoding checks. These are individual demonstration
episodes, not a new 20-seed evaluation. The old yellow-robot pilot manifest and
benchmark records below remain historical evidence; the replaced MP4 files now
correspond only to the white-robot manifests above.

## Measured task results

These historical task results use the original yellow printed-plastic material.
The current default is white (`robot.printed_color = [1, 1, 1]`). The historical
results remain associated with their recorded code/configuration hashes; they
are not an evaluation of the white robot's camera observations.

| Check | Result | Evidence |
| --- | --- | --- |
| Expert, all 20 fixed seeds | **20/20, SR 100%**; bilateral finger contact in every episode | [Manifest](validation/expert-evaluation.json), [independent trace replay](validation/expert-audit.json) |
| Task-specific SmolVLA, same 20 seeds | **0/20, SR 0%**; no execution errors or excluded episodes | [Manifest](validation/smolvla-evaluation.json), [independent trace replay](validation/smolvla-audit.json) |
| Simultaneous simulator and policy on GPU 0 | **3,348 MiB / 12,288 MiB** sampled peak | `gpu_memory` in the SmolVLA manifest |
| Expert-only device memory | 2,165 MiB sampled peak | `gpu_memory` in the expert manifest |
| Physical pilot, demonstration seed 10000 | Success in 18.5 simulated seconds, 5.88 cm lift, bilateral contacts, release and stable placement | [Manifest](validation/expert-pilot.json) |
| Pilot LeRobot export/read-back | 185 frames at 10 Hz; every frame loaded | [Read-back](validation/physical-dataset-pilot.json) |
| Independent demonstration seeds 10001 and 10002 | 2/2 physical successes; trace replay passed | [Manifest](validation/demonstrations.json), [trace replay](validation/demonstration-audit.json) |
| Multi-episode physical LeRobot dataset | 2 episodes, 371 frames at 10 Hz; every image/state/action/timestamp and episode boundary read back | [Read-back](validation/physical-dataset.json) |

All 20 evaluation seeds were retained. The expert was developed using a separate
demonstration seed. The evaluator ran at 120 Hz with the unchanged 2 cm lift,
full-target containment, released-gripper and one-second stability criteria.
Every saved physics sample was replayed independently by `audit`.

None of the SmolVLA episodes achieved the required lift or bilateral finger
contact. This is a measured result for this scene and checkpoint, with correct
execution, rather than an installation error or a substituted denominator. No
fine-tuning was performed. The recorded configuration, source revisions, code
fingerprint, dependency locks and package versions are in both manifests.

GPU usage was sampled every 0.5 seconds with `nvidia-smi -i 0`, including policy
initialization and inference. These are sampled total-device peaks, not claims
about every transient allocation. The simulator and policy ran concurrently in
separate Python environments. The 20-seed runs and their per-episode videos,
lossless frames and 120 Hz traces remain in `runs/wheels-expert-eval/` and
`runs/wheels-smolvla-eval/`; generated run archives are intentionally Git-ignored.

## Installation, rendering and software checks

- The complete `scripts/setup-local.sh` command finished successfully. Its own
  hash-locked Python 3.11.13 environments contain **140 simulator packages** and
  **118 policy packages**, with no incompatible dependencies. See
  [installation record](validation/runtime-installation.json) and
  [runtime doctor](validation/doctor-wheels.json).
- `check-scene` exited cleanly. Both RGB views were inspected; the top view contains
  the complete cube and target. A–B–A resets matched. Drives held the home pose and
  followed a shoulder target, and the image changed with joint motion. A control
  step advanced 0.1000000052 seconds; capture did not advance physics. See
  [scene checks](validation/scene-wheels.json), [top image](validation/scene-top.png)
  and [wrist image](validation/scene-wrist.png).
- **32 tests passed** in the actual policy container environment. They include
  negative success-metric cases, observation filtering, cache corruption detection,
  fixed-denominator reporting, real LeRobot writer/loader read-back, episode
  boundaries, MP4 encode/decode and propagation of simulator process failures.
  The small synthetic dataset fixture is separate from the physical exports above.
- All public robot, checkpoint and tokenizer/configuration files passed pinned
  size and SHA-256 checks. URDF conversion produced six revolute joints, seven
  rigid bodies, 34 meshes and 17 collision shapes. The empty TCP marker is removed
  only from the physics import copy.
- Numeric calibration matches both state and action columns across **34,264**
  public training frames within 4e-6 calibrated units. Public-URDF FK matches the
  recorded TCP positions within 3.4e-8 m. See
  [calibration record](validation/calibration.json).
- All fixed-seed expert trajectories solved with a maximum requested joint step
  of 0.099929 radians at 10 Hz and a maximum duration of 23.6 seconds. See
  [kinematics](validation/kinematics.json). Physical execution is established by
  the separate 20/20 result above.
- The real checkpoint loaded strictly with its saved processors. A separate
  51-action diagnostic crossed an action-chunk boundary and reproduced the first
  action after same-seed reset within 1e-6 radians. See
  [policy-process check](validation/policy-process.json) and
  [installed policy smoke check](validation/policy-wheels.json).
- Python, shell, Ruff and notebook schema/code-cell checks passed. The notebook
  itself was not executed; it calls the tested CLI.

## Existing-driver installation and remaining release checks

Read-only inspection of `b1k-driver-test` found a successful October 6 Isaac 5.1
RGB-D/GPU-physics diagnostic on this same 535.230.02 driver. `b1k_rl_guys` installed
`isaacsim[all,extscache]` in Python 3.11 with PyTorch CUDA 12.6. This module reproduces
that installation method with its own complete locks and a public NVIDIA image;
no B1K environment, private image, source mount or developer asset is required.

The normal NVIDIA Container Toolkit hook still fails on this host with an NVML
unknown-device error before Python starts. A different GPU is unhealthy, while
GPU 0 works. The explicit `SO101_GPU_MODE=direct` fallback successfully uses GPU 0
device nodes and read-only mounts of the existing host driver libraries through
`runc`. Isaac's renderer driver-verification setting was not disabled. No host driver, GPU device state,
Docker daemon settings or other users' images were changed. The original hook
failure is retained in [container-gpu.txt](validation/container-gpu.txt).

An early renderer-less probe faulted during Kit's graceful Python teardown.
Commands now use Kit's default fast shutdown in a child process; the parent
requires both an explicit completion record and a clean exit, and produces videos
afterwards. Errors cannot silently become successful CLI exits.

The main `so101-course:0.1` image built successfully from the public base and
hash-verified wheels downloaded into an empty build wheel directory. Both
installed environments passed dependency checking. The final image reuses their
layer directly, avoiding another roughly 22 GB copy; the snapshot-pinned compiler
remains in the image. See [image provenance](validation/docker-image.json).

A clean **local Git clone** of the unpublished module started with no project data,
cache or wheel environments. Following the README, it built the image, passed
`doctor`, downloaded and verified every asset and weight, converted the URDF,
checked all 20 IK trajectories, checked both cameras/reset/joint motion, passed
all 32 tests, and ran the real policy on GPU 0. A separate physical expert pilot
on seed 10003 also succeeded in that image. Its packaged Python code and scene
configuration hashes exactly match those used in the 20-seed evaluations.
See [clean-clone record](validation/clean-clone.json),
[scene checks](validation/clean-scene.json),
[container tests](validation/container-tests.txt), and
[Docker physical pilot](validation/docker-physical-pilot.json), and its
[trace replay](validation/docker-physical-audit.json).

Reconversion reproduced the contents of all five USD layers exactly. One layer's
binary crate encoding differed, while its exported USDA content was identical;
raw file hashes continue to protect each individual local conversion cache.
See [conversion comparison](validation/conversion-reproducibility.json).

The remaining release check is to repeat the **published** README on an isolated
Docker daemon with an empty image store. The shared daemon was not cleared;
public base/installer layers and the newly built environment layer were reused
when building from the clean clone. This is explicitly recorded rather than
represented as a test of an empty daemon or a published GitHub revision.

`scripts/acceptance.sh` remains the repeatable full protocol for a new installation.
