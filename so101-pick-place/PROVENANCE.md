# Sources and numeric conventions

All downloadable artifacts, byte sizes, checksums and revisions are enumerated in
[sources.lock.json](sources.lock.json). Downloading never follows a moving branch.
The policy loader uses local files only and does not run remote model code.

| Component | Source / license |
| --- | --- |
| Simulator | [NVIDIA Isaac Sim 5.1.0](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim), NVIDIA license; Linux amd64 manifest `sha256:93b0f99635ab126fb5b33298d513c11520f119f0ee60ff8414ccef67ea977829` |
| SO-101 follower | [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/a758567c3978dfeefe282ede0500085a48fe8f78/Simulation/SO101), Apache-2.0; URDF and its referenced STL files, plus LICENSE and README |
| Task policy | [Cache-SCA checkpoint](https://huggingface.co/Cache-SCA/smolvla_IsaacLab-SO101_pick_place_baseCaP_100epi_50ep/tree/6c048c0a0574d88566cc83170341e5fd4306dae5), Apache-2.0; model, configuration and both processor states |
| Tokenizer/backbone configuration | [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/tree/7b375e1b73b11138ff12fe22c8f2822d8fe03467), Apache-2.0; no second set of model weights is needed |
| Training convention evidence | [CoRL2026-CSI dataset](https://huggingface.co/datasets/CoRL2026-CSI/IsaacLab-SO101_pick_place_baseCaP_100epi_10fps/tree/07a40b15d4afcfcbc4e54ff6d27774230acb33a2), Apache-2.0 |
| Inference and dataset implementation | [LeRobot v0.4.4](https://github.com/huggingface/lerobot/tree/8fff0fde7c79f23a93d845d1a50e985de01f8b8a), Apache-2.0; installed wheel and all transitive dependencies are hash-locked |

No upstream robot meshes or model weights are committed in this module. The
conversion copy removes only the empty TCP marker link and its fixed joint; it
preserves every visual/collision mesh and all six actuated joints. The recipe
fixes the base, disables self-collision, and requests convex collision
decomposition. Runtime gains, effort limits, material friction and contact offsets
are explicit in the configuration. The removed marker has no geometry and zero inertia; the expert still computes
its tool frame from the unchanged original URDF. Mesh instances are made editable
in the single-robot scene so contact properties apply to each collision shape.
The composed scene overrides the imported `3d_printed` material's diffuse color
with `robot.printed_color` (white by default). Servo materials remain dark. This
visual override does not modify the downloaded URDF, converted USD files, meshes,
collision shapes, or dynamics.

The checkpoint's processor renames `left_wrist` to `camera1`, `top` to `camera2`.
Although its model config lists `camera3`, the dataset has two cameras, and pinned
SmolVLA's `prepare_images` skips missing cameras when `empty_cameras=0`. This module
preserves that behavior; it does not duplicate an image into the third camera.
The model's own preprocessing handles image resizing, language tokenization,
state normalization and action denormalization.

The dataset contains both calibrated and `radian_urdf0` state/action columns. The
six affine transformations in the default config were fitted to the state column
and checked on the entire action column as well. Maximum discrepancy is below
4e-6 calibrated units. Joint names come from the dataset metadata, with six
positions ordered shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll,
gripper. The final gripper transformation is approximately
`45.7472731 * q_rad + 0.07017546`; zero and 30 in the training data correspond to
roughly -0.001534 and 0.654243 radians. This is different from the generic real-arm
LeRobot 0–100 opening convention. Do not replace it with degrees or an assumed
0–100 scale.

The dataset card links to SCRAPE-IsaacLab collection code, which was unavailable
(HTTP 404) during implementation. Numeric agreement with recorded radians and all recorded TCP positions is
verified; exact visual/physical agreement with the training scene is not claimed.
The public URDF uses its documented new-calibration zero convention. Transfer
between these assets/scenes is an experimental limitation, and no real hardware
calibration is supplied.

The Python lock is regenerated from `requirements-policy.in` for Linux x86-64 /
Python 3.11, using PyPI and the official PyTorch CUDA 12.6 index. Hashes verify all
runtime distributions. Source distributions (currently small platform helpers)
are built with separately pinned Python build tools and `build-essential=12.10ubuntu1`
from the [Ubuntu snapshot service](https://snapshot.ubuntu.com/) at
`20251010T000000Z`. Compilation happens inside the same public Isaac Sim base image. The final image
inherits the installed environment layer directly, avoiding a second roughly
22 GB copy. The snapshot-pinned compiler and installer remain in the image. The simulator has its own `requirements-simulator.lock`: official Isaac Sim
5.1.0.0 wheels, including extension caches, with PyTorch 2.7.0 CUDA 12.6 and
NumPy 1.26. It is independent of the policy environment. Both environments use
the public base image's Python 3.11 ABI. The optional local setup uses exactly
these locks in module-local environments and the same public base image.

The wheel runtime installs the full `isaacsim[all,extscache]==5.1.0.0` extra, matching the B1K setup. This includes `isaacsim-robot`, `isaacsim-robot-motion`, and `isaacsim-example`: the robot schema is required by the core Kit application and the examples browser by the URDF importer even when no robot controller from those packages is used. Kit extension-registry downloads are disabled; startup must resolve extensions from the locked wheels.
