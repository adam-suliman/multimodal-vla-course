# Seminar 07 — Sources, adaptations, and licenses

## Source revisions

The notebook downloads the [source/data archive](assets/seminar07_bundle.zip)
from the course repository and verifies its SHA-256:
`977818a7bc22a498b0a1ad933059657838d90eed348a2f6639c46424254cdd69`.
Pinned revisions are also recorded in [source_revisions.json](assets/source_revisions.json).

| Component | Source | Commit |
|---|---|---|
| Cosmos Policy | [NVlabs/cosmos-policy](https://github.com/NVlabs/cosmos-policy) | `18a2accadf4e7a3531e56754102af5a24d2316da` |
| Unified Video Action | [ShuangLI59/unified_video_action](https://github.com/ShuangLI59/unified_video_action) | `fe3dec81af565fbb2d686072722f12340be31293` |
| PyTorch DiT and Wan VAE infrastructure | [iitimii/SmolWAM](https://github.com/iitimii/SmolWAM/tree/dde1e5bfe35c439499343c6d70e9cd32c96d0b36/smolwam/nn) | `dde1e5bfe35c439499343c6d70e9cd32c96d0b36` |

**Cosmos:** original latent injection, action/value readout, preprocessing,
denormalization, rectified-flow scaling, and 2ab sampler functions were extracted
without algorithm changes. Sampler imports were rewritten, the distributed config
guard was replaced with an identity function, and logging uses standard Python logging.

The PyTorch adaptation retains all 28 DiT blocks and 1,956,413,440 parameters,
original weight names, spatial RoPE scaling, per-frame AdaLN, and 512 text positions.
It uses FP32 and PyTorch SDPA on one GPU instead of NVIDIA's distributed runtime,
Transformer Engine, and BF16. Numerical equivalence with that native runtime has
not been measured. Loading excludes training counters and Transformer Engine
`_extra_state` records, not learned parameters. No SmolWAM checkpoint or action
head is used.

The wrapper packs 33 raw frames into nine latent slots. Camera decoding restores
nonvisual slots and decodes the causal prefix, selecting raw frames 21 and 25
according to the upstream mapping.

**UVA:** model, policy, VAE, preprocessing, normalizer, and CPU PushT modules come
from the pinned repository. Training runners and unrelated robot environments
are omitted. The loader reconstructs training-config arguments and skips separate
base VAE/MAR downloads because those weights are included in the checkpoint.
The complete `state_dicts['ema_model']` is loaded strictly, with its normalizer.

Inference uses one refinement iteration, 100 diffusion steps per head, cfg=1,
temperature=0.95, and latent scale 0.2325. Four images are decoded sequentially.
The original policy-mode early return skips the video head. Equality of action
outputs is tested only for one refinement with matched randomness.

## Weights

Weights are downloaded from public upstream releases and are not stored in the
course repository. Total download size is **5,731,048,061 bytes**.
[weights_manifest.json](assets/weights_manifest.json) records sizes and SHA-256 hashes.

| File | Release | Revision / ID |
|---|---|---|
| `Cosmos-Policy-LIBERO-Predict2-2B.pt` | [NVIDIA Hugging Face](https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B) | `cb689ec0e3347c13667d70a78a3447388f5c3bb8` |
| `Wan2.1_VAE.pth` | [Wan-AI Hugging Face](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | `37ec512624d61f7aa208f7ea8140a131f93afc9a` |
| `uva_pusht.ckpt` | [Authors' Google Drive](https://drive.google.com/file/d/1OduHcxfc2hqUYSccMQNf9g-vAt-q2UhF/view) | `1OduHcxfc2hqUYSccMQNf9g-vAt-q2UhF` |

Hugging Face hashes were checked against LFS hashes. UVA's hash was computed from
the downloaded release; it is not an author-published checksum. All three files
are verified before loading, including UVA's pickle/dill checkpoint load.

## Data and experiments

- **Cosmos:** the official `sample_libero_10_observation.pkl` was converted
  losslessly to NPZ. Camera preprocessing follows upstream helpers. Normalization
  statistics and two cached T5 embeddings come from the pinned checkpoint release;
  the exercises use one instruction. No observed future or expert actions are used.
- **UVA:** upstream CPU PushT uses `legacy=True`, `fix_goal=True`, render size 96,
  and seed 100000. Its initial observation is repeated 16 times; history encoding
  selects indices 3, 7, 11, and 15. Predicted images correspond to horizons
  4, 8, 12, and 16. Commands are absolute 2D targets and are not executed.
- **Planning:** three action candidates and three futures per candidate illustrate
  the upstream sequence of conditional tasks. Selection averages raw normalized
  values. This is a simplified teaching search, not a reproduction of the paper's
  rollout-refined ALOHA planner or task-success results.
- **UVA comparison:** both paths reuse one encoded history and reset the seed.
  Command difference is the maximum absolute difference between corresponding
  normalized x/y coordinates. Call counts distinguish action generation, video
  latent generation, and VAE decoding.

Bundled data hashes and experiment settings are listed in
[data_manifest.json](assets/data_manifest.json).

## Figures and licenses

All four illustrations are notebook attachments:

| Attachment | Source | Treatment |
|---|---|---|
| `cosmos_fig1.png` | [Cosmos Policy, Fig. 1, p. 1](https://proceedings.iclr.cc/paper_files/paper/2026/hash/748becc400a57c0e31cfe6a2e7951467-Abstract-Conference.html) | Cropped original figure |
| `cosmos_fig2.png` | Same paper, Fig. 2, p. 4 | Cropped original figure; caption explains the two-camera LIBERO variant |
| `uva_fig2.png` | [UVA, Fig. 2, p. 3](https://www.roboticsproceedings.org/rss21/p074.pdf) | Cropped original figure |
| `cosmos_planning_schematic.png` | Original teaching diagram | Three candidates, three alternative futures each, selection by mean value |

Paper figures retain their authors' rights; the software licenses below are not
licenses for those illustrations. Each paper figure is attributed in the notebook.

Upstream license texts and notices are included here and in the downloadable archive:

- [Cosmos — Apache-2.0](licenses/cosmos).
- [PyTorch infrastructure — Apache-2.0](licenses/turing_adapter) and
  [upstream NOTICE](licenses/turing_adapter_NOTICE).
- [UVA — MIT](licenses/uva).

Weight terms are governed by their model cards and original releases, separately
from the code licenses.
