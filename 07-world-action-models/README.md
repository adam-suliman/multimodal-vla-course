# Seminar 07 — World-Action Models: Cosmos Policy and UVA

Open [the student notebook](seminar_07_world_action_models_student.ipynb), or
[open it in Google Colab](https://colab.research.google.com/github/wingrune/multimodal-vla-course/blob/main/07-world-action-models/seminar_07_world_action_models_student.ipynb).

Five questions follow one story: read robot commands from video-model latents,
keep observations fixed, change commands and inspect predicted futures, choose
commands using predicted values, then explore when video generation can be skipped.

- **Cosmos Policy, questions 1–4:** shared latent slots for observations, actions,
  futures, and values. Three short coding TODOs implement action readout,
  conditioning, and candidate selection.
- **UVA, question 5:** compare actions-only and joint action/video inference,
  measure command differences, and count calls to the output branches.

The notebook includes three original paper figures and a planning schematic.

## Run in Google Colab

1. Open the notebook and select a **T4 GPU**.
2. Run the four collapsed setup cells in a fresh session.
3. Complete the three coding TODOs and run the experiments in order. Fill in the
   short prediction and answer cells; keep outputs and answers in your submission.

Figures and experiment helpers are embedded in the notebook. Setup downloads a
[code and example-data archive](assets/seminar07_bundle.zip) (about 0.37 MB) from
the course repository and verifies its SHA-256 checksum. Valid local copies in
`assets/` beside the notebook are reused; no companion files need uploading. The first run downloads **5.73 GB** of public pretrained weights.
Internet access is needed for package installation and downloads; no Hugging Face
login is required. Rerun the weight-download cell to resume an interrupted download.
Files are checked by SHA-256 before loading.

Both models use complete pretrained checkpoints in FP32 and load sequentially.
The experiments perform inference without training. Setup preserves the installed
PyTorch/torchvision pair.

## Data and experiment scope

Cosmos uses one recorded LIBERO observation with two cameras, robot state, and a
cached language embedding. The experiments predict 16-command chunks and their
endpoint. The small candidate search illustrates model-based planning with three
action candidates and three futures per candidate.

UVA uses an initial CPU PushT observation and predicts 16 two-dimensional target
positions, optionally with four future images. The two inference paths share the
encoded history and random seed and use one refinement iteration.

Commands are not executed. Differences between predicted futures measure model
sensitivity; value rankings do not establish task success. The Cosmos runtime is
a pure-PyTorch adaptation; its numerical agreement with NVIDIA's native runtime
has not been measured.

## Sources

- [Cosmos Policy paper, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/748becc400a57c0e31cfe6a2e7951467-Abstract-Conference.html).
- [Unified Video Action paper, RSS 2025](https://www.roboticsproceedings.org/rss21/p074.html).
- [Source provenance and licenses](UPSTREAM_PROVENANCE.md).
- [Weight checksums](assets/weights_manifest.json),
  [data manifest](assets/data_manifest.json), and
  [source revisions](assets/source_revisions.json).
