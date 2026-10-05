# Seminar 04 — Sequence Models and Action Chunking with ACT

Open [the student notebook](seminar_04_action_chunking_student.ipynb) and run its
cells in order. Reconstruct the action prediction head of a pretrained Action
Chunking Transformer (ACT), compare execution horizons, and implement temporal
ensembling in the simulated ALOHA Transfer Cube task.

The seminar uses inference with pretrained weights. No model training, physical
robot, or demonstration dataset is required.

## Run in Google Colab

1. Open the notebook from this repository in [Google Colab](https://colab.research.google.com/),
   or download the `.ipynb` and upload it to Colab. The notebook is self-contained:
   helper code and figures are embedded, so no companion files need uploading.
2. Select **Runtime → Change runtime type → Runtime Version → 2026.07**
   (Python 3.12.13) and **T4 GPU**. Python 3.10–3.12 is required; the default
   Python 3.13 runtime can fail when installing `labmaze`, a dependency of
   `dm-control`. See [Colab runtime versions](https://research.google.com/colaboratory/runtime-version-faq.html).
3. Run the installation cell first in a fresh runtime. It installs the simulator
   and other dependencies while preserving Colab's PyTorch/torchvision pair.
4. Run the supplied setup cells, then complete the four coding TODOs and the
   interpretation questions. Setup downloads a public checkpoint (~207 MB);
   no Hugging Face token is required.
5. Save your completed notebook with numerical checks, plots/videos, metric
   tables, and short answers. Download `seminar04_runs/` before ending the session
   to retain cached simulation results.

## Compute resources

The notebook is designed for a single Colab T4 GPU runtime. Internet access is
needed to install packages and download the checkpoint. MuJoCo renders offscreen
with EGL; videos appear directly in notebook outputs. No desktop display is
required. CPU execution is possible, but the simulation and inference grid is
slower. Colab GPU availability and session duration depend on the service.

For local use, install Python 3.10–3.12 and a compatible PyTorch/torchvision pair,
then run the notebook's installation cell. A CUDA GPU is recommended.

## References

- [ACT paper: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705).
- [Original ACT implementation](https://github.com/tonyzhaozh/act).
- [Pretrained ALOHA Transfer Cube checkpoint](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human/tree/ba73b2766f1371cdc133ca4efb97eb090d744625).
- [Source provenance and licenses](UPSTREAM_PROVENANCE.md).
