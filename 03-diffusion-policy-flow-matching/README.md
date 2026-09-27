# Seminar 03 — Diffusion, Flow Matching, and TinyVLA

Open [the student notebook](seminar_03_diffusion_flow_student.ipynb) and run its
cells in order. It has six exercises on a synthetic ring, followed by three
exercises with a small vision-language-action model on PushT and ALOHA.

## Google Colab

Upload the notebook to Colab or open it from this repository. A GPU runtime is
recommended for the VLA training tasks. The notebook installs MuJoCo if needed
and sets up headless rendering. When uploaded by itself, it clones the pinned
zero2robot source for the simulator environments. The architecture diagram and
both demonstration videos are embedded in the notebook.

The full VLA experiment uses 60 demonstrations per task and 200 training epochs.
Set `VLA_SMOKE = True` in the VLA setup cell for a quick code check; smoke results
are too small for the exercise conclusions. Generated datasets and metrics are
cached in `seminar03_vla_runs/`, and ring results in `seminar03_ring_runs/`.
The sighted VLA run trains its camera CNN jointly with the policy. The dataset
cell saves raw frames for this training; the blind run receives no visual feature.
VLA Task 1 reports PushT results and saves the policy. Task 2 loads the same
policy and evaluates ALOHA without retraining it.

## Local use

Launch Jupyter from this directory or the course repository root. Keep the
notebook beside `zero2robot/`: the notebook imports its PushT and ALOHA
simulators and scripted experts. PyTorch, NumPy, Matplotlib, and MuJoCo are
required; the notebook's setup cells install missing packages except PyTorch
when Colab already supplies it.

The `zero2robot/` folder contains only the upstream files needed to run this
seminar and follow its exercise links. See [provenance](UPSTREAM_PROVENANCE.md)
for the pinned revision and licenses.
