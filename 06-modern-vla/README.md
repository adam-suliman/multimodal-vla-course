# Seminar 06 — Modern Continuous VLA: SmolVLA, π₀ and π₀.₅

Open [the student notebook](seminar_06_modern_vla_student.ipynb), or
[open it in Google Colab](https://colab.research.google.com/github/wingrune/multimodal-vla-course/blob/main/06-modern-vla/seminar_06_modern_vla_student.ipynb).
It contains eight questions and five short coding TODOs:

- **SmolVLA, questions 1–3:** distinguish expert inputs from flow targets,
  measure cache amortization, and test partial cache refresh after a state change.
- **π₀, questions 4–5:** trace attention dependencies and reconstruct joint
  attention from separately normalized VLM/expert contributions.
- **π₀.₅, questions 6–8:** examine discrete state conditioning and continuous
  actions, AdaRMS initialization, and gradients through an initially zero gate.

An unnumbered explanation connects the π₀ time/action MLP to π₀.₅ adaptive
normalization. The notebook includes original paper figures, formulas, and an
AdaRMS teaching schematic based on LeRobot source code.

## Run in Google Colab

1. Open the notebook in Colab, or download the `.ipynb` and upload it.
2. Select a **T4 GPU** and run the four setup/data/helper cells in a fresh session.
3. Complete the five TODOs, run each experiment, and fill in its **Explanation**.
   Keep numerical results, tables, plots, and answers in the submitted notebook.

The notebook embeds helper code and all figures. The setup downloads a prepared
[SO100 data and source archive](assets/seminar06_bundle.zip) (about 0.8 MB) from
the course repository and verifies its SHA-256 checksum. Valid local copies in
`assets/` beside the notebook are reused. No companion upload, Hugging Face login, physical robot,
simulator, or full π-model checkpoint is required. The first run downloads about
0.9 GB of public pretrained SmolVLA weights plus configuration/tokenizer files.
Internet access is needed for these downloads and package installation.

The setup pins LeRobot 0.4.4, Transformers 4.57.1, and PyAV 15.1.0 while preserving the
installed PyTorch/torchvision pair and scientific packages. SmolVLA runs in
float32 for the controlled experiments. The notebook does not fine-tune the
full policy; the final gate experiment updates one small teaching parameter.

## Data and experiment scope

SmolVLA uses its complete pretrained checkpoint and recorded SO100 top/wrist
images, language, and robot state. The subset contains eight 50-frame windows
from four episodes, at the original 30 Hz. Normalization statistics come from
episodes 0, 1, and 2. Actions retain their absolute joint-target convention.

The partial state-cache refresh in Question 3 is a proposed extension: the
standard pinned SmolVLA implementation rebuilds the complete prefix cache for
each action chunk. The experiment tests equivalence using a full fresh cache
as its reference; it does not implement a faster state-prefill algorithm.

The π₀/π₀.₅ experiments use small random or controlled modules and source-based
mask, suffix, AdaRMS, and residual operations. They explain architecture and
training dependencies; they do not load pretrained π-models or compare rollout
success. The zero gate in Question 8 is deliberately imposed, rather than
assumed to be the standard π₀.₅ initialization.

The complete notebook with reference TODO solutions is validated on a local
CUDA GPU. A hosted Colab T4 session has not been tested.

## Sources

- [SmolVLA paper](https://arxiv.org/abs/2506.01844).
- [π₀ paper](https://arxiv.org/abs/2410.24164).
- [π₀.₅ paper](https://arxiv.org/abs/2504.16054).
- [LeRobot SmolVLA documentation](https://huggingface.co/docs/lerobot/main/en/smolvla).
- [LeRobot notebooks](https://huggingface.co/docs/lerobot/notebooks).
- [Source provenance and licenses](UPSTREAM_PROVENANCE.md).
- [AdaRMS teaching schematic](assets/adarms_flow.png) and [SVG](assets/adarms_flow.svg).
