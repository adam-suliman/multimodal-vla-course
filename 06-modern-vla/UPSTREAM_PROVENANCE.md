# Sources and licenses

## Original paper figures and formulas

Figures are reproduced unchanged, with paper links beside each notebook figure.
They are embedded as Markdown attachments in the student notebook.

| Model | Paper | Figure file and original source |
|---|---|---|
| SmolVLA | Shukor et al., arXiv:2506.01844v1 | Figure 1, [SmolVLA.png](https://arxiv.org/html/2506.01844v1/SmolVLA.png) |
| π₀ | Black et al., arXiv:2410.24164v1 | Figure 3, [overview.png](https://arxiv.org/html/2410.24164v1/overview.png) |
| π₀.₅ | Physical Intelligence et al., arXiv:2504.16054v1 | Figure 3, [Figure_3.png](https://arxiv.org/html/2504.16054v1/Figure_3.png) |

The figures remain copyrighted by their original authors. They are educational
excerpts; the LeRobot software license does not apply to them. No open license
for these paper figures is asserted.

Formula sources:

- [SmolVLA §3.1 flow objective](https://arxiv.org/html/2506.01844v1#S3.Ex1).
- [π₀ §IV Euler update](https://arxiv.org/html/2410.24164v1#S4.Ex2).
- [π₀.₅ Eq. (1), hybrid loss](https://arxiv.org/html/2504.16054v1#S4.EGx1).
- [π₀.₅ Appendix A-E, time MLP and adaptive RMSNorm](https://arxiv.org/html/2504.16054v1#A1.SS5.p1).

The SmolVLA notebook states the time convention from the executable LeRobot
source separately from the paper's symbolic objective. The scale/shift/gate
and residual formulas are attributed to source code, not a nonexistent numbered
AdaRMS equation in the π₀.₅ paper.

## LeRobot source excerpts

LeRobot 0.4.4 corresponds to commit
`8fff0fde7c79f23a93d845d1a50e985de01f8b8a`.
The downloaded data package contains these unmodified upstream files in `sources/`:

- `modeling_smolvla.py`, `configuration_smolvla.py`, `smolvlm_with_expert.py`.
- `modeling_pi0.py`, `modeling_pi05.py`, `processor_pi05.py`.

[Repository at this commit](https://github.com/huggingface/lerobot/tree/8fff0fde7c79f23a93d845d1a50e985de01f8b8a).
SmolVLA uses the released package with the actual pretrained policy.
The small suffix modules execute the original `embed_suffix` method bodies;
the small two-stream block follows the original projection and mask rules.
AST extraction avoids importing or allocating complete π-model backbones.

`pi_gemma.py` comes from commit
`b9cb121cb7d4e3e26ec5c906d08dda68d151ba52`, which exposes a standalone
`PiGemmaRMSNorm` implementation compatible with the small CPU experiment.
The module executes the original `PiGemmaRMSNorm` and `_gated_residual` bodies.
This newer implementation is explicitly cited separately from LeRobot 0.4.4;
its per-token RTC feature is not used in the exercise.

[Standalone AdaRMS source](https://github.com/huggingface/lerobot/blob/b9cb121cb7d4e3e26ec5c906d08dda68d151ba52/src/lerobot/policies/pi_gemma.py).
Upstream copyright notices are retained. These software files use Apache-2.0.
The small one-head joining block is seminar-specific code based on the source
operations, not an exact reproduction of the full multilayer π₀ forward.

## Pretrained weights and tokenizer

- `lerobot/smolvla_base`, revision
  `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`.
- `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, revision
  `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.

The policy checkpoint already contains VLM parameters. Only the backbone's
configuration and tokenizer/processor files are fetched separately; its model
weights are not downloaded a second time. All policy weights are loaded and
checked for missing/unexpected keys. The experiment uses float32.

## Recorded data

Dataset: [lerobot/svla_so100_pickplace](https://huggingface.co/datasets/lerobot/svla_so100_pickplace),
revision `728583b5eaf9e739a7f119e2def466fa1d552402`.
The dataset card specifies Apache-2.0.

The downloaded subset selects episodes 0,1,2 for train and 49 for a held-out partition.
Each contributes two windows beginning at frames 30 and 120 with 50 consecutive
recorded actions. No temporal resampling or action reconstruction is performed.
Train statistics use all recorded frames of episodes 0,1,2. The released notebook
uses inference and small-module experiments, without a held-out loss evaluation.

Top/wrist video timestamps are decoded with PyAV and checked within one frame
of the requested timestamp. Images keep their 640×480 resolution and are saved
as JPEG at quality 90. Policy preprocessing handles resize/padding and masking
of the absent third camera. Gripper values retain the native dataset scale;
no universal conversion to physical aperture is assumed.

The archive’s `so100_subset/manifest.json` contains source file SHA-256 hashes,
window membership, requested/decoded timestamps, statistics and channel names.
The notebook verifies the downloaded archive SHA-256 hash before extraction.

## Related educational materials

The [official SmolVLA notebook](https://github.com/huggingface/notebooks/blob/main/lerobot/training-smolvla.ipynb)
and [OpenPI examples](https://github.com/Physical-Intelligence/openpi/tree/main/examples)
are linked as further reading. The seminar's questions, joining block, plots and
cache probes are original; they do not copy Zero2Robot exercise text or code.

## Additional teaching schematic

`assets/adarms_flow.png` and `assets/adarms_flow.svg` are original teaching
diagrams created for the seminar, based on the pinned
`PiGemmaRMSNorm` and `_gated_residual` implementation cited in the flow-time explanation.
They are not figures from the π₀.₅ paper. The notebook embeds the PNG directly
in the flow-time explanation Markdown attachment, alongside the three original paper figures.

The Apache-2.0 license text for LeRobot source excerpts and the dataset is included
in [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt). Original paper figures remain
subject to their authors’ copyright.
