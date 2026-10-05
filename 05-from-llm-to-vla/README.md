# Seminar 05 — From LLM to VLA

Open [the student notebook](seminar_05_from_llm_to_vla_student.ipynb). It contains four coding TODOs and interpretation questions:

- Audit target leakage under causal and bidirectional attention.
- Check the input and attention changes used for parallel decoding in OpenVLA-OFT.
- Compare reconstruction errors for OpenVLA scalar quantization with 32, 64 and 256 edges.
- Compare OpenVLA 256 and FAST+ on identical recorded action chunks.

Scalar encode/decode, data loading, normalization and plotting are supplied. The notebook embeds original OpenVLA, OFT and FAST figures and links to pinned source code.

## Run in Google Colab

1. Open the notebook in [Google Colab](https://colab.research.google.com/) or download and upload the `.ipynb`. Helper code and figures are included; the data archives are downloaded by the setup cell, so no companion files need uploading.
2. Select a CPU runtime. Python 3.10–3.13 is supported by the setup; local validation used Python 3.13.
3. Run the environment installation cell first, then the supplied data and helper cells. Setup preserves installed scientific libraries and the PyTorch/torchvision pair. If you ran an older version that replaced NumPy, restart the session before rerunning the updated notebook. Internet access is needed for package installation and the initial archive download (about 1 MB).
4. Complete the four TODOs and the `Question` prompts. Run the checks and retain the tables and plots in your submitted notebook.
5. Download generated CSVs and figures from the printed runtime directory (`/content/seminar05` in Colab) if you want separate copies.

The setup downloads a real BridgeData V2 subset and the pinned FAST+ tokenizer from this repository, verifies SHA-256 hashes and reuses valid cached files. It does not require robot hardware, model training, GPU memory or full OpenVLA weights.

## Download files

- [Prepared BridgeData V2 subset](assets/bridge_subset.zip).
- [FAST+ tokenizer](assets/fast_tokenizer.zip).

The notebook contains links and a short Python download cell. For local execution, keep these archives in `assets/` beside the notebook to skip downloading. If download fails, its output identifies the URL and error.

## Data and interpretation

The subset contains 12 train and 8 test episodes from the official BridgeData V2 1.0.0 release. Movement channels use reached-state differences; normalization statistics come only from train. Sampling stays at 5 Hz.

Both methods receive the same normalized actions. Tables separate clipping from encode/decode error. Plots show errors in mm/step, deg/step and gripper openness so small differences remain visible. Selected detail examples are identified; summary metrics use all test windows.

The attention block uses teaching substitutes for VLM embeddings and fixed random weights. Dependency checks do not measure policy quality. Reconstructing recorded actions and counting tokens do not measure trained-policy success or inference latency.

The required route and student checks with reference solutions passed locally on CPU. A hosted Colab session has not been tested.

## Sources

- [OpenVLA project and paper](https://openvla.github.io/).
- [OpenVLA-OFT project and paper](https://openvla-oft.github.io/).
- [FAST paper](https://www.roboticsproceedings.org/rss21/p012.pdf) and [FAST+ tokenizer](https://huggingface.co/physical-intelligence/fast).
- [BridgeData V2](https://rail-berkeley.github.io/bridgedata/).
- [Source provenance and licenses](UPSTREAM_PROVENANCE.md).
