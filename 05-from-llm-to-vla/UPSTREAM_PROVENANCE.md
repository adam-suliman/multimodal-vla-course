# Upstream provenance

The student notebook embeds original figures and downloads the prepared BridgeData V2 archive and the official FAST+ processor and vocabulary from `assets/`. It contains adapted teaching helpers and linked source excerpts. Teacher solutions and reference-result files are not part of this release.

## Pinned code and tokenizer

| Source | Revision |
|---|---|
| [openvla/openvla](https://github.com/openvla/openvla/tree/c8f03f48af692657d3060c19588038c7220e9af9) | `c8f03f48af692657d3060c19588038c7220e9af9` |
| [moojink/openvla-oft](https://github.com/moojink/openvla-oft/tree/e4287e94541f459edc4feabc4e181f537cd569a8) | `e4287e94541f459edc4feabc4e181f537cd569a8` |
| [moojink/transformers-openvla-oft](https://github.com/moojink/transformers-openvla-oft/tree/bc339d9ad707454c0c115970db43c260067c61ab) | `bc339d9ad707454c0c115970db43c260067c61ab` |
| [physical-intelligence/fast](https://huggingface.co/physical-intelligence/fast/tree/ec4d7aa71691cac0b8bed6942be45684db2110f4) | `ec4d7aa71691cac0b8bed6942be45684db2110f4` |

OpenVLA-derived helpers and excerpts retain the upstream [MIT notice](licenses/OpenVLA-MIT.txt). OFT excerpts retain the [OFT MIT notice](licenses/OpenVLA-OFT-MIT.txt). The Transformers fork and FAST+ use [Apache 2.0](licenses/Apache-2.0.txt). FAST+'s upstream README, configuration, processor and tokenizer files are preserved inside its downloaded archive.

## BridgeData V2

Attribution: Walke et al., *BridgeData V2: A Dataset for Robot Learning at Scale* (2023), [project](https://rail-berkeley.github.io/bridgedata/). Data: [CC BY 4.0](licenses/CC-BY-4.0.txt).

Release: `bridge_dataset/1.0.0`. [Official source shard](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/bridge_dataset-train.tfrecord-00000-of-01024).

Source shard SHA-256: `04e67b7ba8c66a1990d72f9fb2d0861d1203623c74d9d2c4d4678f928ba4f579`.

The archive contains standardized actions, representative original RGB frames, language annotations and metadata for 20 selected episodes. Its manifest records source paths, original episode IDs, record positions and hashes, train/test membership, channel units and window-selection rules. Episode IDs are interpreted together with their source paths; they are not assumed globally unique.

Preprocessing removes the original first step, binarizes gripper commands as in OpenVLA, relabels movement using consecutive reached states, and removes the final step. Statistics use only train episodes. RGB frames retain their original content; the notebook's small teaching encoder resizes them during feature extraction. Frequency remains 5 Hz.

## Original figures

- [OpenVLA model image](https://openvla.github.io/static/images/openvla_model.jpg), Kim et al. (2024), reproduced unchanged.
- [OFT paper](https://arxiv.org/abs/2502.19645), Kim et al. (2025), Figure 2 on PDF page 3; cropped and rendered at 2× scale.
- [FAST paper](https://www.roboticsproceedings.org/rss21/p012.pdf), Pertsch et al. (2025), Figure 4 on PDF page 5; cropped and rendered at 2× scale.

Figures are embedded as notebook attachments and retain attribution beside each image. They are reproduced from the original sources, not redrawn. The code license notices above apply to code; figure authorship is credited separately.

## Downloaded archives

The setup cell downloads [Bridge examples](assets/bridge_subset.zip) and [FAST+](assets/fast_tokenizer.zip), then verifies these SHA-256 hashes before extraction:

| Archive | SHA-256 |
|---|---|
| Bridge subset | `15a1070c52a44dedab98e2d06a2211cb78b0decac90722f68785771c218558da` |
| FAST+ tokenizer | `99780f6c33949f79a82c4a80a500a30e0632b0ae8d2d5d659ef4cf7570cc99cd` |
