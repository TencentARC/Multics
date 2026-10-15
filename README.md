<div align="center">

# *Multics*: Generative Multi-Player World Model with Hierarchical Communication

[Kaiyi Huang](https://karine-huang.github.io/)<sup>1</sup>,
[Wangbo Yu](https://drexubery.github.io/)<sup>2†</sup>,
[Yukun Huang](https://yukun-huang.github.io/)<sup>2</sup>,
[Jiwen Yu](https://yujiwen.github.io/)<sup>1</sup>,
[Kunhao Liu](https://kunhao-liu.github.io/)<sup>2</sup>,
<br>
[Donglin Yang](https://linydthu.github.io/)<sup>1</sup>,
[Wenbo Hu](https://wbhu.github.io/)<sup>2</sup>,
[Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ)<sup>2</sup>,
[Xihui Liu](https://xh-liu.github.io/)<sup>1†</sup>

<sup>1</sup> The University of Hong Kong &emsp; <sup>2</sup> ARC Lab, Tencent IEG

<sup>†</sup> Corresponding authors.

*Multics* enables scalable multi-player video generation through hierarchical communication.

[![arXiv](https://img.shields.io/badge/arXiv-2610.17201-B31B1B)](https://arxiv.org/pdf/2610.17201)
[![Project Page](https://img.shields.io/badge/Project-Page-76B900)](https://karine-huang.github.io/Multics/)
[![Hugging Face Models](https://img.shields.io/badge/Hugging%20Face-Models-FFD21E)](https://huggingface.co/TencentARC/Multics)
[![Docs](https://img.shields.io/badge/Docs-Get__Started-2563EB)](#installation)
[![License](https://img.shields.io/badge/Code__License-Apache__2.0-76B900)](LICENSE)

</div>

## 🎬 Demo video

https://github.com/user-attachments/assets/ba220953-b57f-47a5-bbb8-9b440b55ed11

## 💡 Overview

**Multics** organizes communication across player, group, and global levels, using dynamic grouping based on recent camera trajectories. This hierarchy improves cross-player consistency, action and camera controllability, and robustness to interference while reducing attention computation. Shared group-token parameters allow a model trained with 4 players to generalize to 6, 8, and 10 players without retraining.

## 📰 News

- 🔥 \[2026/10\] [Paper](https://arxiv.org/pdf/2610.17201), training and inference code released.

## 🚀 Get started

### Installation

Requires Linux, Python 3.12, an NVIDIA GPU, the CUDA 12.8 toolkit, and a C++ compiler.

From the repository root, install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
source .venv/bin/activate
```

### Model Weights

Download the complete inference bundle from [TencentARC/Multics](https://huggingface.co/TencentARC/Multics), including the 4-player checkpoint, VAE, text encoder, and tokenizer:

```bash
hf download TencentARC/Multics --local-dir models
```

### Inference

#### 1. 4-player generation

Generate example `4_01` from its first frames, captions, and tick logs:

```bash
python inference.py --config configs/inference_demo.yaml \
  --checkpoint models/multics/4players/checkpoints/model.ckpt \
  --hparams models/multics/4players/hparams.yaml \
  --case 1
```

Outputs are saved to `outputs/inference/4players/`. Use `--cases 1 2 3 4 5` to run all five examples.

#### 2. Scalability

The same 4-player checkpoint supports 6-, 8-, and 10-player generation without retraining:

```bash
python inference.py --config configs/inference_demo.yaml \
  --checkpoint models/multics/4players/checkpoints/model.ckpt \
  --hparams models/multics/4players/hparams.yaml \
  --num_players 8 --cases 1 2
```

| Players | Examples | Arguments |
| --- | --- | --- |
| 6 | `6_01`–`6_03` | `--num_players 6 --cases 1 2 3` |
| 8 | `8_01`–`8_02` | `--num_players 8 --cases 1 2` |
| 10 | `10_01`–`10_02` | `--num_players 10 --cases 1 2` |

Outputs are saved to `outputs/inference/4infer6/`, `4infer8/`, or `4infer10/`. See [example inputs](asset/examples/README.md) for the data format and previews.

#### GPU memory

Single-GPU inference with `configs/inference_demo.yaml`: BF16, 640 × 352 per
player, 165 frames, and 50 sampling steps (PyTorch 2.8.0, CUDA 12.8).

GPU memory grows approximately linearly with player count over the tested 4–10 player range:

| Players | Peak allocated (GiB) | Peak reserved (GiB) |
| --- | ---: | ---: |
| 4 | 59.9 | 74.9 |
| 6 | 75.3 | 97.4 |
| 8 | 90.6 | 119.0 |
| 10 | 106.1 | 139.2 |

Measured PyTorch peaks for reference. Allocated measures tensor memory; reserved also includes allocator cache.

### Training

To train the 4-player model, prepare your dataset in the [precomputed input format](asset/examples/README.md#precomputed-inputs) and download the original Wan2.2 backbone for initialization:

```bash
hf download Wan-AI/Wan2.2-TI2V-5B \
  --include 'diffusion_pytorch_model*.safetensors' \
  --local-dir models/Wan-AI/Wan2.2-TI2V-5B

uv sync --locked --extra distributed

NPROC_PER_NODE=8 bash scripts/train.sh \
  --config configs/train.yaml \
  --dataset_path /path/to/data \
  --metadata_file_name train.csv
```

Set `NPROC_PER_NODE` to your GPU count. Training defaults are provided in [configs/train.yaml](configs/train.yaml).

## 🤗 Acknowledgements

This project is built on [Wan](https://github.com/Wan-Video/Wan2.2) and [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio). We thank the authors for their open-source contributions.

We also thank the creators of [OpenCS2](https://huggingface.co/datasets/blanchon/opencs2_dataset) for making their dataset publicly available.

The code is released under the [Apache-2.0 license](LICENSE). Model weights and example content retain their upstream terms.

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@misc{huang2026multics,
  title  = {Multics: Generative Multi-Player World Model with Hierarchical Communication},
  author = {Huang, Kaiyi and Yu, Wangbo and Huang, Yukun and Yu, Jiwen and Liu, Kunhao and Yang, Donglin and Hu, Wenbo and Shan, Ying and Liu, Xihui},
  year   = {2026},
  eprint = {2610.17201},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2610.17201}
}
```
