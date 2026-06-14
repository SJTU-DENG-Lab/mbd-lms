# Multi-Block Diffusion Language Models

<p align="center">
  <a href="https://sjtu-deng-lab.github.io/mbd-lms/"><b>Project Page</b></a> |
  <a href="https://github.com/SJTU-DENG-Lab/mbd-lms"><b>Code</b></a> |
  <b>Paper: TODO</b>
</p>

<p align="center">
  <b>Yijie Jin, Jiajun Xu, Yuxuan Liu, Chenkai Xu, Yi Tu, Jiajun Li, Dandan Tu, Xiaohui Ye, Kai Yu, Pengfei Liu, Zhijie Deng</b>
</p>

<p align="center">
  Shanghai Jiao Tong University &nbsp; | &nbsp; Xi'an Jiao Tong University &nbsp; | &nbsp; Huawei
</p>

## Overview

This repository will host the official implementation of **Multi-Block Diffusion Language Models (MBD-LMs)**.

Block Diffusion Language Models (BD-LMs) support KV caching and flexible-length generation, but native BD-LMs usually perform **Single-Block Diffusion (SingleBD)**: each forward pass refines one noisy block conditioned on a clean cached prefix. This preserves the serving advantages of BD-LMs, but blocks are still processed sequentially.

MBD-LMs target reliable **Multi-Block Diffusion (MultiBD)**. The method decodes a bounded running-set of consecutive blocks concurrently, aligns training with this practical inference regime through **Multi-block Teacher Forcing (MultiTF)**, and serves the model with an optimized **Block Buffer** inference engine.

<p align="center">
  <img src="docs/assets/fig1_singlebd_vs_multibd.png" width="95%" alt="SingleBD versus MultiBD">
</p>

## Highlights

- **Unified MBD-LM formulation.** MBD-LMs view BD-LM inference through a bounded running-set of consecutive blocks, covering SingleBD and practical MultiBD within a unified formulation.
- **MultiTF post-training.** Multi-block Teacher Forcing constructs inference-like bounded noise-groups with heterogeneous slot-wise noise patterns.
- **Optimized MultiBD inference.** The Block Buffer mechanism preserves prefix-cache reuse, keeps physical input shapes static, and enables decode-store overlap.
- **Improved accuracy-parallelism trade-off.** MBD-LLaDA2-Mini increases average TPF from 3.47 to 6.19 and improves average accuracy from 79.95% to 81.03% in the reported experiments. When combined with DMax, MBD-LLaDA2-Mini-DMax reaches an average TPF of 9.34 with only a 1.02 percentage-point average accuracy drop.

## News

- **2026-06:** Initial repository scaffold and project page.

## Project Page

The GitHub Pages project site is placed under [`docs/`](docs/). After enabling GitHub Pages from the `main` branch and `/docs` folder, the page will be available at:

```text
https://sjtu-deng-lab.github.io/mbd-lms/
```

## Installation

The code release is in preparation. Please replace this section with the final environment setup commands when the implementation is public.

```bash
git clone https://github.com/SJTU-DENG-Lab/mbd-lms.git
cd mbd-lms

# TODO: create environment
# conda create -n mbd-lms python=3.10
# conda activate mbd-lms

# TODO: install dependencies
# pip install -r requirements.txt
```

## Quick Start

Inference and evaluation commands will be added after the code release.

```bash
# TODO: download or specify model checkpoints

# TODO: run MultiBD inference
# python ...

# TODO: run benchmark evaluation
# bash ...
```

## Reproducing Experiments

The paper evaluates math reasoning and code generation on GSM8K, MATH500, MBPP+, and HumanEval+. The exact scripts, checkpoints, and configuration files should be added after release.

```bash
# TODO: add reproduction commands for GSM8K / MATH500
# TODO: add reproduction commands for MBPP+ / HumanEval+
# TODO: add throughput measurement commands
```

## Repository Structure

```text
mbd-lms/
├── README.md
├── docs/
│   ├── index.html
│   ├── style.css
│   ├── .nojekyll
│   └── assets/
│       ├── fig1_singlebd_vs_multibd.png
│       ├── fig2_alignment_stats.png
│       ├── fig3_train_inference_paradigms.png
│       ├── fig4_multitf_overview.png
│       ├── fig5_block_buffer.png
│       ├── table1_main_results.png
│       ├── table2_transfer_ablation.png
│       └── table3_throughput.png
└── .gitignore
```

Suggested code folders to add later:

```text
mbd_lms/              # source code
scripts/              # training, inference, evaluation scripts
configs/              # model and decoding configs
requirements.txt      # Python dependencies
tests/                # unit tests
```

## Results

| Model | Avg. Acc | Avg. TPF | Avg. TPS |
|---|---:|---:|---:|
| LLaDA2-Mini | 79.95 | 3.47 | 517.16 |
| MBD-LLaDA2-Mini | 81.03 | 6.19 | 745.92 |
| LLaDA2-Mini-DMax | 79.59 | 6.35 | 779.49 |
| MBD-LLaDA2-Mini-DMax | 78.57 | 9.34 | 926.67 |

The full benchmark tables are included in the project page under [`docs/index.html`](docs/index.html).

## Citation

The current draft does not include an official BibTeX entry. Please add the official citation after the paper is publicly released.

```bibtex
% TODO: add official BibTeX after release
```

## License

TODO: add license information before the public code release.

## Contact

For questions about the paper or repository, please contact:

```text
Zhijie Deng: zhijied@sjtu.edu.cn
```
