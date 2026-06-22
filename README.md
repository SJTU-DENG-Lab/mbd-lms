# Multi-Block Diffusion Language Models

<p align="center">
  <a href="https://sjtu-deng-lab.github.io/mbd-lms/">
    <img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page">
  </a>
  <a href="https://sjtu-deng-lab.github.io/blogs/mbd/">
    <img src="https://img.shields.io/badge/Blog-Post-blueviolet" alt="Blog">
  </a>
  <a href="paper/MBD-LMs.pdf">
    <img src="https://img.shields.io/badge/Paper-PDF-b91c1c" alt="Paper PDF">
  </a>
  <a href="https://github.com/SJTU-DENG-Lab/mbd-lms">
    <img src="https://img.shields.io/badge/Training%20Code-mbd--lms-0f766e" alt="MBD-LMs Training Code">
  </a>
  <a href="https://github.com/SJTU-DENG-Lab/Diffulex/tree/mbd-lms">
    <img src="https://img.shields.io/badge/Reproduce-Diffulex%20mbd--lms-blue" alt="Reproduction Branch">
  </a>
  <a href="https://github.com/SJTU-DENG-Lab/Diffulex/tree/main">
    <img src="https://img.shields.io/badge/Engine-Diffulex%20main-green" alt="Diffulex Engine">
  </a>
  <a href="#license">
    <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License">
  </a>
</p>

This repository is the **training and method repository** for **Multi-Block Diffusion Language Models (MBD-LMs)**. It defines the paradigm and contains the training-side assets needed to build MBD-LMs:

- Multi-block Teacher Forcing (MultiTF) training code and configs;
- dataset preparation and training setup guidelines;
- multi-node training launch scripts;
- checkpoint conversion utilities;
- the project page and method documentation.

Block Diffusion Language Models (BD-LMs) support KV caching and flexible-length generation, but native BD-LMs usually decode with **Single-Block Diffusion (SingleBD)**: each forward pass refines one noisy block while later blocks wait for the current block to be completed and cached. This creates KV-cache storing bubbles and leaves inter-block parallelism underused.

MBD-LMs target **Multi-Block Diffusion (MultiBD)**, where a bounded running-set of consecutive blocks is decoded concurrently. We introduce **Multi-block Teacher Forcing (MultiTF)** for train-inference alignment and a **Block Buffer** inference mechanism for efficient static-shape execution.

The repository roles are split intentionally. Use the training repository for
model-side work, and use Diffulex for inference and systems work:

| Repository / branch | Role |
|---|---|
| [`SJTU-DENG-Lab/mbd-lms`](https://github.com/SJTU-DENG-Lab/mbd-lms) | Training and method repository: MultiTF, training configs, dataset setup, checkpoint conversion, and paper/project documentation. |
| Diffulex [`mbd-lms`](https://github.com/SJTU-DENG-Lab/Diffulex/tree/mbd-lms) | Experiment reproduction branch for running the reported MBD-LMs inference/evaluation setup. |
| Diffulex [`main`](https://github.com/SJTU-DENG-Lab/Diffulex/tree/main) | Active inference engine branch for runtime development, open-source contributions, and new dLLM decoding algorithms. |

<p align="center">
  <img src="docs/assets/fig1_singlebd_vs_multibd.png" alt="SingleBD vs MultiBD" width="88%">
</p>

---

## Quick Start

### Training and Method Work

Start from this repository when you are working on MultiTF training, data
preparation, or checkpoint conversion:

```bash
git clone https://github.com/SJTU-DENG-Lab/mbd-lms.git
cd mbd-lms
```

Then follow the guides:

1. [Training Setup](docs/guidelines/training_setup.md)
2. [Start Training](docs/guidelines/train.md)
3. [Inference Setup](docs/guidelines/inference_setup.md)
4. [Run Benchmarks](docs/guidelines/benchmark.md)

### Experiment Reproduction

For the reported MBD-LMs inference/evaluation setup, use the Diffulex
`mbd-lms` branch:

```bash
git clone https://github.com/SJTU-DENG-Lab/Diffulex.git
cd Diffulex
git checkout mbd-lms
```

**Reproducibility note.** The scores and throughput numbers reported in the
paper were produced with the Diffulex `mbd-lms` branch. Use this branch to
reproduce the paper tables, including the throughput table. The actively
optimized Diffulex `main` branch may produce different latency, TPS, or
benchmark numbers because the runtime has continued to change after the paper
experiments.

### Engine Development

For new runtime features, open-source contributions, and new dLLM decoding
algorithms, use Diffulex `main`:

```bash
git checkout main
```

---

## Highlights

- **Multi-Block Diffusion formulation.**  
  We formulate MBD-LMs as BD-LMs that recover a bounded running-set of consecutive blocks conditioned on a clean cached prefix.

- **MultiTF post-training.**  
  MultiTF trains BD-LMs on bounded noisy block groups with heterogeneous slot-wise mask ratios, matching practical MultiBD inference states.

- **Block Buffer inference.**  
  A fixed-size Block Buffer preserves prefix-cache reuse, enables decode-store overlap, and keeps tensor shapes static for CUDA Graph-friendly execution.

- **Improved parallelism and throughput.**  
  On math and code benchmarks, MBD-LLaDA2-Mini increases average TPF from **3.47** to **6.19** while improving average accuracy from **79.95%** to **81.03%**. With DMax, MBD-LLaDA2-Mini-DMax reaches **9.34** average TPF.

---

## Method

### Multi-block Teacher Forcing

MultiTF post-trains BD-LMs with bounded **noise-groups** that approximate the running-set states seen during MultiBD inference. It combines systematic and random group layouts, applies a chain-uniform noise scheduler to create heterogeneous slot-wise mask ratios, and uses a Group-Aware Dual-Stream Mask to control visibility between noisy and clean blocks.

<p align="center">
  <img src="docs/assets/fig4_multitf_overview.png" alt="MultiTF overview" width="92%">
</p>

### Block Buffer Inference

Naive MultiBD has a dynamic running-set whose length changes during decoding, which is inefficient for static-shape execution. The Block Buffer mechanism instead maintains a fixed number of physical block slots. Future blocks enter by activating dummy slots, and completed blocks are committed into the KV cache.

Each slot follows the transition:

```text
dummy -> active -> to-cache -> in-cache
```

This design exposes inter-block parallelism while preserving the serving advantages of BD-LMs.

<p align="center">
  <img src="docs/assets/fig5_block_buffer.png" alt="Block Buffer inference" width="90%">
</p>

---

## Results

We evaluate on **GSM8K**, **MATH500**, **MBPP+**, and **HumanEval+**. Accuracy is exact match for math and pass@1 for code. TPF denotes Tokens Per Forward pass, and AUP summarizes the accuracy-parallelism trade-off.

### Main Results

| Base Model | Native Avg. Acc. | Native Avg. TPF | MBD Avg. Acc. | MBD Avg. TPF | AUP: Native -> MBD |
|---|---:|---:|---:|---:|---:|
| LLaDA2-Mini-DMax | 79.59 | 6.35 | 78.57 | 9.34 | 459.54 -> 661.28 |
| LLaDA2-Mini | 79.95 | 3.47 | 81.03 | 6.19 | 247.41 -> 449.18 |
| SDAR-8B-Chat-b32 | 69.00 | 2.54 | 69.74 | 4.46 | 141.64 -> 210.42 |
| SDAR-8B-Chat-b4 | 75.59 | 1.25 | 75.27 | 2.42 | 85.46 -> 148.65 |

MBD-LMs consistently improve decoding parallelism over native SingleBD. MultiTF also recovers or improves quality compared with training-free MultiBD in most settings, indicating that train-inference alignment is important for reliable MultiBD.

### Throughput

Throughput is measured for single-sample decoding on two H100 GPUs with tensor
parallelism degree 2. These are paper-reproduction numbers from the Diffulex
`mbd-lms` branch. For exact reproduction, use that branch rather than the
actively optimized Diffulex `main` branch; newer engine versions may differ
from the reported table.

| Model | Avg. TPF | Step Latency | Avg. TPS |
|---|---:|---:|---:|
| LLaDA2-Mini | 3.47 | 7.07 ms | 517.16 |
| MBD-LLaDA2-Mini | 6.19 | 8.78 ms | 745.92 |
| LLaDA2-Mini-DMax | 6.35 | 9.02 ms | 779.49 |
| MBD-LLaDA2-Mini-DMax | 9.34 | 11.20 ms | 926.67 |

The larger Block Buffer increases per-step latency, but the gain in useful tokens committed per forward pass leads to higher realized throughput.

---

## License

This repository is released under the [MIT License](LICENSE).
