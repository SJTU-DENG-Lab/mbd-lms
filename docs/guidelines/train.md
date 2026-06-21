# Training Guide

## Hardware Requirements

| Aspect | Local Test / Smoke | Production Training |
|--------|--------------------|----------------------|
| GPUs | 8× H100/H200 | 32+× H100/H200 |
| Nodes | 1 | 4+ |
| GPU memory | 80 GB (H100) / 141 GB (H200) | 141 GB (H200 preferred for coverage layouts) |

All reported experiments used 32 GPUs. Wall-clock time per run ranges from **4 to 10 hours**. H200 is preferred when coverage layouts produce large per-GPU sample packs that exceed H100 memory.

## Preparation

### 1. Base Model Weights

Download one of the following base models before training:

| Model | Type | HuggingFace |
|-------|------|-------------|
| DMax-Math-16B | MoE | [Zigeng/DMax-Math-16B](https://huggingface.co/Zigeng/DMax-Math-16B) |
| DMax-Coder-16B | MoE | [Zigeng/DMax-Coder-16B](https://huggingface.co/Zigeng/DMax-Coder-16B) |
| LLaDA2.0-mini | MoE (base) | [inclusionAI/LLaDA2.0-mini](https://huggingface.co/inclusionAI/LLaDA2.0-mini) |
| SDAR-8B-Chat | Dense | [JetLM/SDAR-8B-Chat](https://huggingface.co/JetLM/SDAR-8B-Chat) |
| SDAR-8B-Chat-b32 | Dense | [JetLM/SDAR-8B-Chat-b32](https://huggingface.co/JetLM/SDAR-8B-Chat-b32) |

### 2. MoE Weight Conversion

DMax models are fine-tuned from LLaDA2.0-mini, so both require the same **MoE merge** step before training:

```sh
python mbd_lm/scripts/moe_convertor.py \
  -i inclusionAI/LLaDA2.0-mini \
  -o inclusionAI/LLaDA2.0-mini-convert \
  -m merge
```

| Note | Dense models like SDAR do not require this step |
|------|------------------------------------------------|

### 3. Download Training Data

Use the dataset download script to fetch `SJTU-DENG-Lab/MBD-LMs-MultiTF-Datasets` and link it to the expected location:

```sh
scripts/download_dataset.sh
```

| What it does | Downloads data from HuggingFace, then symlinks each `.jsonl` into `dataset/` |
|--------------|--------------------------------------------------------------------------------|

The dataset contains four training splits — see [`dataset/README.md`](../dataset/README.md) for details on each.

## Usage

### Single-GPU / Local Test (Not Recommended)

Use `torchrun` directly to smoke-test a config on a single GPU:

```sh
cd /path/to/mbd-lms
source .venv/bin/activate

torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    mbd_lm/tasks/llada2/train_llada2_multi_tf_oput.py \
    mbd_lm/configs/sft/llada2/train_llada2_multi_tf_oput_code_b32.yaml
```

| Note | Single-GPU is only suitable for verifying configs parse correctly. Full training is prohibitively slow and will not produce usable results |
|------|-------------------------------------------------------------------------------------------------------------------------------------------|

### Multi-Node via Launch Script

For real training, use [`mbd_lm/train_multi_nodes_mbd_lm.sh`](../mbd_lm/train_multi_nodes_mbd_lm.sh). On a managed cluster (K8s), node IPs, rank, and world size are injected by the scheduler — the same command runs on every node with no manual coordination:

```sh
cd /path/to/mbd-lms
source .venv/bin/activate
TASK_REL_PATH=llada2/train_llada2_multi_tf_oput \
  CONFIG=mbd_lm/configs/sft/llada2/train_llada2_multi_tf_oput_math_b32.yaml \
  bash mbd_lm/train_multi_nodes_mbd_lm.sh
```

### What the Launch Script Does

| Responsibility | Detail |
|----------------|--------|
| NCCL configuration | InfiniBand timeouts, async error handling, debug logging |
| Node/GPU detection | Reads `PET_NNODES` / `PET_NODE_RANK` for managed clusters; auto-detects GPU count via `nvidia-smi` |
| Python path | Prepends repo root and `VeOmni/` to `PYTHONPATH` |
| Task & config resolution | `TASK_REL_PATH` and `CONFIG` can be overridden via environment variables |
| Logging | Timestamped, per-rank log files under `mbd_lm/logs/` |
| Torchrun launch | Builds distributed args for single- or multi-node mode |

## After Training

Checkpoints are saved in FSDP DCP format under `<output_dir>/checkpoints/` as `global_step_N/` directories. Use the batch conversion script to turn them into usable HuggingFace weights.

### Basic Usage

```sh
python mbd_lm/scripts/batch_convert_fsdp_to_moe.py \
    --checkpoints-dir /path/to/checkpoints \
    --model-assets-dir /path/to/model_assets
```

| Argument | Description |
|----------|-------------|
| `--checkpoints-dir` | Directory containing `global_step_N/` DCP subdirectories |
| `--model-assets-dir` | Directory with `config.json` + `tokenizer.json` (any already-converted checkpoint works) |

### What It Does

For each `global_step_N/` under `--checkpoints-dir`:

| Step | Output | Description |
|------|--------|-------------|
| 1. DCP → HF | `global_step_N/hf_ckpt/` | Converts FSDP DCP to HuggingFace format (all models) |
| 2. MoE split | `global_step_N/hf_ckpt_convert/` | Splits stacked expert weights into individual experts (MoE only) |

Already-converted checkpoints are skipped automatically. Both steps are always run — for dense models step 2 is a no-op.

### Common Options

| Option | Effect |
|--------|--------|
| `--steps 2000 7500 15000` | Convert only specific steps |
| `--skip-step2` | Skip MoE split (dense models, or if you only need merged weights) |
| `--force` | Re-run both steps even if outputs exist |
| `--dry-run` | Preview what would run without executing |
| `--moe-convertor <path>` | Override the auto-detected path to `moe_convertor.py` |

## Key Environment Variables

| Variable | Default | Set by |
|----------|---------|--------|
| `TASK_REL_PATH` | `sdar/train_sdar_multi_bd_distill_v2` | User |
| `CONFIG` | `mbd_lm/configs/sft/<TASK_REL_PATH>.yaml` | User |
| `PET_NNODES` | `1` | K8s scheduler (auto-injected) |
| `PET_NODE_RANK` | `0` | K8s scheduler (auto-injected) |
| `MASTER_ADDR` | `127.0.0.1` | K8s scheduler (auto-injected) |
| `MASTER_PORT` | `29531` | K8s scheduler (auto-injected) |
| `NPROC` | `$(nvidia-smi -L \| wc -l)` | Auto-detected |

| Note | `TASK_REL_PATH` and `CONFIG` are the only variables users typically need to set. Everything else is handled by the scheduler or auto-detection |
|------|-------------------------------------------------------------------------------------------------------------------------------------------------|

## Available Tasks

| Task | Script | Description |
|------|--------|-------------|
| LLaDA2-DMax | `llada2/train_llada2_multi_tf_oput.py` | LLaDA2 MultiTF variant with DMax-OPUT |
| LLaDA2 | `llada2/train_llada2_multi_tf.py` | LLaDA2 MultiTF with CE loss |
| SDAR | `sdar/train_sdar_multi_tf.py` | SDAR MultiTF training with CE loss |

## Customization

### Switching Tasks

```sh
TASK_REL_PATH=sdar/train_sdar_multi_tf bash mbd_lm/train_multi_nodes_mbd_lm.sh
```

### Overriding Config

```sh
CONFIG=/path/to/custom_config.yaml bash mbd_lm/train_multi_nodes_mbd_lm.sh
```

### Adapting to Other Cluster Schedulers

The script reads `PET_*` variables. For other schedulers, map their equivalents before invoking:

| Scheduler | `NNODES` | `NODE_RANK` | `MASTER_ADDR` |
|-----------|----------|-------------|---------------|
| SLURM | `$SLURM_NNODES` | `$SLURM_NODEID` | `$(scontrol show hostname $SLURM_NODELIST \| head -n1)` |
| PBS | `$(sort -u $PBS_NODEFILE \| wc -l)` | node index from `$PBS_NODEFILE` | set manually |

If you encounter an issue not covered here, please open a GitHub issue.
