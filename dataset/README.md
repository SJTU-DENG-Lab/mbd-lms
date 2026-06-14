# MBD-LMs MultiTF Datasets

## Contents

| Dataset | Task | Samples |
|---------|------|---------|
| `llada2_math_multi_tf_60k_oput.jsonl` | Math reasoning | 60k |
| `llada2_code_multi_tf_60k_oput.jsonl` | Code generation | 60k |
| `sdar_code_multi_tf_10k.jsonl` | Code generation | 10k |
| `sdar_math_multi_tf_20k.jsonl` | Math reasoning | 20k |

## Data Sources

### LLaDA2

Both LLaDA2 datasets are randomly sampled (60k each) from the DMax training trajectories:

| Split | Source |
|-------|--------|
| Code | [Zigeng/DMax-LLaDA-2.0-Mini-Code-Trajectories](https://huggingface.co/datasets/Zigeng/DMax-LLaDA-2.0-Mini-Code-Trajectories) |
| Math | [Zigeng/DMax-LLaDA-2.0-Mini-Math-Trajectories](https://huggingface.co/datasets/Zigeng/DMax-LLaDA-2.0-Mini-Math-Trajectories) |

### SDAR

| Split | Source | Notes |
|-------|--------|-------|
| Math (20k) | [When-Does-Reasoning-Matter/math-reasoning-ift-pairs](https://huggingface.co/datasets/When-Does-Reasoning-Matter/math-reasoning-ift-pairs) | Randomly sampled |
| Code (10k) | [Zigeng/DMax-LLaDA-2.0-Mini-Code-Trajectories](https://huggingface.co/datasets/Zigeng/DMax-LLaDA-2.0-Mini-Code-Trajectories) | Sampled then regenerated with [JetLM/SDAR-30B-A3B-Chat-b32](https://huggingface.co/JetLM/SDAR-30B-A3B-Chat-b32) |
