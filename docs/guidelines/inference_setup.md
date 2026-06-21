# Inference Setup

MBD-LMs are trained and defined in this repository, but the production inference
engine is Diffulex.

Use the Diffulex `mbd-lms` branch when you want to reproduce the reported
MBD-LMs experiment behavior:

```sh
git clone https://github.com/SJTU-DENG-Lab/Diffulex.git
cd Diffulex
git checkout mbd-lms
```

Use Diffulex `main` when you want to develop the engine, contribute new runtime
features, or explore new decoding algorithms:

```sh
git checkout main
```

The method name in the paper is **Multi-Block Diffusion (MultiBD)**. In
Diffulex configs, the corresponding runtime strategy is:

```yaml
decoding_strategy: multi_bd
```

Diffulex `main` may contain newer performance optimizations and model-specific
runtime changes, so use `mbd-lms` for experiment reproduction and `main` for
future development.
