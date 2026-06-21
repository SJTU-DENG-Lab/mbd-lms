# Benchmark Guide

For reported MBD-LMs results, benchmark through the Diffulex `mbd-lms`
reproduction branch:

<https://github.com/SJTU-DENG-Lab/Diffulex/tree/mbd-lms>

That branch is the stable reference for reproducing the paper experiments. It
keeps the benchmark configs, model choices, and runtime assumptions aligned with
the MBD-LMs training and evaluation setup.

Diffulex `main` is the active engine branch:

<https://github.com/SJTU-DENG-Lab/Diffulex/tree/main>

Use `main` for continued engine development, open-source contributions, new
algorithm exploration, and deployment-oriented optimization. Because `main`
evolves with runtime changes, its exact metrics may differ from the reproduction
branch.

When comparing results, keep the terminology consistent:

| Term | Meaning |
|------|---------|
| SingleBD | Native one-block-at-a-time block diffusion decoding. |
| MultiBD | Multi-Block Diffusion over a bounded running-set of blocks. |
| MultiTF | The post-training recipe used to align MBD-LMs with practical MultiBD inference states. |
| Block Buffer | The fixed-size inference mechanism that makes MultiBD static-shape friendly. |
