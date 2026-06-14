# Training Setup Guide

Step-by-step instructions for setting up the MBD-LM training environment from scratch.

## 0. Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | **3.11** | `pyproject.toml` requires `>=3.11, <3.13` |
| uv | **>= 0.8.14** | Follow the [UV Setup Guide](uv_setup.md) to install and configure |

## 1. Clone with Submodules

```sh
git clone https://github.com/SJTU-DENG-Lab/mbd-lms.git
cd mbd-lms
```

The `VeOmni` submodule is already included in the repo (no `--recurse-submodules` needed unless a fresh clone).

## 2. Bootstrap the venv

The first `uv sync` will fail because `VeOmni/dist/veomni-0.1.0-py3-none-any.whl` doesn't exist yet. This is expected — the `.venv` is still created successfully.

```sh
unset UV_PYTHON && uv sync --all-packages --extra gpu --dev
```

| What happens | Detail |
|--------------|--------|
| venv | `.venv/` is provisioned with the base Python and all resolvable dependencies |
| VeOmni | Fails with `Distribution not found at: .../VeOmni/dist/veomni-0.1.0-py3-none-any.whl` |
| Result | The error is harmless — the venv is ready to use; just missing the VeOmni wheel |

Expected error output:

```
Using CPython 3.11.14
Creating virtual environment at: .venv
error: Distribution not found at: file:///.../mbd-lms/VeOmni/dist/veomni-0.1.0-py3-none-any.whl
```

## 3. Build VeOmni

Activate the bootstrapped venv, then build the VeOmni wheel:

```sh
uvon                           # activates .venv and sets UV_PYTHON
cd VeOmni
uv pip install build
python -m build
cd ..
```

| Step | Command | Effect |
|------|---------|--------|
| 1 | `uvon` | Activates `.venv` and points `UV_PYTHON` to it (equivalent to `source .venv/bin/activate`) |
| 2 | `uv pip install build` | Installs the PEP 517 build frontend into the venv |
| 3 | `python -m build` | Builds VeOmni, producing `dist/veomni-0.1.0-py3-none-any.whl` |

After this step, `VeOmni/dist/` contains the wheel that `pyproject.toml` references via:

```toml
[tool.uv.sources]
veomni = { path = "VeOmni/dist/veomni-0.1.0-py3-none-any.whl", marker = "extra == 'gpu'" }
```

## 4. Final Sync

With the VeOmni wheel in place, re-run sync to resolve all dependencies:

```sh
uv sync --all-packages --extra gpu --dev
uv pip install -e .
```

| Step | Command | Effect |
|------|---------|--------|
| 1 | `uv sync --all-packages --extra gpu --dev` | Resolves all dependencies including VeOmni, flash-attn, torch (CUDA 12.8) |
| 2 | `uv pip install -e .` | Installs `mbd_lm` in editable mode |

## 5. Verify

```sh
python -c "import mbd_lm; print(mbd_lm.__version__)"
```

|  |  |
|-------|----------|
| Expected output | `0.1.0` |

## Quick Reference

| Stage | Commands |
|-------|----------|
| Bootstrap | `unset UV_PYTHON && uv sync --all-packages --extra gpu --dev` |
| Build VeOmni | `uvon && cd VeOmni && uv pip install build && python -m build && cd ..` |
| Final sync | `uv sync --all-packages --extra gpu --dev && uv pip install -e .` |
| Verify | `python -c "import mbd_lm; print(mbd_lm.__version__)"` |

> If you run into download issues (e.g. mirrors unreachable), switch to a working index: `uv-index-set aliyun` or `uv-index-set tuna`.
