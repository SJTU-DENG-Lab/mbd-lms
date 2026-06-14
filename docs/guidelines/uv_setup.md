# UV Setup Guide

This guide covers installing uv, configuring shell helpers, and using the companion functions to quickly set up a Python development environment.

## 1. Installing uv

Use the official installer with `UV_INSTALL_DIR` to set a custom path:

```sh
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/opt/uv" sh
```

`$HOME/opt/uv` is used as the example path; replace it with any directory you prefer.

| Artifact | Path |
|----------|------|
| uv binary | `$HOME/opt/uv/uv` |
| Env script | `$HOME/opt/uv/env` |

| Tip | Hand this doc to Claude Code or another AI coding assistant — they can run through the steps directly |
|-----|------------------------------------------------------------------------------------------------------|

## 2. Shell Configuration

Add the following to your `.zshrc` or `.bashrc` (adjust paths as needed).

### 2.1 Paths & Environment

```sh
# Load uv environment variables
. "$HOME/opt/uv/env"

# Ensure local bin and uv are on PATH
export PATH="$HOME/.local/bin:$PATH"
export PATH="$HOME/opt/uv:$PATH"

# Enable shell completion (zsh)
eval "$(uv generate-shell-completion zsh)"
```

| Config | Purpose |
|--------|---------|
| `. "$HOME/opt/uv/env"` | Sources the env script generated at install time, adding uv's bin to PATH |
| `generate-shell-completion` | Generates zsh/bash tab-completion — type `uv ` then press Tab to see subcommands |

### 2.2 Global Defaults

```sh
# uv package cache (keep it off $HOME to save space)
export UV_CACHE_DIR="$HOME/data/.cache/uv"

# Default Python interpreter for creating venvs
export UV_PYTHON="$HOME/data/.cache/micromamba/envs/syspy/bin/python"

# Default PyPI index
export UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple"
```

| Config | Purpose | Recommendation |
|--------|---------|----------------|
| `UV_CACHE_DIR` | Where uv stores downloaded wheel caches | Point to a large partition to avoid filling `$HOME` |
| `UV_PYTHON` | Base Python used by `uv sync` / `uv venv` | Use a reliable system Python managed by micromamba or conda |
| `UV_DEFAULT_INDEX` | Default PyPI index for `uv pip install` / `uv sync` | Aliyun mirror is recommended for users in China |

| About `UV_PYTHON` | uv needs a base Python interpreter to create `.venv`. The `syspy` here is a CPython installed via micromamba, isolated from the system Python |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------|

---

## 3. Shell Helper Functions

These functions provide shortcuts for switching mirrors, managing Python versions, and activating/creating environments.

### 3.1 `uv-index-set` — Switch PyPI Mirror

```sh
uv-index-set() {
    case "$1" in
        aliyun)
            export UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple"
            echo "✅ Switched to Aliyun mirror"
            ;;
        tuna)
            export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
            echo "✅ Switched to TUNA mirror"
            ;;
        pypi)
            export UV_DEFAULT_INDEX="https://pypi.org/simple"
            echo "✅ Switched to PyPI official"
            ;;
        *)
            echo "❌ Unknown index: '$1'. Options: aliyun, tuna, pypi" >&2
            return 1
            ;;
    esac
}
```

| Command | Effect |
|---------|--------|
| `uv-index-set aliyun` | Switch to Aliyun mirror (recommended in China) |
| `uv-index-set tuna` | Switch to Tsinghua TUNA mirror |
| `uv-index-set pypi` | Switch to PyPI official |

| Property | Detail |
|----------|--------|
| Scope | Current shell session only; not persisted across terminals |
| Mechanism | Modifies the `UV_DEFAULT_INDEX` env var; all subsequent `uv sync` / `uv pip install` use the specified index |

| Note | This project already pins indices via `[[tool.uv.index]]` in `pyproject.toml` (Aliyun default + PyTorch-specific), so dependency resolution points to Aliyun even without this function |
|------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

---

### 3.2 `uv-set-system-python` — Restore System Python

```sh
uv-set-system-python() {
    export UV_PYTHON="$HOME/data/.cache/micromamba/envs/syspy/bin/python"
    echo "✅ UV_PYTHON restored to system syspy"
}
```

| Property | Detail |
|----------|--------|
| When to use | After `uvon` has pointed `UV_PYTHON` to a `.venv`; call this before creating a new venv |
| Typical usage | `uv-set-system-python` then `uv sync` will use syspy as the base |

---

### 3.3 `uv-set-default-python` — Point UV_PYTHON to Current venv

```sh
uv-set-default-python() {
    export UV_PYTHON="$(pwd)/.venv/bin/python"
    echo "✅ UV_PYTHON set to $(pwd)/.venv/bin/python"
}
```

| Property | Detail |
|----------|--------|
| Called by | `uvon` invokes this automatically; no need to run it manually |
| Purpose | Keeps `UV_PYTHON` in sync with the active venv so subsequent `uv` commands use the same interpreter |

---

### 3.4 `uv-unset-python` — Clear UV_PYTHON

```sh
uv-unset-python() {
    unset UV_PYTHON
    echo "✅ UV_PYTHON cleared"
}
```

| Property | Detail |
|----------|--------|
| When to use | When you want uv to auto-discover Python from PATH rather than using a preset |
| Typical scenario | The first step inside `uv-create-venv` |

---

### 3.5 `uvon` — Activate the Project venv

```sh
uvon() {
    if [ -f .venv/bin/activate ]; then
        source .venv/bin/activate
        echo "✅ uv venv activated"
        uv-set-default-python
    else
        echo ".venv/bin/activate not found"
        return 1
    fi
}
```

| Note | This is the most frequently used command — just `cd` into a project and run `uvon` |
|------|-------------------------------------------------------------------------------------|

| Step | Action | Effect |
|------|--------|--------|
| 1 | `source .venv/bin/activate` | Activates the venv; `python` and `pip` now point to `.venv` |
| 2 | `uv-set-default-python` | Points `UV_PYTHON` to the active venv for future uv operations |

| Scenario | Command |
|----------|---------|
| Start daily work | `cd <project-dir> && uvon` |
| Run scripts afterwards | `python ...` |

| Note | `uvon` only activates an existing `.venv` — it does not install anything. If the venv doesn't exist yet, use `uv-create-venv` first |
|------|-------------------------------------------------------------------------------------------------------------------------------------|

---

### 3.6 `uv-create-venv` — Create & Activate a New venv

```sh
uv-create-venv() {
    uv-unset-python
    uv sync
    echo "✅ uv venv created"
    uvon
}
```

| Note | This is the complete command for setting up a fresh environment |
|------|-----------------------------------------------------------------|

| Step | Action | Effect |
|------|--------|--------|
| 1 | `uv-unset-python` | Clears `UV_PYTHON` so uv uses the system default Python |
| 2 | `uv sync` | Reads `pyproject.toml` + `uv.lock`, creates `.venv`, and installs all dependencies |
| 3 | `uvon` | Activates the newly created venv and configures the environment |

| Scenario | Command |
|----------|---------|
| After first clone | `git clone ... && cd <project-dir> && uv-create-venv` |
| Rebuild after lockfile update | `rm -rf .venv && uv-create-venv` |

| vs. bare `uv sync` | `uv-create-venv` runs `uv-unset-python` first to ensure the system Python is used, avoiding nested references from an already-active venv. If you understand this nuance, `uv sync && uvon` is equivalent |
|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

---

## 4. Typical Workflows

### 4.1 First-Time Project Setup

```sh
# 1. Clone the project
git clone --recurse-submodules <repo-url>
cd <project-dir>

# 2. Initial sync
uv-create-venv
```

### 4.2 Daily Development

```sh
cd <project-dir>
uvon           # Activate the environment
python ...     # Run scripts or training
```

### 4.3 Switching Mirrors

```sh
uv-index-set pypi     # Temporarily use the official index (e.g. to check package versions)
uv pip install <pkg>
uv-index-set aliyun   # Switch back
```

### 4.4 Adding Dependencies

```sh
uv add <package>           # Production dependency
uv add --dev <package>     # Dev dependency
uv add --extra gpu <pkg>   # GPU-extra dependency
```

---

## 5. Configuration Hierarchy

| Layer | Key Contents | Scope | Priority | Description |
|-------|-------------|-------|----------|-------------|
| Env vars (shell rc) | `UV_CACHE_DIR`, `UV_PYTHON`, `UV_DEFAULT_INDEX` | All projects | Low (overridable) | Global default behavior |
| `pyproject.toml` | `[tool.uv]` block | Current project | Medium (overrides env vars) | Project-level: indices, sources, build isolation, etc. |
| `uv.lock` | Exact pinned dependency tree | Current project | High (final authority) | Auto-maintained by `uv sync`; guarantees identical dependencies for all developers |
