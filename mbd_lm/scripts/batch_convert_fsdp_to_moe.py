#!/usr/bin/env python3
"""
Batch convert all FSDP checkpoints:
  1. FSDP/DCP → hf_ckpt/        (VeOmni save_model_weights, merged experts)
  2. hf_ckpt  → hf_ckpt_convert/ (moe_convertor.py split, individual experts)

For each global_step_N/ under --checkpoints-dir:
  - If hf_ckpt/ exists → skip step 1
  - If hf_ckpt_convert/ exists → skip step 2

Usage:
    python scripts/batch_convert_fsdp_to_moe.py \
        --checkpoints-dir .../checkpoints \
        --model-assets-dir .../checkpoints/global_step_15000/hf_ckpt_convert
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Batch FSDP → HF (VeOmni) → split MoE (moe_convertor.py)"
    )
    parser.add_argument(
        "--checkpoints-dir", type=Path, required=True,
        help="Directory containing global_step_* subdirectories",
    )
    parser.add_argument(
        "--model-assets-dir", type=Path, required=True,
        help="Directory with config.json, tokenizer.json, etc.",
    )
    parser.add_argument(
        "--moe-convertor", type=Path, default=None,
        help="Path to moe_convertor.py (default: auto-detect)",
    )
    parser.add_argument(
        "--steps", nargs="*", default=None,
        help="Specific steps, e.g. --steps 2000 7500",
    )
    parser.add_argument("--force", action="store_true", help="Force re-run both steps")
    parser.add_argument("--force-hf", action="store_true", help="Force re-run step 1")
    parser.add_argument("--force-split", action="store_true", help="Force re-run step 2")
    parser.add_argument("--skip-step1", action="store_true", help="Only do step 2")
    parser.add_argument("--skip-step2", action="store_true", help="Only do step 1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    checkpoints_dir = args.checkpoints_dir.resolve()
    model_assets_dir = args.model_assets_dir.resolve()

    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints dir not found: {checkpoints_dir}")
    if not model_assets_dir.exists():
        raise FileNotFoundError(f"Model assets dir not found: {model_assets_dir}")
    if not (model_assets_dir / "config.json").exists():
        raise FileNotFoundError(f"config.json not found in: {model_assets_dir}")

    # Resolve moe_convertor.py path
    if args.moe_convertor:
        moe_convertor = args.moe_convertor.resolve()
    else:
        moe_convertor = (
            Path(__file__).resolve().parents[1]
            / "multi_bd_train" / "scripts" / "moe_convertor.py"
        )
    if not moe_convertor.exists():
        raise FileNotFoundError(f"moe_convertor.py not found: {moe_convertor}")

    # Select checkpoints
    if args.steps:
        ckpts = []
        for step in args.steps:
            name = step if step.startswith("global_step_") else f"global_step_{step}"
            ckpts.append(checkpoints_dir / name)
    else:
        ckpts = sorted(
            checkpoints_dir.glob("global_step_*"),
            key=lambda p: int(p.name.rsplit("_", 1)[-1]),
        )

    missing = [p for p in ckpts if not p.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint dirs: {missing}")

    print(f"checkpoints_dir = {checkpoints_dir}")
    print(f"model_assets_dir = {model_assets_dir}")
    print(f"moe_convertor    = {moe_convertor}")
    print(f"selected = {', '.join(p.name for p in ckpts)}")

    for ckpt in ckpts:
        print(f"\n{'='*60}")
        print(f"=== {ckpt.name} ===")
        print(f"{'='*60}", flush=True)

        hf_ckpt = ckpt / "hf_ckpt"
        hf_split = ckpt / "hf_ckpt_convert"

        if args.dry_run:
            do1 = not args.skip_step1
            do2 = not args.skip_step2
            print(f"  hf_ckpt:          {'CREATE' if do1 and (args.force or args.force_hf or not hf_ckpt.exists()) else 'skip'}")
            print(f"  hf_ckpt_convert:  {'CREATE' if do2 and (args.force or args.force_split or not hf_split.exists()) else 'skip'}")
            continue

        # --- Step 1: FSDP/DCP → hf_ckpt (VeOmni) ---
        need_step1 = args.force or args.force_hf or (not args.skip_step1 and not hf_ckpt.exists())

        if args.force or args.force_hf:
            if hf_ckpt.exists():
                shutil.rmtree(hf_ckpt)
                print("[force] removed existing hf_ckpt")

        if args.skip_step1:
            print("[skip] step1 disabled by --skip-step1")
        elif not need_step1:
            print(f"[skip] hf_ckpt already exists: {hf_ckpt}")
        else:
            run_fsdp_to_hf(ckpt, hf_ckpt, model_assets_dir)

        # --- Step 2: hf_ckpt → hf_ckpt_convert (moe_convertor.py split) ---
        need_step2 = args.force or args.force_split or (not args.skip_step2 and not hf_split.exists())

        if args.force or args.force_split:
            if hf_split.exists():
                shutil.rmtree(hf_split)
                print("[force] removed existing hf_ckpt_convert")

        if args.skip_step2:
            print("[skip] step2 disabled by --skip-step2")
        elif not hf_ckpt.exists():
            print(f"[skip] step2: hf_ckpt not found, cannot split")
        elif not need_step2:
            print(f"[skip] hf_ckpt_convert already exists: {hf_split}")
        else:
            run_moe_split(hf_ckpt, hf_split, moe_convertor)

        print(f"[done] {ckpt.name}", flush=True)

    print("\n=== All done ===")


# ---------------------------------------------------------------------------
# Step 1 implementation (same as convert_all_fsdp_ckpts_to_hf_split.py)
# ---------------------------------------------------------------------------

def run_fsdp_to_hf(ckpt: Path, output: Path, assets_dir: Path) -> None:
    """FSDP/DCP → merged HF via VeOmni save_model_weights."""
    print(f"[step1] FSDP → hf_ckpt: {ckpt}")

    from veomni.checkpoint import dcp_to_torch_state_dict
    from veomni.models import build_tokenizer, save_model_weights
    from transformers import AutoConfig

    state_dict = dcp_to_torch_state_dict(save_checkpoint_path=str(ckpt))
    if not state_dict:
        raise RuntimeError(f"Empty state_dict loaded from {ckpt}")

    sample_key = next(iter(state_dict))
    if sample_key.startswith("model.model."):
        state_dict = {key[len("model."):]: value for key, value in state_dict.items()}

    config = AutoConfig.from_pretrained(str(assets_dir), trust_remote_code=True)
    tokenizer = build_tokenizer(str(assets_dir))
    save_model_weights(str(output), state_dict, model_assets=[config, tokenizer])
    print(f"[step1] Done: {output}")


# ---------------------------------------------------------------------------
# Step 2 implementation (calls moe_convertor.py)
# ---------------------------------------------------------------------------

def run_moe_split(hf_ckpt: Path, output: Path, moe_convertor: Path) -> None:
    """Call moe_convertor.py --mode split to unstack expert weights."""
    print(f"[step2] moe_convertor split: {hf_ckpt} → {output}")
    cmd = [
        sys.executable,
        str(moe_convertor),
        "--input-path", str(hf_ckpt),
        "--output-path", str(output),
        "--mode", "split",
    ]
    subprocess.run(cmd, check=True)
    print(f"[step2] Done: {output}")


if __name__ == "__main__":
    main()
