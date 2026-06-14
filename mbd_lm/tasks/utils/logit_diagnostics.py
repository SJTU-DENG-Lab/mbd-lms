"""Helpers for lightweight logit diagnostics during training."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


def empty_logit_diag() -> Dict[str, float]:
    return {
        "count_positions": 0.0,
        "count_values": 0.0,
        "sum": 0.0,
        "sumsq": 0.0,
        "abs_sum": 0.0,
        "entropy_sum": 0.0,
        "top1_conf_sum": 0.0,
    }


def summarize_masked_logits(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute aggregate moments over supervised positions."""
    valid_mask = labels != -100
    if not torch.any(valid_mask):
        return empty_logit_diag()

    selected = logits[valid_mask].float()
    if selected.numel() == 0:
        return empty_logit_diag()

    probs = F.softmax(selected, dim=-1)
    log_probs = F.log_softmax(selected, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    top1_conf = probs.max(dim=-1).values
    flat = selected.reshape(-1)

    return {
        "count_positions": float(selected.shape[0]),
        "count_values": float(flat.numel()),
        "sum": float(flat.sum().item()),
        "sumsq": float((flat * flat).sum().item()),
        "abs_sum": float(flat.abs().sum().item()),
        "entropy_sum": float(entropy.sum().item()),
        "top1_conf_sum": float(top1_conf.sum().item()),
    }


def add_logit_diag(dst: Dict[str, float], src: Dict[str, float]) -> None:
    for key, value in src.items():
        dst[key] += float(value)


def finalize_logit_diag(prefix: str, totals: Dict[str, float]) -> Dict[str, float]:
    count_positions = totals["count_positions"]
    count_values = totals["count_values"]

    mean = totals["sum"] / count_values if count_values > 0 else 0.0
    variance = (totals["sumsq"] / count_values - mean * mean) if count_values > 0 else 0.0
    variance = max(variance, 0.0)
    std = math.sqrt(variance)
    mean_abs = totals["abs_sum"] / count_values if count_values > 0 else 0.0
    entropy = totals["entropy_sum"] / count_positions if count_positions > 0 else 0.0
    top1_conf = totals["top1_conf_sum"] / count_positions if count_positions > 0 else 0.0

    return {
        f"{prefix}_count_positions": count_positions,
        f"{prefix}_logit_mean": mean,
        f"{prefix}_logit_std": std,
        f"{prefix}_logit_mean_abs": mean_abs,
        f"{prefix}_entropy": entropy,
        f"{prefix}_top1_conf": top1_conf,
    }
