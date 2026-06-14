"""Utilities for distillation temperature schedules (teacher-side schedule + fixed student)."""

from __future__ import annotations

import math


def interpolate_value(start: float, end: float, ratio: float, style: str = "linear") -> float:
    ratio = max(0.0, min(1.0, ratio))

    if style == "linear":
        return start + (end - start) * ratio
    if style == "cosine":
        cosine_ratio = (1 - math.cos(ratio * math.pi)) / 2
        return start + (end - start) * cosine_ratio
    raise ValueError(f"Unknown interpolation style: {style}")


class DistillTemperatureController:
    """Teacher temperature may schedule; student temperature is fixed (kl_temperature_student_fixed)."""

    def __init__(self, train_args):
        self.train_args = train_args
        self.kl_ema: float | None = None
        self.kl_baseline: float | None = None

    def get(self, step: int, total_steps: int) -> tuple[float, float]:
        args = self.train_args
        teacher_start = float(getattr(args, "kl_temperature_teacher_start", 1.0))
        teacher_end = float(getattr(args, "kl_temperature_teacher_end", teacher_start))
        student_temp = float(getattr(args, "kl_temperature_student_fixed", 1.0))
        style = getattr(args, "teacher_temperature_schedule_style", "fixed")

        if style == "fixed" or total_steps <= 0:
            teacher_temp = teacher_start
        elif style in {"linear", "cosine"}:
            teacher_temp = interpolate_value(teacher_start, teacher_end, step / total_steps, style)
        elif style == "loss_aware":
            if self.kl_ema is None or self.kl_baseline is None:
                teacher_temp = teacher_start
            else:
                progress = 1.0 - min(max(self.kl_ema / max(self.kl_baseline, 1e-8), 0.0), 1.0)
                teacher_temp = teacher_start + (teacher_end - teacher_start) * progress
        else:
            raise ValueError(f"Unknown teacher temperature schedule style: {style}")

        return teacher_temp, student_temp

    def update_from_kl(self, kl_loss_value: float) -> None:
        args = self.train_args
        if getattr(args, "teacher_temperature_schedule_style", "fixed") != "loss_aware":
            return

        val = max(float(kl_loss_value), 0.0)
        momentum = float(getattr(args, "teacher_temperature_loss_ema_momentum", 0.9))

        if self.kl_ema is None:
            self.kl_ema = val
            self.kl_baseline = max(val, 1e-8)
            return

        self.kl_ema = momentum * self.kl_ema + (1.0 - momentum) * val
        self.kl_baseline = max(self.kl_baseline or 0.0, self.kl_ema, 1e-8)
