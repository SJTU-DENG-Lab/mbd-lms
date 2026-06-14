"""Loss weight scheduler for CE/KL in LLaDA2 dual_bd / multi_bd distill training."""

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from mbd_lm.tasks.utils.llada2_args import LLaDA2TrainingArguments


def loss_weight_schedule_factor(
    step: int,
    warmup_steps: int,
    hold_steps: int,
    decay_steps: int,
    decay_ratio: float,
) -> float:
    """Return multiplier in [0, 1] or [decay_ratio, 1]: warmup -> hold -> linear decay."""
    if step < warmup_steps:
        return float(step) / warmup_steps if warmup_steps > 0 else 1.0
    if step < warmup_steps + hold_steps:
        return 1.0
    decay_start = warmup_steps + hold_steps
    if step >= decay_start + decay_steps:
        return decay_ratio
    # linear decay from 1.0 to decay_ratio over decay_steps
    progress = (step - decay_start) / decay_steps
    return 1.0 + (decay_ratio - 1.0) * progress


class LossWeightScheduler:
    """
    Scheduler for ce/kl loss weights: step-based (warmup/hold/decay) and/or loss-aware (balance by EMA of losses).
    Optionally normalizes effective weights so ce_weight_eff + kl_weight_eff = 1.
    """

    def __init__(self, args: "LLaDA2TrainingArguments"):
        self.args = args
        self._ce_ema: Optional[float] = None
        self._kl_ema: Optional[float] = None
        self._momentum = getattr(args, "loss_weight_schedule_ema_momentum", 0.9)

    def update_ema(self, ce_loss: float, kl_loss: float) -> None:
        """Update EMA of ce/kl loss (call after each step)."""
        if self._ce_ema is None:
            self._ce_ema = ce_loss
            self._kl_ema = kl_loss
        else:
            self._ce_ema = self._momentum * self._ce_ema + (1.0 - self._momentum) * ce_loss
            self._kl_ema = self._momentum * self._kl_ema + (1.0 - self._momentum) * kl_loss

    def get_weights(self, global_step: int) -> Tuple[float, float]:
        """
        Return (ce_weight_eff, kl_weight_eff) for this step.
        Uses step-based schedule and/or loss-aware balance; then optionally normalizes to sum 1.
        """
        style = getattr(self.args, "loss_weight_schedule_style", "step") or "step"
        warmup = getattr(self.args, "loss_weight_schedule_warmup_steps", 0)
        hold = getattr(self.args, "loss_weight_schedule_hold_steps", 0)
        fix_steps = getattr(self.args, "loss_weight_schedule_fix_steps", 500)
        use_step_phase = style == "step" or (style == "step_then_balance" and global_step < warmup + hold)
        use_fix_phase = style == "fix_then_balance" and global_step < fix_steps

        if use_fix_phase:
            # fix_then_balance: keep preset ce_weight/kl_weight unchanged
            ce_eff = self.args.ce_weight
            kl_eff = self.args.kl_weight
        elif use_step_phase:
            factor = loss_weight_schedule_factor(
                global_step - 1,
                getattr(self.args, "loss_weight_schedule_warmup_steps", 0),
                getattr(self.args, "loss_weight_schedule_hold_steps", 0),
                getattr(self.args, "loss_weight_schedule_decay_steps", 0),
                getattr(self.args, "loss_weight_schedule_decay_ratio", 0.5),
            )
            ce_eff = self.args.ce_weight * (factor if self.args.loss_weight_schedule_ce else 1.0)
            kl_eff = self.args.kl_weight * (factor if self.args.loss_weight_schedule_kl else 1.0)
        else:
            # loss_balance or step_then_balance after warmup+hold: set weights so (ce_eff*ce_ema):(kl_eff*kl_ema) = ce_weight:kl_weight
            ce_ema = self._ce_ema if self._ce_ema is not None and self._ce_ema > 0 else 1.0
            kl_ema = self._kl_ema if self._kl_ema is not None and self._kl_ema > 0 else 1.0
            # ce_eff/kl_eff = (ce_weight/kl_weight)*(kl_ema/ce_ema) => set ce_eff=R, kl_eff=1 then normalize
            r = (self.args.ce_weight / max(self.args.kl_weight, 1e-8)) * (kl_ema / max(ce_ema, 1e-8))
            if self.args.loss_weight_schedule_ce and self.args.loss_weight_schedule_kl:
                ce_eff = r
                kl_eff = 1.0
            elif self.args.loss_weight_schedule_ce:
                ce_eff = self.args.ce_weight
                kl_eff = self.args.kl_weight
            elif self.args.loss_weight_schedule_kl:
                ce_eff = self.args.ce_weight
                kl_eff = self.args.kl_weight
            else:
                ce_eff = self.args.ce_weight
                kl_eff = self.args.kl_weight

        if getattr(self.args, "loss_weight_schedule_normalize", True) and (ce_eff + kl_eff) > 0:
            s = ce_eff + kl_eff
            ce_eff = ce_eff / s
            kl_eff = kl_eff / s
        return ce_eff, kl_eff

    def get_log_state(self, global_step: int) -> Dict[str, Any]:
        """Return dict of weight scheduler internal state for wandb (EMA, phase, etc.)."""
        style = getattr(self.args, "loss_weight_schedule_style", "step") or "step"
        warmup = getattr(self.args, "loss_weight_schedule_warmup_steps", 0)
        hold = getattr(self.args, "loss_weight_schedule_hold_steps", 0)
        fix_steps = getattr(self.args, "loss_weight_schedule_fix_steps", 500)
        use_fix = style == "fix_then_balance" and global_step < fix_steps
        use_step = style == "step" or (style == "step_then_balance" and global_step < warmup + hold)
        if use_fix:
            phase = "fix"
        elif use_step:
            phase = "step"
        else:
            phase = "balance"
        out: Dict[str, Any] = {
            "weight_schedule/phase": phase,
            "weight_schedule/ce_loss_ema": self._ce_ema if self._ce_ema is not None else 0.0,
            "weight_schedule/kl_loss_ema": self._kl_ema if self._kl_ema is not None else 0.0,
            "weight_schedule/ema_momentum": self._momentum,
        }
        return out
