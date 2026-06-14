"""Early stopping monitor for LLaDA2 dual_bd / multi_bd distill training."""

from typing import Any, Dict, List, Optional, Tuple


class EarlyStoppingMonitor:
    """Monitor total_loss, ce_loss, kl_loss; trigger when any rises significantly or all stagnate."""

    def __init__(
        self,
        min_steps: int,
        patience: int,
        rise_ratio: float,
        rise_window: int,
        stagnation_steps: int,
        stagnation_tol_ratio: float,
    ):
        self.min_steps = min_steps
        self.patience = patience
        self.rise_ratio = rise_ratio
        self.rise_window = rise_window
        self.stagnation_steps = stagnation_steps
        self.stagnation_tol_ratio = stagnation_tol_ratio
        self._loss_history: List[Tuple[float, float, float]] = []  # (total, ce, kl)
        self._rise_triggered_at: Optional[int] = None
        self._best_total: Optional[float] = None
        self._steps_since_best: int = 0

    def step(self, global_step: int, total_loss: float, ce_loss: float, kl_loss: float) -> Tuple[bool, str]:
        """
        Returns (should_stop, reason).
        """
        self._loss_history.append((total_loss, ce_loss, kl_loss))
        if self.rise_window > 0 and len(self._loss_history) > self.rise_window:
            self._loss_history.pop(0)

        if global_step < self.min_steps:
            return False, ""

        # Check rise: any loss significantly above recent min
        if len(self._loss_history) >= self.rise_window:
            recent = self._loss_history[-self.rise_window:]
            min_total = min(x[0] for x in recent)
            min_ce = min(x[1] for x in recent)
            min_kl = min(x[2] for x in recent)
            if (
                total_loss >= min_total * (1.0 + self.rise_ratio)
                or ce_loss >= min_ce * (1.0 + self.rise_ratio)
                or (min_kl > 0 and kl_loss >= min_kl * (1.0 + self.rise_ratio))
            ):
                if self._rise_triggered_at is None:
                    self._rise_triggered_at = global_step
                if global_step - self._rise_triggered_at >= self.patience:
                    return True, "early_stop_rise"
            else:
                self._rise_triggered_at = None

        # Check stagnation: no improvement in total_loss for stagnation_steps
        if self._best_total is None or total_loss < self._best_total:
            self._best_total = total_loss
            self._steps_since_best = 0
        else:
            self._steps_since_best += 1
        if self._steps_since_best >= self.stagnation_steps:
            # Optionally require relative change below tol; here we use "no new best" as stagnation
            return True, "early_stop_stagnation"

        return False, ""

    def get_log_state(self) -> Dict[str, Any]:
        """Return dict of early stopping internal state for wandb."""
        return {
            "early_stop/rise_triggered_at": self._rise_triggered_at if self._rise_triggered_at is not None else -1,
            "early_stop/best_total": self._best_total if self._best_total is not None else 0.0,
            "early_stop/steps_since_best": self._steps_since_best,
        }
