"""Shared LLaDA2 argument dataclasses for dual_bd / multi_bd distill training."""

from dataclasses import dataclass, field
from typing import Literal, Optional

from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments


logger = helper.create_logger(__name__)


@dataclass
class LLaDA2ModelArguments(ModelArguments):
    attn_implementation: Optional[Literal["eager", "sdpa", "flex_attention"]] = field(
        default="sdpa",
        metadata={"help": "Attention implementation to use."},
    )


@dataclass
class LLaDA2DataArguments(DataArguments):
    data_type: Literal["conversation", "tokenid"] = field(
        default="conversation",
        metadata={"help": "Type of the training data."},
    )
    datasets_type: Literal["mapping", "local"] = field(
        default="mapping",
        metadata={"help": "Type of the datasets."},
    )
    text_keys: str = field(
        default="messages",
        metadata={"help": "Key to get text from the training data."},
    )
    noise_range: list[list[float]] = field(
        default_factory=lambda: [[0.1, 0.3], [0.5, 0.8]],
        metadata={"help": "Noise level for random flip input_ids to mask_ids"},
    )
    noise_range_dict: dict[str, float] = field(
        default_factory=lambda: {"low_min": 0.1, "low_max": 0.3, "high_min": 0.5, "high_max": 0.8},
        metadata={"help": "Noise level for random flip input_ids to mask_ids"},
    )
    buffer_size: int = field(
        default=2,
        metadata={"help": "Buffer size for multi_bd (e.g. overlap); dual_bd training ignores this."},
    )
    noise_transition_margin_ratio: float = field(
        default=0.05,
        metadata={"help": "Multi-BD noise transition: effective_high = high - margin_ratio * (high - low); probability never reaches high. 0 = allow touching high."},
    )
    noise_transition_gamma: float = field(
        default=2.0,
        metadata={"help": "Multi-BD noise transition: power-law exponent (>1 biases each step toward effective_high). Larger = stronger pull toward upper bound."},
    )
    buffer_internal_ramp: Literal["random", "linear", "linear_random", "chain_uniform", "chain_uniform_opt", "sorted_uniform"] = field(
        default="random",
        metadata={"help": "Multi-BD per-buffer ramp: 'random' = power-law steps; 'linear' = deterministic linear low->high; 'linear_random' = n intervals, sample t uniformly in i-th interval, prob = low + (high-low)*t; 'chain_uniform' = monotonic chained uniform sampling; 'sorted_uniform' = sample n uniform values and sort ascending."},
    )
    explicit_block1_noise_min: float = field(
        default=-1.0,
        metadata={"help": "Optional explicit noise min for the 1st block inside each active buffer. Negative disables explicit override."},
    )
    explicit_block1_noise_max: float = field(
        default=-1.0,
        metadata={"help": "Optional explicit noise max for the 1st block inside each active buffer. Negative disables explicit override."},
    )
    explicit_block2_noise_min: float = field(
        default=-1.0,
        metadata={"help": "Optional explicit noise min for the 2nd block inside each active buffer. Negative disables explicit override."},
    )
    explicit_block2_noise_max: float = field(
        default=-1.0,
        metadata={"help": "Optional explicit noise max for the 2nd block inside each active buffer. Negative disables explicit override."},
    )
    noise_schedule_mode: Literal["buffer_ramp", "d2f"] = field(
        default="buffer_ramp",
        metadata={"help": "Multi-BD noise scheduling mode. buffer_ramp uses per-buffer ramps; d2f uses one global low-to-high schedule."},
    )
    coverage_enable: bool = field(
        default=False,
        metadata={"help": "If True, generate multiple buffer-layout variants for each sample."},
    )
    coverage_random_n: int = field(
        default=0,
        metadata={"help": "Number of random buffer layouts per sample when coverage_enable=True."},
    )
    coverage_mode: Literal["full", "random_only"] = field(
        default="full",
        metadata={"help": "Coverage layout mode: full=shifted deterministic + random; random_only=random only."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.noise_range = [
            [self.noise_range_dict["low_min"], self.noise_range_dict["low_max"]],
            [self.noise_range_dict["high_min"], self.noise_range_dict["high_max"]],
        ]
        if len(self.noise_range) != 2:
            raise ValueError(f"noise_range must be [(low_min, low_max), (high_min, high_max)], got {self.noise_range}")
        if self.noise_range[0][0] > self.noise_range[0][1]:
            raise ValueError(f"low_min ({self.noise_range[0][0]}) cannot be greater than low_max ({self.noise_range[0][1]})")
        if self.noise_range[1][0] > self.noise_range[1][1]:
            raise ValueError(f"high_min ({self.noise_range[1][0]}) cannot be greater than high_max ({self.noise_range[1][1]})")
        if self.noise_range[0][0] < 0.0 or self.noise_range[0][1] > 1.0:
            raise ValueError(f"low_min and low_max must be between 0.0 and 1.0")
        if self.noise_range[1][0] < 0.0 or self.noise_range[1][1] > 1.0:
            raise ValueError(f"high_min and high_max must be between 0.0 and 1.0")
        if self.coverage_random_n < 0:
            raise ValueError(f"coverage_random_n must be >= 0, got {self.coverage_random_n}")
        if self.coverage_mode not in ("full", "random_only"):
            raise ValueError(f"coverage_mode must be one of ['full', 'random_only'], got {self.coverage_mode}")
        if self.coverage_enable and self.coverage_mode == "random_only" and self.coverage_random_n <= 0:
            raise ValueError("coverage_mode=random_only requires coverage_random_n > 0 when coverage_enable=True.")
        explicit_vals = [
            self.explicit_block1_noise_min,
            self.explicit_block1_noise_max,
            self.explicit_block2_noise_min,
            self.explicit_block2_noise_max,
        ]
        enabled_flags = [v >= 0.0 for v in explicit_vals]
        if any(enabled_flags) and not all(enabled_flags):
            raise ValueError(
                "explicit_block1/2_noise_min/max must be provided together, or all left negative to disable."
            )
        for name, value in (
            ("explicit_block1_noise_min", self.explicit_block1_noise_min),
            ("explicit_block1_noise_max", self.explicit_block1_noise_max),
            ("explicit_block2_noise_min", self.explicit_block2_noise_min),
            ("explicit_block2_noise_max", self.explicit_block2_noise_max),
        ):
            if value >= 0.0 and not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")
        if self.explicit_block1_noise_min >= 0.0 and self.explicit_block1_noise_min > self.explicit_block1_noise_max:
            raise ValueError("explicit_block1_noise_min cannot be greater than explicit_block1_noise_max")
        if self.explicit_block2_noise_min >= 0.0 and self.explicit_block2_noise_min > self.explicit_block2_noise_max:
            raise ValueError("explicit_block2_noise_min cannot be greater than explicit_block2_noise_max")


@dataclass
class LLaDA2TrainingArguments(TrainingArguments):
    beta1: float = field(default=0.9, metadata={"help": "AdamW optimizer beta1."})
    beta2: float = field(default=0.999, metadata={"help": "AdamW optimizer beta2"})
    dual_bd_mode: bool = field(
        default=False,
        metadata={"help": "If train in dual-BD mode (student/teacher, block layout). Used by dual_bd_distill scripts."},
    )
    block_diffusion_mode: bool = field(
        default=False,
        metadata={"help": "If train naive BD in block_diffusion mode (single noisy+clean concat, block diagonal mask). Used by train_llada2_bd."},
    )
    block_size: int = field(default=32, metadata={"help": "The block size for block diffusion block size"})
    same_token_labels: bool = field(
        default=False,
        metadata={"help": "If use same token location labels. True: no shift, False: use next-token prediction shift."},
    )
    ce_weight: float = field(default=1.0, metadata={"help": "Weight for CE loss."})
    kl_weight: float = field(default=0.0, metadata={"help": "Weight for KL loss (distillation)."})
    kl_temperature: float = field(default=1.0, metadata={"help": "Distillation temperature for KL loss (default 1.0)."})
    normalize_loss_weights: bool = field(
        default=False,
        metadata={"help": "If True, normalize ce_weight and kl_weight to sum to 1 (alpha-style mixing)."},
    )
    use_distill: bool = field(default=False, metadata={"help": "If True, use distillation loss."})
    wandb_online: bool = field(default=True, metadata={"help": "If True, use online wandb."})
    log_steps: int = field(
        default=1,
        metadata={"help": "Logging frequency (every N steps). Only wandb/train_metrics use this; tqdm still updates every step."},
    )
    save_hf_weights_per_epoch: bool = field(
        default=False,
        metadata={
            "help": "If True, after each epoch export HuggingFace weights under "
            "{output_dir}/epoch_{n}_step_{global_step}_hf_ckpt/. "
            "Reuses DCP from save_epochs when it ran this epoch; otherwise saves DCP once then converts."
        },
    )
    # Early stopping
    early_stopping_enabled: bool = field(
        default=False,
        metadata={"help": "If True, monitor total/ce/kl loss and stop when any rises or all stagnate."},
    )
    early_stopping_min_steps: int = field(default=200, metadata={"help": "Do not trigger early stop before this step."})
    early_stopping_patience: int = field(default=30, metadata={"help": "Stop within this many steps after condition."})
    early_stopping_rise_ratio: float = field(default=0.08, metadata={"help": "Loss increase ratio vs recent min to count as rise (e.g. 0.08 = 8%%)."})
    early_stopping_rise_window: int = field(default=50, metadata={"help": "Compare current loss to min over last N steps."})
    early_stopping_stagnation_steps: int = field(default=40, metadata={"help": "No improvement for this many steps = stagnate."})
    early_stopping_stagnation_tol_ratio: float = field(default=0.002, metadata={"help": "Relative change below this = stagnate."})
    # Loss weight schedule (warmup -> hold -> decay)
    loss_weight_schedule_ce: bool = field(default=False, metadata={"help": "If True, apply schedule to ce_weight."})
    loss_weight_schedule_kl: bool = field(default=False, metadata={"help": "If True, apply schedule to kl_weight."})
    loss_weight_schedule_warmup_steps: int = field(default=100, metadata={"help": "Warmup steps (0 -> 1.0)."})
    loss_weight_schedule_hold_steps: int = field(default=500, metadata={"help": "Hold at 1.0 for this many steps after warmup."})
    loss_weight_schedule_decay_ratio: float = field(default=0.5, metadata={"help": "Final multiplier after decay (e.g. 0.5 -> weight halves)."})
    loss_weight_schedule_decay_steps: int = field(default=1000, metadata={"help": "Decay over this many steps after hold."})
    loss_weight_schedule_normalize: bool = field(
        default=True,
        metadata={"help": "If True, after schedule normalize ce_weight_eff + kl_weight_eff = 1."},
    )
    loss_weight_schedule_style: Literal["step", "loss_balance", "step_then_balance", "fix_then_balance"] = field(
        default="step",
        metadata={"help": "step: warmup/hold/decay by step; loss_balance: weight by loss EMA; step_then_balance: step until warmup+hold then loss_balance; fix_then_balance: keep preset ce/kl weights for fix_steps then loss_balance."},
    )
    loss_weight_schedule_fix_steps: int = field(
        default=500,
        metadata={"help": "For fix_then_balance: keep ce_weight/kl_weight fixed for this many steps, then switch to loss_balance."},
    )
    loss_weight_schedule_ema_momentum: float = field(
        default=0.9,
        metadata={"help": "EMA momentum for ce/kl loss in loss_balance (ema = momentum*ema + (1-momentum)*current)."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.ce_weight < 0 or self.kl_weight < 0:
            raise ValueError(
                f"ce_weight and kl_weight must be >= 0, got ce_weight={self.ce_weight}, kl_weight={self.kl_weight}"
            )
        if self.normalize_loss_weights:
            s = float(self.ce_weight + self.kl_weight)
            if s <= 0:
                raise ValueError(
                    f"normalize_loss_weights is enabled, but ce_weight + kl_weight <= 0 "
                    f"(ce_weight={self.ce_weight}, kl_weight={self.kl_weight}, sum={s})"
                )
            self.ce_weight = float(self.ce_weight) / s
            self.kl_weight = float(self.kl_weight) / s
            logger.info_rank0(f"Normalized ce_weight and kl_weight to sum to 1, ce_weight={self.ce_weight}, kl_weight={self.kl_weight}")
        if self.kl_weight == 0.0:
            self.use_distill = False
            logger.info_rank0("KL weight is 0.0, distillation is disabled.")
        else:
            self.use_distill = True
            logger.info_rank0("KL weight is not 0.0, distillation is enabled.")


@dataclass
class Arguments:
    model: LLaDA2ModelArguments = field(default_factory=LLaDA2ModelArguments)
    data: LLaDA2DataArguments = field(default_factory=LLaDA2DataArguments)
    train: LLaDA2TrainingArguments = field(default_factory=LLaDA2TrainingArguments)
