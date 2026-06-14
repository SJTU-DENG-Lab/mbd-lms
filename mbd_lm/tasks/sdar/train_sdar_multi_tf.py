"""
Multi-BD Distillation V2 - Advanced Training with Configurable Temperature Scheduling and Confidence Loss.

Key Features:
1. KL temperature: teacher may schedule (linear/cosine/fixed/loss_aware); student fixed (kl_temperature_student_fixed)
2. Optional confidence loss to encourage flatter distributions on correct predictions
3. Features switchable via config flags

Usage (temperature):
    # Fixed teacher 0.3, student 1.0
    teacher_temperature_schedule_style=fixed kl_temperature_teacher_start=0.3 kl_temperature_student_fixed=1.0
    # Ramp teacher 4 -> 1, student fixed 1.0
    teacher_temperature_schedule_style=linear kl_temperature_teacher_start=4 kl_temperature_teacher_end=1
"""

import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal, Tuple, Optional, Callable

import wandb
import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import trange
from einops import rearrange
from torch.nn.attention.flex_attention import create_block_mask

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    build_dataloader,
    build_iterative_dataset,
    build_mapping_dataset,
)
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_tokenizer, save_model_assets, save_model_weights
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import parse_args, save_args, DataArguments, ModelArguments, TrainingArguments
from veomni.utils.device import (
    get_device_type,
    get_nccl_backend,
    get_torch_device,
    synchronize,
)
from veomni.utils.dist_utils import all_reduce
from veomni.models.registry import ModelRegistry

ModelRegistry.register_modeling_path("mbd_lm.models.sdar")

from mbd_lm.tasks.dataset.data_transform_multibd import process_multi_bd_sft
from mbd_lm.tasks.dataset import build_local_dataset
from mbd_lm.tasks.utils.multi_bd_mask import (
    multi_bd_attn_mask_student,
    multi_bd_attn_mask_student_noisy_block_causal_only,
    multi_bd_attn_mask_teacher,
    multi_bd_attn_mask_generator,
)
from mbd_lm.tasks.utils.loss_weight_scheduler import LossWeightScheduler
from mbd_lm.tasks.utils.early_stopping import EarlyStoppingMonitor
from mbd_lm.tasks.utils.distill_temperature import (
    DistillTemperatureController,
    interpolate_value,
)
from mbd_lm.tasks.utils.logit_diagnostics import (
    add_logit_diag,
    empty_logit_diag,
    finalize_logit_diag,
    summarize_masked_logits,
)


logger = helper.create_logger(__name__)


# ============================================================================
# Extended Arguments with Advanced Temperature and Confidence Loss Controls
# ============================================================================

@dataclass
class AdvancedModelArguments(ModelArguments):
    attn_implementation: Optional[Literal["eager", "sdpa", "flex_attention"]] = field(
        default="sdpa",
        metadata={"help": "Attention implementation to use."},
    )
    student_model_path: str = field(
        default="",
        metadata={"help": "Path to student model weights. Empty = use model_path."},
    )
    teacher_model_path: str = field(
        default="",
        metadata={"help": "Path to teacher model weights. Empty = use model_path (or student_model_path if set)."},
    )


@dataclass
class AdvancedDataArguments(DataArguments):
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
        default=lambda: [[0.1, 0.3], [0.5, 0.8]],
        metadata={"help": "Noise level for random flip input_ids to mask_ids"},
    )
    noise_range_dict: dict[str, float] = field(
        default=lambda: {"low_min": 0.1, "low_max": 0.3, "high_min": 0.5, "high_max": 0.8},
        metadata={"help": "Noise level for random flip input_ids to mask_ids"},
    )
    buffer_size: int = field(
        default=4,
        metadata={"help": "Buffer size for multi_bd (number of parallel block groups)."},
    )
    noise_transition_margin_ratio: float = field(
        default=0.05,
        metadata={"help": "Multi-BD noise transition: effective_high = high - margin_ratio * (high - low)"},
    )
    noise_transition_gamma: float = field(
        default=2.0,
        metadata={"help": "Multi-BD noise transition: power-law exponent (>1 biases each step toward effective_high)"},
    )
    buffer_internal_ramp: Literal["random", "linear", "linear_random", "chain_uniform", "chain_uniform_opt", "sorted_uniform", "linear_ascend"] = field(
        default="random",
        metadata={"help": "Multi-BD per-buffer ramp strategy."},
    )
    noise_schedule_mode: Literal["buffer_ramp", "d2f"] = field(
        default="buffer_ramp",
        metadata={
            "help": "Noise scheduling mode: buffer_ramp uses per-buffer ramps; "
            "d2f ignores random buffer partition and applies a global low->high linear schedule."
        },
    )
    coverage_enable: bool = field(
        default=False,
        metadata={
            "help": "If True, generate multiple buffer-layout variants for each sample "
            "(shifted uniform layouts + random layouts)."
        },
    )
    coverage_random_n: int = field(
        default=0,
        metadata={"help": "Number of additional random buffer layouts per sample when coverage_enable=True."},
    )
    coverage_mode: Literal["full", "random_only"] = field(
        default="full",
        metadata={"help": "Coverage layout mode: full=shifted deterministic + random; random_only=random only."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.coverage_random_n < 0:
            raise ValueError(f"coverage_random_n must be >= 0, got {self.coverage_random_n}")
        if self.coverage_mode not in ("full", "random_only"):
            raise ValueError(f"coverage_mode must be one of ['full', 'random_only'], got {self.coverage_mode}")
        if self.coverage_enable and self.coverage_mode == "random_only" and self.coverage_random_n <= 0:
            raise ValueError("coverage_mode=random_only requires coverage_random_n > 0 when coverage_enable=True.")
        if self.noise_schedule_mode != "d2f" and self.buffer_size <= 1:
            raise ValueError(
                f"data.buffer_size must be > 1 for Multi-BD distill (got {self.buffer_size}). "
                "buffer_size=1 degenerates to teacher-like single-block behavior."
            )
        self.noise_range = [
            [self.noise_range_dict["low_min"], self.noise_range_dict["low_max"]],
            [self.noise_range_dict["high_min"], self.noise_range_dict["high_max"]],
        ]


@dataclass
class AdvancedTrainingArguments(TrainingArguments):
    beta1: float = field(default=0.9, metadata={"help": "AdamW optimizer beta1."})
    beta2: float = field(default=0.999, metadata={"help": "AdamW optimizer beta2"})
    # Note: This is Multi-BD specific training script
    # Uses multi_bd_attn_mask with buffer_ids for parallel block decoding
    block_size: int = field(
        default=32,
        metadata={"help": "Student Multi-BD block size (data noise, buffers, student attention mask)."},
    )
    block_size_teacher: Optional[int] = field(
        default=None,
        metadata={
            "help": "Teacher Multi-BD block size for teacher attention mask only. "
            "None means same as block_size. Sequence layout is padded to lcm(block_size, block_size_teacher)."
        },
    )
    same_token_labels: bool = field(
        default=False,
        metadata={"help": "If use same token location labels. True: no shift, False: use next-token prediction shift."},
    )
    student_mask_mode: Literal["buffer_multi_bd", "noisy_block_causal_only"] = field(
        default="buffer_multi_bd",
        metadata={
            "help": "Student attention mask mode. buffer_multi_bd=original buffer-aware mask; "
            "noisy_block_causal_only=only top-left noisy block-causal is visible."
        },
    )
    
    # Basic loss weights
    ce_weight: float = field(default=0.0, metadata={"help": "Weight for CE loss. Set to 0 to disable."})
    kl_weight: float = field(default=1.0, metadata={"help": "Weight for KL loss (distillation)."})
    use_distill: bool = field(default=True, metadata={"help": "If True, use distillation loss."})
    
    # ============================================================================
    # KL distillation temperatures (teacher schedule + fixed student)
    # ============================================================================
    kl_temperature_teacher_start: float = field(
        default=1.0,
        metadata={"help": "Teacher softmax temperature at schedule start (also used when style=fixed)."},
    )
    kl_temperature_teacher_end: float = field(
        default=4.0,
        metadata={"help": "Teacher softmax temperature at schedule end (ignored when style=fixed)."},
    )
    teacher_temperature_schedule_style: Literal["fixed", "linear", "cosine", "loss_aware"] = field(
        default="linear",
        metadata={"help": "fixed=constant teacher_start; linear/cosine ramp start->end; loss_aware uses KL EMA."},
    )
    kl_temperature_student_fixed: float = field(
        default=1.0,
        metadata={"help": "Student softmax temperature (constant)."},
    )
    teacher_temperature_loss_ema_momentum: float = field(
        default=0.9,
        metadata={"help": "EMA momentum for loss-aware teacher temperature scheduling."},
    )
    scale_kl_by_temperature_product: bool = field(
        default=False,
        metadata={"help": "If True, multiply KL loss by current T_student * T_teacher to stabilize gradient magnitude."},
    )

    # ============================================================================
    # Feature 2: Confidence Loss (Switchable)
    # ============================================================================
    enable_confidence_loss: bool = field(
        default=False,
        metadata={"help": "If True, add confidence loss to encourage flatter distributions on correct predictions."},
    )
    confidence_beta: float = field(
        default=0.0,
        metadata={"help": "Weight for confidence loss. Typical values: 0.1-0.5."},
    )
    enable_confidence_beta_schedule: bool = field(
        default=False,
        metadata={"help": "If True, schedule confidence_beta from start to end during training."},
    )
    confidence_beta_start: float = field(
        default=0.3,
        metadata={"help": "Starting confidence_beta (used when enable_confidence_beta_schedule=True)."},
    )
    confidence_beta_end: float = field(
        default=0.05,
        metadata={"help": "Ending confidence_beta (used when enable_confidence_beta_schedule=True)."},
    )
    confidence_beta_schedule_style: Literal["linear", "cosine"] = field(
        default="linear",
        metadata={"help": "Interpolation style for confidence beta schedule."},
    )
    
    # Other training configs
    normalize_loss_weights: bool = field(default=False, metadata={"help": "Normalize ce_weight and kl_weight to sum to 1."})
    enable_logit_diagnostics: bool = field(
        default=False,
        metadata={"help": "Log lightweight student/teacher logit diagnostics (std, mean_abs, entropy, top1_conf) to wandb."},
    )
    wandb_online: bool = field(default=True, metadata={"help": "If True, use online wandb."})
    log_steps: int = field(default=1, metadata={"help": "Logging frequency."})
    save_hf_weights_per_epoch: bool = field(
        default=False,
        metadata={
            "help": "If True, after each epoch export HuggingFace weights under "
            "{output_dir}/epoch_{n}_step_{global_step}_hf_ckpt/. "
            "Reuses DCP from save_epochs when it ran this epoch; otherwise saves DCP once then converts."
        },
    )

    # Early stopping
    early_stopping_enabled: bool = field(default=False, metadata={"help": "Enable early stopping."})
    early_stopping_min_steps: int = field(default=200, metadata={"help": "Do not trigger early stop before this step."})
    early_stopping_patience: int = field(default=30, metadata={"help": "Stop within this many steps after condition."})
    early_stopping_rise_ratio: float = field(default=0.08, metadata={"help": "Loss increase ratio to count as rise."})
    early_stopping_rise_window: int = field(default=50, metadata={"help": "Compare current loss to min over last N steps."})
    early_stopping_stagnation_steps: int = field(default=40, metadata={"help": "No improvement for this many steps = stagnate."})
    early_stopping_stagnation_tol_ratio: float = field(default=0.002, metadata={"help": "Relative change below this = stagnate."})
    
    # Loss weight schedule
    loss_weight_schedule_ce: bool = field(default=False, metadata={"help": "Apply schedule to ce_weight."})
    loss_weight_schedule_kl: bool = field(default=False, metadata={"help": "Apply schedule to kl_weight."})
    loss_weight_schedule_warmup_steps: int = field(default=100, metadata={"help": "Warmup steps."})
    loss_weight_schedule_hold_steps: int = field(default=500, metadata={"help": "Hold at 1.0 for this many steps after warmup."})
    loss_weight_schedule_decay_ratio: float = field(default=0.5, metadata={"help": "Final multiplier after decay."})
    loss_weight_schedule_decay_steps: int = field(default=1000, metadata={"help": "Decay over this many steps after hold."})
    loss_weight_schedule_normalize: bool = field(default=True, metadata={"help": "Normalize after schedule."})
    loss_weight_schedule_style: Literal["step", "loss_balance", "step_then_balance", "fix_then_balance"] = field(
        default="step", metadata={"help": "Schedule style."}
    )
    loss_weight_schedule_fix_steps: int = field(default=500, metadata={"help": "Fixed steps before switching to loss_balance."})
    loss_weight_schedule_ema_momentum: float = field(default=0.9, metadata={"help": "EMA momentum for loss_balance."})

    def __post_init__(self):
        super().__post_init__()
        if self.ce_weight < 0 or self.kl_weight < 0:
            raise ValueError(f"ce_weight and kl_weight must be >= 0")
        if self.normalize_loss_weights:
            s = float(self.ce_weight + self.kl_weight)
            if s <= 0:
                raise ValueError(f"normalize_loss_weights enabled but ce_weight + kl_weight <= 0")
            self.ce_weight = float(self.ce_weight) / s
            self.kl_weight = float(self.kl_weight) / s
            logger.info_rank0(f"Normalized: ce_weight={self.ce_weight}, kl_weight={self.kl_weight}")
        
        if self.kl_temperature_teacher_start <= 0 or self.kl_temperature_teacher_end <= 0:
            raise ValueError("kl_temperature_teacher_start and kl_temperature_teacher_end must be > 0")
        if self.kl_temperature_student_fixed <= 0:
            raise ValueError("kl_temperature_student_fixed must be > 0")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")
        if self.block_size_teacher is not None and self.block_size_teacher <= 0:
            raise ValueError(f"block_size_teacher must be > 0 or null, got {self.block_size_teacher}")
        logger.info_rank0(
            f"KL temperatures: teacher {self.kl_temperature_teacher_start}->{self.kl_temperature_teacher_end} "
            f"({self.teacher_temperature_schedule_style}), student={self.kl_temperature_student_fixed}"
        )

        # Validate confidence loss settings
        if self.enable_confidence_loss:
            if self.confidence_beta < 0:
                raise ValueError(f"confidence_beta must be >= 0, got {self.confidence_beta}")
            if self.enable_confidence_beta_schedule:
                logger.info_rank0(
                    f"Confidence beta scheduling enabled: {self.confidence_beta_start} -> {self.confidence_beta_end}"
                )
            else:
                logger.info_rank0(f"Confidence loss enabled with fixed beta={self.confidence_beta}")

@dataclass
class Arguments:
    model: AdvancedModelArguments = field(default_factory=AdvancedModelArguments)
    data: AdvancedDataArguments = field(default_factory=AdvancedDataArguments)
    train: AdvancedTrainingArguments = field(default_factory=AdvancedTrainingArguments)


# ============================================================================
# Utility Functions
# ============================================================================

def compute_confidence_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate the average entropy of the output distribution at positions where the model predicts correctly.
    
    This loss encourages the model to have flatter (higher entropy) distributions on correct predictions,
    which helps prevent over-confidence and improves calibration.
    
    Args:
        logits: Raw output logits, shape (batch_size, seq_len, vocab_size)
        labels: Ground truth labels, shape (batch_size, seq_len). -100 indicates positions to ignore.
        
    Returns:
        Scalar tensor representing the confidence loss. Returns 0 if no correct predictions.
    """
    labels = labels.to(logits.device)
    
    valid_mask = (labels != -100)
    if not valid_mask.any():
        return logits.sum() * 0.0
    
    predicted_tokens = torch.argmax(logits, dim=-1)
    correct_mask = (predicted_tokens == labels) & valid_mask
    
    if correct_mask.sum() == 0:
        return logits.sum() * 0.0
    
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy_per_token = -torch.sum(probs * log_probs, dim=-1)
    
    entropy_at_correct_positions = entropy_per_token[correct_mask]
    confidence_loss = entropy_at_correct_positions.mean()
    
    return confidence_loss


def get_confidence_beta(args, step: int, total_steps: int) -> float:
    """
    Get confidence beta based on current step and configuration.
    
    Args:
        args: Training arguments
        step: Current training step
        total_steps: Total training steps
        
    Returns:
        Confidence beta value
    """
    if not args.train.enable_confidence_loss:
        return 0.0
    
    if args.train.enable_confidence_beta_schedule and total_steps > 0:
        ratio = step / total_steps
        beta = interpolate_value(
            args.train.confidence_beta_start,
            args.train.confidence_beta_end,
            ratio,
            args.train.confidence_beta_schedule_style
        )
    else:
        beta = args.train.confidence_beta

    return beta


def compute_self_normalized_ipw_losses(
    *,
    token_ce: torch.Tensor,
    labels: torch.Tensor,
    p_mask: Optional[torch.Tensor],
    fsdp_group,
    fsdp_world_size: int,
    micro_batch_count: int,
    token_kl: Optional[torch.Tensor] = None,
    kl_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """
    Compute self-normalized inverse-probability-weighted CE/KL losses.

    For masked token set M with per-token mask probability p_t, use:
        w_t = 1 / p_t
        L = sum_{t in M} w_t * loss_t / sum_{t in M} w_t

    To make backward under FSDP/DDP equivalent to the true global weighted loss,
    each rank backprops:
        world_size * local_num / global_den
    because gradients are averaged across ranks.
    """
    if labels.shape != token_ce.shape:
        raise ValueError(f"labels shape {tuple(labels.shape)} must match token_ce shape {tuple(token_ce.shape)}")
    if p_mask is not None and p_mask.shape != labels.shape:
        raise ValueError(f"p_mask shape {tuple(p_mask.shape)} must match labels shape {tuple(labels.shape)}")

    valid_mask = (labels != -100)
    weights = valid_mask.to(dtype=token_ce.dtype)
    if p_mask is not None:
        weights = weights / p_mask.to(dtype=token_ce.dtype).clamp(min=1e-6)

    den_local = weights.sum()
    if den_local.item() <= 0:
        zero = token_ce.sum() * 0.0
        return zero, zero, 0.0, 0.0

    ce_num_local = (token_ce * weights).sum()
    kl_num_local = (
        (token_kl * weights).sum()
        if token_kl is not None
        else token_ce.new_zeros(())
    )

    stats = torch.stack(
        [
            ce_num_local.detach(),
            kl_num_local.detach(),
            den_local.detach(),
        ]
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=fsdp_group)
    ce_num_global, kl_num_global, den_global = stats.unbind(0)
    den_global = den_global.clamp_min(1e-6)

    ce_loss = fsdp_world_size * ce_num_local / den_global.to(dtype=ce_num_local.dtype)
    kl_loss = fsdp_world_size * kl_num_local / den_global.to(dtype=kl_num_local.dtype)

    ce_loss = ce_loss / micro_batch_count
    kl_loss = kl_loss / micro_batch_count
    if token_kl is not None and kl_scale != 1.0:
        kl_loss = kl_loss * kl_scale

    ce_metric = (ce_num_global / den_global).item() / micro_batch_count
    kl_metric = (kl_num_global / den_global).item() / micro_batch_count
    if token_kl is not None and kl_scale != 1.0:
        kl_metric *= kl_scale

    return ce_loss, kl_loss, ce_metric, kl_metric


def compute_global_mean_from_token_values(
    *,
    token_values: torch.Tensor,
    valid_mask: torch.Tensor,
    fsdp_group,
    fsdp_world_size: int,
    micro_batch_count: int,
) -> tuple[torch.Tensor, float]:
    """Compute a globally correct mean over token_values[valid_mask]."""
    if token_values.shape != valid_mask.shape:
        raise ValueError(
            f"token_values shape {tuple(token_values.shape)} must match valid_mask shape {tuple(valid_mask.shape)}"
        )
    if not torch.any(valid_mask):
        zero = token_values.sum() * 0.0
        return zero, 0.0

    num_local = token_values.masked_select(valid_mask).sum()
    den_local = valid_mask.sum().to(dtype=token_values.dtype)
    stats = torch.stack([num_local.detach(), den_local.detach()])
    dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=fsdp_group)
    num_global, den_global = stats.unbind(0)
    den_global = den_global.clamp_min(1e-6)

    loss = fsdp_world_size * num_local / den_global.to(dtype=num_local.dtype)
    loss = loss / micro_batch_count
    metric = (num_global / den_global).item() / micro_batch_count
    return loss, metric


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    # =======================================================
    # Initialize process group and parallel state
    # =======================================================
    dist.init_process_group(backend=get_nccl_backend())
    args = parse_args(Arguments)
    if args.train.use_distill and args.train.kl_weight <= 0:
        logger.info_rank0(
            "KL distillation disabled because kl_weight <= 0; switching to CE-only SFT mode (student-only forward)."
        )
        args.train.use_distill = False
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    if args.train.global_rank == 0:
        save_args(args, args.train.output_dir)

    student_weights_path = args.model.student_model_path or args.model.model_path
    teacher_weights_path = args.model.teacher_model_path or student_weights_path
    logger.info_rank0(f"Resolved student weights path: {student_weights_path}")
    if args.train.use_distill:
        logger.info_rank0(f"Resolved teacher weights path: {teacher_weights_path}")

    Checkpointer = build_checkpointer(
        dist_backend=args.train.data_parallel_mode, 
        ckpt_manager=args.train.ckpt_manager
    )

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    # =======================================================
    # Prepare data
    # =======================================================
    logger.info_rank0("Prepare data")
    tokenizer = build_tokenizer(args.model.tokenizer_path)
    
    mask_token = "<|MASK|>"
    if mask_token not in tokenizer.get_vocab():
        special_tokens_dict = {"additional_special_tokens": [mask_token]}
        tokenizer.add_special_tokens(special_tokens_dict)
        logger.info_rank0(f"Added mask token {mask_token}, token_id={tokenizer.convert_tokens_to_ids(mask_token)}")
    mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)

    block_size_student = args.train.block_size
    block_size_teacher_eff = args.train.block_size_teacher or args.train.block_size
    max_sl = args.data.max_seq_len
    if max_sl % block_size_student != 0:
        raise ValueError(f"data.max_seq_len ({max_sl}) must be divisible by train.block_size ({block_size_student})")
    if max_sl % block_size_teacher_eff != 0:
        raise ValueError(
            f"data.max_seq_len ({max_sl}) must be divisible by teacher block size ({block_size_teacher_eff}); "
            "set train.block_size_teacher or adjust max_seq_len."
        )
    chat_align_bs = block_size_student
    if block_size_teacher_eff != block_size_student:
        chat_align_bs = (block_size_student * block_size_teacher_eff) // math.gcd(
            block_size_student, block_size_teacher_eff
        )
        if max_sl % chat_align_bs != 0:
            raise ValueError(
                f"max_seq_len must be divisible by lcm(student_block,teacher_block)={chat_align_bs}; got max_seq_len={max_sl}"
            )
        logger.info_rank0(
            f"Heterogeneous Multi-BD blocks: student={block_size_student}, teacher={block_size_teacher_eff}, "
            f"chat_template pad alignment={chat_align_bs} (lcm)"
        )
    
    if args.data.data_type == "conversation":
        if not tokenizer.chat_template:
            raise ValueError(f"No chat template found in the tokenizer.")

        transform = partial(
            process_multi_bd_sft,
            tokenizer=tokenizer,
            max_seq_len=args.data.max_seq_len,
            block_size=args.train.block_size,
            chat_align_block_size=chat_align_bs,
            text_keys=args.data.text_keys,
            noise_range=args.data.noise_range,
            mask_token_id=mask_token_id,
            buffer_size=args.data.buffer_size,
            noise_transition_margin_ratio=args.data.noise_transition_margin_ratio,
            noise_transition_gamma=args.data.noise_transition_gamma,
            buffer_internal_ramp=args.data.buffer_internal_ramp,
            noise_schedule_mode=args.data.noise_schedule_mode,
            coverage_enable=args.data.coverage_enable,
            coverage_random_n=args.data.coverage_random_n,
            coverage_mode=args.data.coverage_mode,
        )
    else:
        raise NotImplementedError(f"Unsupported data type: {args.data.data_type}")

    if args.data.dataloader_type == "native":
        if args.data.datasets_type == "iterable":
            logger.info_rank0("Start building iterative dataset")
            train_dataset = build_iterative_dataset(
                args.data.train_path, transform=transform, seed=args.train.seed
            )
        elif args.data.datasets_type == "mapping":
            logger.info_rank0("Start building mapping dataset")
            train_dataset = build_mapping_dataset(args.data.train_path, transform=transform)
        elif args.data.datasets_type == "local":
            logger.info_rank0("Start building local dataset")
            train_dataset = build_local_dataset(args.data.train_path, transform=transform)
        
        dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
        if args.data.datasets_type in ("mapping", "local"):
            dataset_length = dataset_length / args.train.data_parallel_size
        args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)

        train_dataloader = build_dataloader(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train.train_steps,
            rmpad=args.train.rmpad,
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
            dyn_bsz_margin=args.train.dyn_bsz_margin,
            dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor,
        )
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}")

    # =======================================================
    # Prepare student model
    # =======================================================
    logger.info_rank0("Prepare student model")
    student_model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=student_weights_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        attn_implementation=args.model.attn_implementation,
        moe_implementation=args.model.moe_implementation,
        init_device=args.train.init_device,
        force_use_huggingface=args.model.force_use_huggingface,
    )
    student_model_config = student_model.config
    
    if len(tokenizer) > student_model.get_input_embeddings().weight.shape[0]:
        logger.info_rank0(f"Resizing embeddings from {student_model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
        student_model.resize_token_embeddings(len(tokenizer))
    helper.print_device_mem_info("VRAM usage after building student model")

    get_optimizer_pre_hook = getattr(student_model, "get_optimizer_pre_hook", None)
    student_model = build_parallelize_model(
        student_model,
        init_device=args.train.init_device,
        weights_path=student_weights_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        basic_modules=student_model._no_split_modules + args.model.basic_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
        broadcast_model_weights_from_rank0=args.train.broadcast_model_weights_from_rank0
    )
    
    # =======================================================
    # Prepare teacher model
    # =======================================================
    if args.train.use_distill:
        logger.info_rank0("Prepare teacher model")
        teacher_model = build_foundation_model(
            config_path=args.model.config_path,
            weights_path=teacher_weights_path,
            torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
            attn_implementation=args.model.attn_implementation,
            moe_implementation=args.model.moe_implementation,
            init_device=args.train.init_device,
            force_use_huggingface=args.model.force_use_huggingface,
        )
        
        if len(tokenizer) > teacher_model.get_input_embeddings().weight.shape[0]:
            logger.info_rank0(f"Resizing teacher embeddings from {teacher_model.get_input_embeddings().weight.shape[0]} to {len(tokenizer)}")
            teacher_model.resize_token_embeddings(len(tokenizer))
        helper.print_device_mem_info("VRAM usage after building teacher model")
        
        teacher_model = build_parallelize_model(
            teacher_model,
            init_device=args.train.init_device,
            weights_path=teacher_weights_path,
            enable_full_shard=args.train.enable_full_shard,
            enable_mixed_precision=args.train.enable_mixed_precision,
            enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
            enable_fsdp_offload=args.train.enable_fsdp_offload,
            basic_modules=student_model._no_split_modules + args.model.basic_modules,
            enable_reentrant=args.train.enable_reentrant,
            enable_forward_prefetch=args.train.enable_forward_prefetch,
            broadcast_model_weights_from_rank0=args.train.broadcast_model_weights_from_rank0
        )
        teacher_model_config = teacher_model.config
    else:
        teacher_model = None
        teacher_model_config = None

    # =======================================================
    # Prepare optimizer and lr scheduler
    # =======================================================
    optimizer = build_optimizer(
        student_model,
        lr=args.train.lr,
        betas=(args.train.beta1, args.train.beta2),
        weight_decay=args.train.weight_decay,
        fused=True,
        optimizer_type=args.train.optimizer,
    )

    if get_optimizer_pre_hook is not None:
        optimizer_pre_hook = get_optimizer_pre_hook(student_model, student_model_config, args.train.data_parallel_mode)
        optimizer.register_step_pre_hook(optimizer_pre_hook)

    total_train_steps = args.train.train_steps * args.train.num_train_epochs
    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=total_train_steps,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    # =======================================================
    # Prepare wandb and save model assets
    # =======================================================
    if args.train.global_rank == 0:
        if args.train.use_wandb:
            wandb.init(
                project=args.train.wandb_project,
                name=args.train.wandb_name,
                config={**vars(args.model), **vars(args.data), **vars(args.train)},
                mode="online" if args.train.wandb_online else "offline",
            )

        model_assets = [student_model_config, tokenizer]
        save_model_assets(args.train.model_assets_dir, model_assets)

    # =======================================================
    # Prepare profiler
    # =======================================================
    if args.train.profile_this_rank:
        profiler = helper.create_profiler(
            start_step=args.train.profile_start_step,
            end_step=args.train.profile_end_step,
            trace_dir=args.train.profile_trace_dir,
            record_shapes=args.train.profile_record_shapes,
            profile_memory=args.train.profile_profile_memory,
            with_stack=args.train.profile_with_stack,
            global_rank=args.train.global_rank,
        )
        profiler.start()

    # =======================================================
    # Prepare environment meter and checkpoint path
    # =======================================================
    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=student_model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        empty_cache_steps=args.train.empty_cache_steps,
        enable_multisource=args.data.enable_multisource,
        dataloader=train_dataloader,
        data_path=args.data.train_path,
    )

    # =======================================================
    # Load checkpoint
    # =======================================================
    if args.train.load_checkpoint_path:
        student_state = {"model": student_model, "optimizer": optimizer, "extra_state": {}}
        Checkpointer.load(args.train.load_checkpoint_path, student_state)
        global_step = student_state["extra_state"]["global_step"]
        start_epoch = global_step // args.train.train_steps
        start_step = global_step % args.train.train_steps
        lr_scheduler.load_state_dict(student_state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(student_state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(student_state["extra_state"]["environ_meter"])
        torch.set_rng_state(student_state["extra_state"]["torch_rng_state"])
        if start_step == 0:
            iter(train_dataloader)

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    # =======================================================
    # Build Multi-BD attention mask
    # =======================================================
    multi_bd_full_len = 2 * args.data.max_seq_len

    def mask_flag_fn_generator(
        mask_flag_fn: Callable,
        buffer_ids: torch.Tensor = None,
        *,
        block_size: int,
        active_blocks: Optional[int] = None,
        prompt_length: Optional[int] = None,
        content_length: Optional[int] = None,
    ) -> Callable:
        if buffer_ids is None:
            fn = partial(
                mask_flag_fn,
                batch_size=args.train.global_batch_size,
                num_kv_heads=student_model_config.num_key_value_heads,
                q_ids=torch.arange(multi_bd_full_len)[:, None],
                kv_ids=torch.arange(multi_bd_full_len)[None, :],
                block_size=block_size,
                seq_len=args.data.max_seq_len,
            )
        else:
            bid = buffer_ids.squeeze(0) if buffer_ids.dim() > 1 else buffer_ids
            buffer_ids_full = torch.cat([bid, bid], dim=0)
            fn = partial(
                mask_flag_fn,
                batch_size=args.train.global_batch_size,
                num_kv_heads=student_model_config.num_key_value_heads,
                q_ids=torch.arange(multi_bd_full_len, device=bid.device)[:, None],
                kv_ids=torch.arange(multi_bd_full_len, device=bid.device)[None, :],
                buffer_ids=buffer_ids_full,
                block_size=block_size,
                seq_len=args.data.max_seq_len,
                active_blocks=active_blocks,
                prompt_length=prompt_length,
                content_length=content_length,
            )
        return fn
    
    student_multi_bd_attn_mask_prototype = None  # Sample-specific, generated per batch
    if args.train.use_distill:
        teacher_multi_bd_attn_mask_prototype = multi_bd_attn_mask_generator(
            mask_flag_fn_generator(multi_bd_attn_mask_teacher, block_size=block_size_teacher_eff),
            args.train.enable_mixed_precision
        )

    # =======================================================
    # Start training
    # =======================================================
    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, 
        args.train.enable_gradient_checkpointing, 
        args.train.activation_gpu_limit
    )
    
    if args.train.use_distill:
        student_model.train()
        teacher_model.eval()
    else:
        student_model.train()
    fsdp_group = get_parallel_state().fsdp_group
    fsdp_world_size = dist.get_world_size(group=fsdp_group)
        
    logger.info(
        f"rank{args.train.local_rank} Start training, train_steps: {args.train.train_steps}, "
        f"epochs: {args.train.num_train_epochs}, total_steps: {total_train_steps}"
    )
    if args.data.coverage_enable:
        if args.data.coverage_mode == "full":
            expected_variants_per_sample = sum(range(2, args.data.buffer_size + 1)) + max(args.data.coverage_random_n, 0)
        else:
            expected_variants_per_sample = max(args.data.coverage_random_n, 0)
    else:
        expected_variants_per_sample = 1
    logger.info_rank0(
        "[coverage-config] "
        f"coverage_enable={args.data.coverage_enable} "
        f"noise_schedule_mode={args.data.noise_schedule_mode} "
        f"buffer_size={args.data.buffer_size} "
        f"coverage_mode={args.data.coverage_mode} "
        f"coverage_random_n={args.data.coverage_random_n} "
        f"expected_variants_per_sample={expected_variants_per_sample}"
    )
    
    # Log enabled features
    if args.train.local_rank == 0:
        features = []
        if args.train.use_distill:
            features.append(
                f"kl_temp(Tt {args.train.kl_temperature_teacher_start}->{args.train.kl_temperature_teacher_end} "
                f"{args.train.teacher_temperature_schedule_style}, Ts={args.train.kl_temperature_student_fixed})"
            )
        if args.train.scale_kl_by_temperature_product:
            features.append("kl_scale(Ts*Tt)")
        if args.train.enable_confidence_loss:
            if args.train.enable_confidence_beta_schedule:
                features.append(f"conf_loss(beta={args.train.confidence_beta_start}->{args.train.confidence_beta_end})")
            else:
                features.append(f"conf_loss(beta={args.train.confidence_beta})")
        if args.train.enable_logit_diagnostics:
            features.append("logit_diag")
        if args.train.save_hf_weights_per_epoch:
            features.append("hf_ckpt_per_epoch")
        if features:
            logger.info_rank0(f"Enabled advanced features: {', '.join(features)}")
    
    early_stopping_monitor = None
    if args.train.early_stopping_enabled and args.train.global_rank == 0:
        early_stopping_monitor = EarlyStoppingMonitor(
            min_steps=args.train.early_stopping_min_steps,
            patience=args.train.early_stopping_patience,
            rise_ratio=args.train.early_stopping_rise_ratio,
            rise_window=args.train.early_stopping_rise_window,
            stagnation_steps=args.train.early_stopping_stagnation_steps,
            stagnation_tol_ratio=args.train.early_stopping_stagnation_tol_ratio,
        )
    early_stop_triggered = False
    
    loss_weight_scheduler = None
    if args.train.loss_weight_schedule_ce or args.train.loss_weight_schedule_kl:
        loss_weight_scheduler = LossWeightScheduler(args.train)
    temperature_controller = DistillTemperatureController(args.train)
        
    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        data_loader_tqdm = trange(
            args.train.train_steps,
            desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
            total=args.train.train_steps,
            initial=start_step,
            disable=args.train.local_rank != 0,
        )
        data_iterator = iter(train_dataloader)
        
        for _ in range(start_step, args.train.train_steps):
            global_step += 1

            # Get effective loss weights
            if loss_weight_scheduler is not None:
                ce_weight_eff, kl_weight_eff = loss_weight_scheduler.get_weights(global_step)
            else:
                ce_weight_eff = args.train.ce_weight
                kl_weight_eff = args.train.kl_weight

            # Get temperatures for this step
            teacher_temp, student_temp = temperature_controller.get(global_step, total_train_steps)
            
            # Get confidence beta for this step
            conf_beta = get_confidence_beta(args, global_step, total_train_steps)
            collect_logit_diag = args.train.enable_logit_diagnostics and (
                args.train.log_steps <= 0 or global_step % args.train.log_steps == 0
            )

            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            should_log_coverage = (global_step <= 3) or (
                args.train.log_steps > 0 and global_step % args.train.log_steps == 0
            )
            if should_log_coverage:
                micro_bsz_list: List[int] = []
                variant_counter: Counter[int] = Counter()
                mb_with_variant = 0
                p_mask_shapes = set()
                for mb in micro_batches:
                    micro_bsz_list.append(int(mb["input_ids"].shape[0]))
                    cov_idx = mb.get("coverage_variant_idx", None)
                    if isinstance(cov_idx, torch.Tensor):
                        vals = [int(v) for v in cov_idx.detach().view(-1).cpu().tolist()]
                    elif cov_idx is None:
                        vals = []
                    else:
                        vals = [int(cov_idx)]
                    if vals:
                        mb_with_variant += 1
                        variant_counter.update(vals)
                    pm = mb.get("p_mask", None)
                    if isinstance(pm, torch.Tensor):
                        p_mask_shapes.add(tuple(pm.shape))

                hist_preview = dict(sorted(variant_counter.items())[:8])
                logger.info_rank0(
                    "[coverage-runtime] "
                    f"step={global_step} "
                    f"num_micro={len(micro_batches)} "
                    f"micro_bsz={micro_bsz_list} "
                    f"total_local_samples={sum(micro_bsz_list)} "
                    f"mb_with_variant={mb_with_variant}/{len(micro_batches)} "
                    f"unique_variant_ids={len(variant_counter)} "
                    f"variant_hist_preview={hist_preview} "
                    f"p_mask_shapes={sorted(list(p_mask_shapes))}"
                )

            total_loss = 0
            total_ce_loss = 0
            total_kl_loss = 0
            total_conf_loss = 0
            total_student_diag = empty_logit_diag()
            total_teacher_diag = empty_logit_diag()
            
            synchronize()
            start_time = time.time()
            
            for micro_batch in micro_batches:
                environ_meter.add(micro_batch)
                if args.data.enable_multisource:
                    micro_batch.pop("ds_idx", None)
                    micro_batch.pop("source_name", None)
                
                student_micro_batch = {}
                teacher_micro_batch = {}
                
                # Multi-BD mode: process noisy+clean concatenated input with block masks
                position_ids = micro_batch["position_ids"]
                noisy_input_ids = micro_batch["noisy_input_ids"]
                clean_input_ids = micro_batch["input_ids"]
                batch_size = noisy_input_ids.shape[0]
                full_input_ids = torch.cat([noisy_input_ids, clean_input_ids], dim=1)
                
                def micro_batch_generator(attn_mask, *, num_attention_heads: int):
                    use_flex = args.model.attn_implementation == "flex_attention"
                    if use_flex:
                        device = get_device_type()
                        if attn_mask.dim() == 4:
                            mask_4d = (attn_mask == 0.0).to(device)
                            if mask_4d.shape[1] == 1:
                                boolean_3d = rearrange(mask_4d, "b 1 q k -> b q k")

                                def block_attn_mask_mod(batch_idx, head_idx, q_idx, kv_idx):
                                    return boolean_3d[batch_idx, q_idx, kv_idx]
                            else:
                                def block_attn_mask_mod(batch_idx, head_idx, q_idx, kv_idx):
                                    return mask_4d[batch_idx, head_idx, q_idx, kv_idx]
                        elif attn_mask.dim() == 3:
                            boolean_3d = attn_mask.to(device=device, dtype=torch.bool)

                            def block_attn_mask_mod(batch_idx, head_idx, q_idx, kv_idx):
                                return boolean_3d[batch_idx, q_idx, kv_idx]
                        elif attn_mask.dim() == 2:
                            boolean_2d = attn_mask.to(device=device, dtype=torch.bool)

                            def block_attn_mask_mod(batch_idx, head_idx, q_idx, kv_idx):
                                return boolean_2d[q_idx, kv_idx]
                        else:
                            raise ValueError(f"Unsupported attn_mask dim for flex path: {attn_mask.dim()}")
                        dev = f"{device}:{args.train.local_rank}"
                        block_mask = create_block_mask(
                            block_attn_mask_mod,
                            B=batch_size,
                            H=num_attention_heads,
                            Q_LEN=full_input_ids.shape[1],
                            KV_LEN=full_input_ids.shape[1],
                            device=dev,
                        )
                        attn_mask_out = block_mask
                    else:
                        attn_mask_out = attn_mask.expand(batch_size, -1, -1, -1)
                        if attn_mask_out.device.type != get_device_type():
                            attn_mask_out = attn_mask_out.to(get_device_type())
                    return {
                        "input_ids": full_input_ids,
                        "position_ids": position_ids,
                        "attention_mask": attn_mask_out,
                    }

                # Student mask
                if args.train.student_mask_mode == "noisy_block_causal_only":
                    student_multi_bd_attn_mask_prototype = multi_bd_attn_mask_generator(
                        mask_flag_fn_generator(
                            multi_bd_attn_mask_student_noisy_block_causal_only, block_size=block_size_student
                        ),
                        args.train.enable_mixed_precision,
                    )
                else:
                    # Sample-specific buffer-aware mask
                    buf_ids = micro_batch["buffer_ids"]
                    prompt_len_mb = micro_batch.get("prompt_length", None)
                    content_len_mb = micro_batch.get("content_length", None)
                    if buf_ids.dim() == 1 or buf_ids.shape[0] == 1:
                        active_blocks_mb = micro_batch.get("active_blocks", None)
                        active_blocks_val = (
                            int(active_blocks_mb.item())
                            if isinstance(active_blocks_mb, torch.Tensor) and active_blocks_mb.numel() == 1
                            else None
                        )
                        prompt_len_val = (
                            int(prompt_len_mb.item())
                            if isinstance(prompt_len_mb, torch.Tensor) and prompt_len_mb.numel() == 1
                            else None
                        )
                        content_len_val = (
                            int(content_len_mb.item())
                            if isinstance(content_len_mb, torch.Tensor) and content_len_mb.numel() == 1
                            else None
                        )
                        student_multi_bd_attn_mask_prototype = multi_bd_attn_mask_generator(
                            mask_flag_fn_generator(
                                multi_bd_attn_mask_student,
                                buffer_ids=buf_ids,
                                block_size=block_size_student,
                                active_blocks=active_blocks_val,
                                prompt_length=prompt_len_val,
                                content_length=content_len_val,
                            ),
                            args.train.enable_mixed_precision,
                        )
                    else:
                        active_blocks_mb = micro_batch.get("active_blocks", None)
                        student_masks = [
                            multi_bd_attn_mask_generator(
                                mask_flag_fn_generator(
                                    multi_bd_attn_mask_student,
                                    buffer_ids=buf_ids[b],
                                    block_size=block_size_student,
                                    active_blocks=(
                                        int(active_blocks_mb[b].item())
                                        if isinstance(active_blocks_mb, torch.Tensor)
                                        else None
                                    ),
                                    prompt_length=(
                                        int(prompt_len_mb[b].item())
                                        if isinstance(prompt_len_mb, torch.Tensor)
                                        else None
                                    ),
                                    content_length=(
                                        int(content_len_mb[b].item())
                                        if isinstance(content_len_mb, torch.Tensor)
                                        else None
                                    ),
                                ),
                                args.train.enable_mixed_precision,
                            )
                            for b in range(buf_ids.shape[0])
                        ]
                        student_multi_bd_attn_mask_prototype = torch.cat(student_masks, dim=0)
                
                if args.train.use_distill:
                    student_micro_batch = micro_batch_generator(
                        student_multi_bd_attn_mask_prototype,
                        num_attention_heads=student_model_config.num_attention_heads,
                    )
                    teacher_micro_batch = micro_batch_generator(
                        teacher_multi_bd_attn_mask_prototype,
                        num_attention_heads=teacher_model_config.num_attention_heads,
                    )
                else:
                    student_micro_batch = micro_batch_generator(
                        student_multi_bd_attn_mask_prototype,
                        num_attention_heads=student_model_config.num_attention_heads,
                    )
                    teacher_micro_batch = None

                micro_batch_checker = (
                    lambda micro_batch: {
                        k: v.to(get_device_type(), non_blocking=True) 
                        if isinstance(v, torch.Tensor) else v
                        for k, v in micro_batch.items()
                    }
                )
                
                if args.train.use_distill:
                    student_micro_batch = micro_batch_checker(student_micro_batch)
                    teacher_micro_batch = micro_batch_checker(teacher_micro_batch)
                else:
                    student_micro_batch = micro_batch_checker(student_micro_batch)
                    
                gt_labels = micro_batch.pop("gt_labels", None).to(get_device_type(), non_blocking=True)
                p_mask_tensor = micro_batch.pop("p_mask", None)
                if p_mask_tensor is not None:
                    p_mask_tensor = p_mask_tensor.to(get_device_type(), non_blocking=True)

                with model_fwd_context:
                    def model_forward_fn(model, micro_batch):
                        logits = model(**micro_batch, use_cache=False, output_router_logits=False).logits
                        # Extract noisy part (first half of concatenated sequence)
                        noisy_logits = logits[:, :noisy_input_ids.shape[1]].contiguous()
                        return noisy_logits
                    
                    if args.train.use_distill:
                        with torch.no_grad():
                            teacher_noisy_logits = model_forward_fn(teacher_model, teacher_micro_batch)

                    student_noisy_logits = model_forward_fn(student_model, student_micro_batch)

                    # CE loss
                    if args.train.same_token_labels:
                        token_ce = F.cross_entropy(
                            student_noisy_logits.view(-1, student_noisy_logits.shape[-1]),
                            gt_labels.view(-1),
                            reduction="none",
                        ).view_as(gt_labels)

                        if args.train.use_distill:
                            kl_student_logits = student_noisy_logits
                            kl_teacher_logits = teacher_noisy_logits
                            kl_labels = gt_labels
                            kl_p_mask = p_mask_tensor
                        diag_student_logits = student_noisy_logits
                        diag_labels = gt_labels
                        diag_teacher_logits = teacher_noisy_logits if args.train.use_distill else None
                    else:
                        shifted_student_logits = student_noisy_logits[:, :-1, :].contiguous()
                        shifted_labels = gt_labels[:, 1:].contiguous()

                        token_ce = F.cross_entropy(
                            shifted_student_logits.view(-1, shifted_student_logits.shape[-1]),
                            shifted_labels.view(-1),
                            reduction="none",
                        ).view(shifted_student_logits.shape[0], -1)

                        if args.train.use_distill:
                            shifted_teacher_logits = teacher_noisy_logits[:, :-1, :].contiguous()
                            kl_student_logits = shifted_student_logits
                            kl_teacher_logits = shifted_teacher_logits
                            kl_labels = shifted_labels
                            kl_p_mask = p_mask_tensor[:, 1:].contiguous() if p_mask_tensor is not None else None
                        diag_student_logits = shifted_student_logits
                        diag_labels = shifted_labels
                        diag_teacher_logits = shifted_teacher_logits if args.train.use_distill else None

                    # KL loss with separate temperatures
                    token_kl = None
                    if args.train.use_distill and kl_weight_eff > 0:
                        if teacher_temp <= 0 or student_temp <= 0:
                            raise ValueError(f"Temperatures must be > 0")

                        # Teacher: high temperature -> flat target distribution
                        p = F.softmax(kl_teacher_logits / teacher_temp, dim=-1)
                        # Student: scheduled temperature
                        log_q = F.log_softmax(kl_student_logits / student_temp, dim=-1)
                        
                        token_kl = F.kl_div(log_q, p, reduction="none").sum(dim=-1)
                        kl_scale = (teacher_temp * student_temp) if args.train.scale_kl_by_temperature_product else 1.0
                    else:
                        kl_scale = 1.0

                    ce_loss, kl_loss, ce_metric, kl_metric = compute_self_normalized_ipw_losses(
                        token_ce=token_ce,
                        labels=diag_labels,
                        p_mask=kl_p_mask if args.train.use_distill else (p_mask_tensor if args.train.same_token_labels else (p_mask_tensor[:, 1:].contiguous() if p_mask_tensor is not None else None)),
                        fsdp_group=fsdp_group,
                        fsdp_world_size=fsdp_world_size,
                        micro_batch_count=len(micro_batches),
                        token_kl=token_kl,
                        kl_scale=kl_scale,
                    )

                    # Confidence loss (optional)
                    conf_loss = ce_loss.new_zeros(())
                    conf_metric = 0.0
                    if args.train.enable_confidence_loss and conf_beta > 0:
                        valid_mask = (gt_labels != -100)
                        predicted_tokens = torch.argmax(student_noisy_logits, dim=-1)
                        correct_mask = (predicted_tokens == gt_labels) & valid_mask
                        log_probs = F.log_softmax(student_noisy_logits, dim=-1)
                        probs = torch.exp(log_probs)
                        entropy_per_token = -torch.sum(probs * log_probs, dim=-1)
                        conf_loss, conf_metric = compute_global_mean_from_token_values(
                            token_values=entropy_per_token,
                            valid_mask=correct_mask,
                            fsdp_group=fsdp_group,
                            fsdp_world_size=fsdp_world_size,
                            micro_batch_count=len(micro_batches),
                        )

                    if collect_logit_diag:
                        add_logit_diag(total_student_diag, summarize_masked_logits(diag_student_logits, diag_labels))
                        if args.train.use_distill and diag_teacher_logits is not None:
                            add_logit_diag(total_teacher_diag, summarize_masked_logits(diag_teacher_logits, diag_labels))

                    # Combined loss
                    if args.train.use_distill:
                        loss = ce_weight_eff * ce_loss + kl_weight_eff * kl_loss + conf_beta * conf_loss
                    else:
                        loss = ce_loss

                with model_bwd_context:
                    loss.backward()

                    loss_metric = ce_weight_eff * ce_metric + kl_weight_eff * kl_metric + conf_beta * conf_metric
                    total_loss += loss_metric
                    total_ce_loss += ce_metric
                    if args.train.use_distill:
                        total_kl_loss += kl_metric
                    if args.train.enable_confidence_loss:
                        total_conf_loss += conf_metric
                    
                del micro_batch

            # Gradient clipping
            if hasattr(student_model, "clip_grad_norm_"):
                _gn = student_model.clip_grad_norm_(args.train.max_grad_norm)
                grad_norm = _gn.item() if hasattr(_gn, "item") else float(_gn)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(student_model.parameters(), args.train.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # Collect metrics
            total_loss, total_ce_loss, total_kl_loss, total_conf_loss, grad_norm, \
                stu_count_pos, stu_count_vals, stu_sum, stu_sumsq, stu_abs_sum, stu_entropy_sum, stu_top1_conf_sum, \
                tea_count_pos, tea_count_vals, tea_sum, tea_sumsq, tea_abs_sum, tea_entropy_sum, tea_top1_conf_sum = all_reduce(
                (
                    total_loss, total_ce_loss, total_kl_loss, total_conf_loss, grad_norm,
                    total_student_diag["count_positions"], total_student_diag["count_values"], total_student_diag["sum"],
                    total_student_diag["sumsq"], total_student_diag["abs_sum"], total_student_diag["entropy_sum"], total_student_diag["top1_conf_sum"],
                    total_teacher_diag["count_positions"], total_teacher_diag["count_values"], total_teacher_diag["sum"],
                    total_teacher_diag["sumsq"], total_teacher_diag["abs_sum"], total_teacher_diag["entropy_sum"], total_teacher_diag["top1_conf_sum"],
                ),
                group=fsdp_group
            )
            synchronize()
            
            if loss_weight_scheduler is not None:
                loss_weight_scheduler.update_ema(total_ce_loss, total_kl_loss if args.train.use_distill else 0.0)
            if args.train.use_distill:
                temperature_controller.update_from_kl(total_kl_loss)

            reduced_student_diag = {
                "count_positions": stu_count_pos,
                "count_values": stu_count_vals,
                "sum": stu_sum,
                "sumsq": stu_sumsq,
                "abs_sum": stu_abs_sum,
                "entropy_sum": stu_entropy_sum,
                "top1_conf_sum": stu_top1_conf_sum,
            }
            reduced_teacher_diag = {
                "count_positions": tea_count_pos,
                "count_values": tea_count_vals,
                "sum": tea_sum,
                "sumsq": tea_sumsq,
                "abs_sum": tea_abs_sum,
                "entropy_sum": tea_entropy_sum,
                "top1_conf_sum": tea_top1_conf_sum,
            }
                
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            # Build progress bar postfix
            postfix_parts = [f"loss: {total_loss:.2f}", f"grad_norm: {grad_norm:.2f}", f"lr: {lr:.2e}"]
            if args.train.use_distill:
                postfix_parts.append(f"kl: {total_kl_loss:.3f}")
            if args.train.enable_confidence_loss:
                postfix_parts.append(f"conf: {total_conf_loss:.3f}")
            if args.train.use_distill:
                postfix_parts.append(f"T_t: {teacher_temp:.2f}, T_s: {student_temp:.2f}")
            data_loader_tqdm.set_postfix_str(", ".join(postfix_parts))
            data_loader_tqdm.update()

            # Early stopping
            if early_stopping_monitor is not None and args.train.global_rank == 0:
                should_stop, stop_reason = early_stopping_monitor.step(
                    global_step, total_loss, total_ce_loss, total_kl_loss
                )
                if should_stop:
                    early_stop_triggered = True
                    logger.info_rank0(f"Early stopping triggered: {stop_reason} at step {global_step}")
            if args.train.early_stopping_enabled:
                stop_tensor = torch.tensor(
                    1 if (args.train.global_rank == 0 and early_stop_triggered) else 0,
                    device=torch.device(f"{get_device_type()}:{args.train.local_rank}"),
                    dtype=torch.int32,
                )
                dist.broadcast(stop_tensor, src=0)
                if stop_tensor.item() == 1:
                    early_stop_triggered = True
                    break

            # Logging
            if args.train.global_rank == 0:
                if args.train.use_wandb and (args.train.log_steps <= 0 or global_step % args.train.log_steps == 0):
                    log_dict = {
                        "training/loss": total_loss,
                        "training/ce_loss": total_ce_loss,
                        "training/grad_norm": grad_norm,
                        "training/lr": lr,
                    }
                    if args.train.use_distill:
                        log_dict["training/kl_loss"] = total_kl_loss
                        log_dict["training/temperature_teacher"] = teacher_temp
                        log_dict["training/temperature_student"] = student_temp
                        log_dict["training/temperature_product"] = teacher_temp * student_temp
                    if collect_logit_diag:
                        log_dict.update({f"training/{k}": v for k, v in finalize_logit_diag("student", reduced_student_diag).items()})
                        if args.train.use_distill:
                            teacher_diag_metrics = finalize_logit_diag("teacher", reduced_teacher_diag)
                            log_dict.update({f"training/{k}": v for k, v in teacher_diag_metrics.items()})
                            teacher_std = teacher_diag_metrics.get("teacher_logit_std", 0.0)
                            student_std = log_dict.get("training/student_logit_std", 0.0)
                            teacher_entropy = teacher_diag_metrics.get("teacher_entropy", 0.0)
                            student_entropy = log_dict.get("training/student_entropy", 0.0)
                            if teacher_std > 0:
                                log_dict["training/student_teacher_logit_std_ratio"] = student_std / teacher_std
                            log_dict["training/student_teacher_entropy_gap"] = student_entropy - teacher_entropy
                    if args.train.enable_confidence_loss:
                        log_dict["training/confidence_loss"] = total_conf_loss
                        log_dict["training/confidence_beta"] = conf_beta
                    if loss_weight_scheduler is not None:
                        log_dict["training/ce_weight"] = ce_weight_eff
                        log_dict["training/kl_weight"] = kl_weight_eff
                        log_dict.update(loss_weight_scheduler.get_log_state(global_step))
                    if early_stopping_monitor is not None:
                        log_dict.update(early_stopping_monitor.get_log_state())
                    wandb.log(log_dict, step=global_step)

            # Profiling
            if args.train.profile_this_rank and global_step <= args.train.profile_end_step:
                profiler.step()
                if global_step == args.train.profile_end_step:
                    profiler.stop()

            # Checkpointing
            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": student_model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

        data_loader_tqdm.close()
        start_step = 0
        if early_stop_triggered:
            logger.info_rank0(f"Early stopping: exiting training after epoch {epoch + 1}, step {global_step}")
            break
        helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")

        dcp_dir_this_epoch: Optional[str] = None
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            dcp_dir_this_epoch = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": student_model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                },
            }
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            dist.barrier()
            save_checkpoint_path = dcp_dir_this_epoch
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

        if args.train.save_hf_weights_per_epoch:
            if dcp_dir_this_epoch is None:
                helper.empty_cache()
                dcp_dir_this_epoch = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": student_model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                dist.barrier()
                save_checkpoint_path = dcp_dir_this_epoch
                logger.info_rank0(
                    f"Distributed checkpoint saved for HF export at {save_checkpoint_path} successfully!"
                )
            if args.train.global_rank == 0:
                hf_dir = os.path.join(
                    args.train.output_dir, f"epoch_{epoch + 1}_step_{global_step}_hf_ckpt"
                )
                os.makedirs(hf_dir, exist_ok=True)
                model_state_dict = ckpt_to_state_dict(
                    save_checkpoint_path=dcp_dir_this_epoch,
                    output_dir=args.train.output_dir,
                    ckpt_manager=args.train.ckpt_manager,
                )
                save_model_weights(hf_dir, model_state_dict, model_assets=model_assets)
                logger.info_rank0(f"HuggingFace checkpoint saved at {hf_dir} successfully!")
            dist.barrier()

    synchronize()
    del optimizer, lr_scheduler
    helper.empty_cache()

    if (
        args.train.global_rank == 0
        and args.train.save_hf_weights
        and save_checkpoint_path is not None
        and not args.train.save_hf_weights_per_epoch
    ):
        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=save_checkpoint_path,
            output_dir=args.train.output_dir,
            ckpt_manager=args.train.ckpt_manager,
        )
        save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
        logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
