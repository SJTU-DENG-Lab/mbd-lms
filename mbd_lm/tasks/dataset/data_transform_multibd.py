import torch
from typing import Any, Literal, Optional
from transformers import PreTrainedTokenizer
from veomni.utils import helper


logger = helper.create_logger(__name__)


def _sample_active_buffer_sizes(n_blocks: int, buffer_size: int, device: torch.device) -> list[int]:
    if n_blocks <= 0:
        return []
    if n_blocks == 1:
        return [1]
    sizes: list[int] = []
    used = 0
    while used < n_blocks:
        sampled = int(torch.randint(2, buffer_size + 1, (1,), device=device).item())
        if used + sampled >= n_blocks:
            remainder = n_blocks - used
            if remainder > 0:
                # Prefer avoiding singleton buffers; keep 1 only for unavoidable conflicts.
                if remainder == 1 and buffer_size >= 3:
                    repaired = False
                    for i, s in enumerate(sizes):
                        if s < buffer_size:
                            sizes[i] = s + 1
                            repaired = True
                            break
                    if not repaired:
                        for i, s in enumerate(sizes):
                            if s >= 3:
                                sizes[i] = s - 1
                                sizes.append(2)
                                repaired = True
                                break
                    if not repaired:
                        sizes.append(1)
                else:
                    sizes.append(remainder)
            break
        sizes.append(sampled)
        used += sampled
    if len(sizes) > 1:
        order = torch.randperm(len(sizes), device=device).tolist()
        sizes = [sizes[i] for i in order]
    return sizes


def _build_buffer_ids_from_sizes(
    *,
    block_sizes: list[int],
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    buffer_ids = []
    for gid, sz in enumerate(block_sizes):
        if sz <= 0:
            continue
        buffer_ids.append(torch.full((sz * block_size,), gid, device=device, dtype=torch.long))
    if buffer_ids:
        ids = torch.cat(buffer_ids, dim=0)
    else:
        ids = torch.empty((0,), device=device, dtype=torch.long)
    sizes = torch.tensor(block_sizes, device=device, dtype=torch.long)
    return ids, sizes


def _coverage_shifted_uniform_sizes(active_blocks: int, x: int, shift: int) -> list[int]:
    if active_blocks <= 0:
        return []
    prefix = min(max(shift, 0), active_blocks)
    sizes: list[int] = []
    if prefix > 0:
        sizes.append(prefix)
    remaining = active_blocks - prefix
    while remaining > 0:
        take = min(x, remaining)
        sizes.append(take)
        remaining -= take
    return sizes


def generate_coverage_active_buffer_sizes(
    *,
    active_blocks: int,
    buffer_size: int,
    random_n: int,
    coverage_mode: Literal["full", "random_only"] = "full",
    device: torch.device,
) -> list[list[int]]:
    """
    Build coverage layouts.
    - full: deterministic shifted-uniform groups (x in [2, buffer_size], shifts 0..x-1)
      plus random groups sampled by existing strategy, count=random_n.
    - random_only: only random groups sampled by existing strategy, count=random_n.
    """
    layouts: list[list[int]] = []
    if coverage_mode == "full":
        for x in range(2, buffer_size + 1):
            for shift in range(x):
                layouts.append(_coverage_shifted_uniform_sizes(active_blocks, x, shift))
    elif coverage_mode != "random_only":
        raise ValueError(f"Unsupported coverage_mode: {coverage_mode}")

    for _ in range(max(random_n, 0)):
        layouts.append(_sample_active_buffer_sizes(active_blocks, buffer_size, device))
    return layouts


def process_multi_bd_sft(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
    block_size: int,
    chat_align_block_size: Optional[int] = None,
    text_keys: str | list[str] = "messages",
    noise_range: tuple[float, float] = [(0.1, 0.3), (0.5, 0.8)],
    mask_token_id: int = 156895,
    buffer_size: int = 4,
    source_name: str | None = None,
    noise_transition_margin_ratio: float = 0.05,
    noise_transition_gamma: float = 2.0,
    buffer_internal_ramp: Literal["random", "linear", "linear_random", "chain_uniform", "chain_uniform_opt", "sorted_uniform", "linear_ascend"] = "linear_random",
    explicit_buffer_block_noise_ranges: Optional[list[list[float]]] = None,
    noise_schedule_mode: Literal["buffer_ramp", "d2f"] = "buffer_ramp",
    coverage_enable: bool = False,
    coverage_random_n: int = 0,
    coverage_mode: Literal["full", "random_only"] = "full",
    content_keep_trailing_pad_tokens: int = 0,
) -> list[dict[str, torch.Tensor]]:
    """
    block_size: Multi-BD blocks for noise, buffers, and student attention (student block granularity).
    chat_align_block_size: If set, used only for apply_chat_template_mdm padding alignment so the
        sequence length is a multiple of both student and teacher block sizes (typically lcm).
    """
    if isinstance(text_keys, str):
        messages = example[text_keys]
    elif isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                messages = example[key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    
    logger.info_rank0(f"Multi-BD Training: noise_range: {noise_range} -> {(noise_range[0][0], noise_range[1][1])}")
    noise_range = (noise_range[0][0], noise_range[1][1])
    
    examples = []
    if block_size > 1 and max_seq_len % block_size != 0:
        raise ValueError(f"max_seq_len ({max_seq_len}) must be divisible by block_size ({block_size})")
    if noise_schedule_mode != "d2f" and buffer_size <= 1:
        raise ValueError(
            f"buffer_size must be > 1 for Multi-BD distill, got {buffer_size}. "
            "Single-block buffers are explicitly disallowed."
        )
    if coverage_enable and noise_schedule_mode == "d2f":
        raise ValueError("coverage_enable is only supported with noise_schedule_mode=buffer_ramp.")
    if coverage_enable and buffer_size <= 1:
        raise ValueError("coverage_enable requires buffer_size > 1.")
    if coverage_enable and coverage_mode not in ("full", "random_only"):
        raise ValueError(f"Unsupported coverage_mode: {coverage_mode}")
    if coverage_enable and coverage_mode == "random_only" and coverage_random_n <= 0:
        raise ValueError(
            "coverage_mode=random_only requires coverage_random_n > 0 to produce non-empty layout variants."
        )
    align_bs = block_size if chat_align_block_size is None else int(chat_align_block_size)
    if align_bs > 1 and max_seq_len % align_bs != 0:
        raise ValueError(f"max_seq_len ({max_seq_len}) must be divisible by chat_align_block_size ({align_bs})")
    input_ids, prompt_length, content_length = apply_chat_template_mdm(
        messages=messages,
        tokenizer=tokenizer,
        max_length=max_seq_len,
        block_size=align_bs,
        content_keep_trailing_pad_tokens=content_keep_trailing_pad_tokens,
    )

    # Build block metadata for noisy-side schedule.
    if noise_schedule_mode == "d2f":
        position_ids, maskable_ids, base_buffer_ids, base_buffer_sizes = preprocess_input_token_ids_global_linear(
            input_ids, block_size, prompt_length, content_length=content_length
        )
        noise_ramp_style = "linear"
        layout_variants = [(base_buffer_ids, base_buffer_sizes)]
    else:
        position_ids = torch.arange(0, len(input_ids), device=input_ids.device)
        maskable_ids = position_ids >= prompt_length
        num_blocks = len(input_ids) // block_size
        active_len = int(content_length) if content_length is not None else len(input_ids)
        active_blocks = active_len // block_size
        tail_blocks = num_blocks - active_blocks

        if coverage_enable:
            layouts = generate_coverage_active_buffer_sizes(
                active_blocks=active_blocks,
                buffer_size=buffer_size,
                random_n=coverage_random_n,
                coverage_mode=coverage_mode,
                device=input_ids.device,
            )
            layout_variants = []
            for active_layout in layouts:
                block_sizes = list(active_layout)
                if tail_blocks > 0:
                    block_sizes.append(tail_blocks)
                ids, sizes = _build_buffer_ids_from_sizes(
                    block_sizes=block_sizes,
                    block_size=block_size,
                    device=input_ids.device,
                )
                layout_variants.append((ids, sizes))
        else:
            _, _, ids, sizes = preprocess_input_token_ids(
                input_ids, block_size, prompt_length, buffer_size, content_length=content_length
            )
            layout_variants = [(ids, sizes)]
        noise_ramp_style = buffer_internal_ramp

    pos = torch.arange(max_seq_len, device=input_ids.device)
    content_mask = pos < content_length

    if maskable_ids.numel() == input_ids.numel():
        maskable_mask = maskable_ids.to(dtype=torch.bool, device=input_ids.device) & content_mask
    else:
        maskable_mask = (pos >= prompt_length) & content_mask

    position_ids_full = torch.cat([position_ids, torch.arange(max_seq_len, device=input_ids.device)], dim=0)
    for variant_idx, (buffer_ids, buffer_sizes) in enumerate(layout_variants):
        noisy_input_ids, p_mask = multi_bd_sft_noise_transition(
            x_0=input_ids.clone(),
            noise_range=noise_range,
            maskable_mask=maskable_mask,
            mask_token_id=mask_token_id,
            block_size=block_size,
            buffer_sizes=buffer_sizes,
            noise_transition_margin_ratio=noise_transition_margin_ratio,
            noise_transition_gamma=noise_transition_gamma,
            buffer_internal_ramp=noise_ramp_style,
            explicit_buffer_block_noise_ranges=explicit_buffer_block_noise_ranges,
        )

        labels = input_ids.clone()
        labels[~maskable_mask] = -100
        labels[content_length:] = -100
        loss_mask = (noisy_input_ids == mask_token_id)
        labels[~loss_mask] = -100

        examples.append(
            {
                "input_ids": input_ids,
                "noisy_input_ids": noisy_input_ids,
                "attention_mask": torch.ones_like(noisy_input_ids),
                "gt_labels": labels,
                "position_ids": position_ids_full,
                "buffer_ids": buffer_ids,
                "buffer_sizes": buffer_sizes,
                "prompt_length": torch.tensor(prompt_length, device=input_ids.device, dtype=torch.long),
                "content_length": torch.tensor(content_length, device=input_ids.device, dtype=torch.long),
                "active_blocks": torch.tensor(content_length // block_size, device=input_ids.device, dtype=torch.long),
                "p_mask": p_mask,
                "coverage_variant_idx": torch.tensor(variant_idx, device=input_ids.device, dtype=torch.long),
            }
        )

    return examples


def multi_bd_sft_noise_transition(
    x_0: torch.Tensor,
    noise_range: tuple[float, float],
    maskable_mask: torch.Tensor,
    mask_token_id: int,
    block_size: int,
    buffer_sizes: torch.Tensor,
    noise_transition_margin_ratio: float = 0.02,
    noise_transition_gamma: float = 0.1,
    buffer_internal_ramp: Literal["random", "linear", "linear_random", "chain_uniform", "sorted_uniform"] = "random",
    explicit_buffer_block_noise_ranges: Optional[list[list[float]]] = None,
):
    """
    Per-buffer schedule: within each buffer mask ratio increases monotonically from low to high;
    buffers are independent (each buffer restarts from low).

    Parameters controlling block-to-block difference within a buffer:
      - buffer_internal_ramp: "random" = power-law random steps (noise_transition_gamma controls
        steepness); "linear" = deterministic linear low->high per buffer; "linear_random" = split
        [0,1] into n intervals, sample t uniformly in the i-th interval for block i, then
        prob = low + (effective_high - low) * t (monotonic but random within each segment).
        "chain_uniform" = sample a per-buffer floor L ~ Uniform(low, effective_high); block 0 has
        p ~ Uniform(L, effective_high); each following block uses previous p as lower bound and
        samples p ~ Uniform(prev_p, effective_high) (non-decreasing across blocks).
        "sorted_uniform" = sample n values from Uniform(low, effective_high) and sort ascending
        to form a monotonic per-buffer schedule.
      - noise_transition_margin_ratio: effective upper bound = high - this ratio * (high - low).

    Returns:
        (torch.Tensor, torch.Tensor): noisy sequence, per-position p_mask.
    """
    if x_0.dim() != 1:
        raise ValueError(f"x_0 must be 1D, got shape={tuple(x_0.shape)}")
    if maskable_mask.shape != x_0.shape:
        raise ValueError(f"maskable_mask shape {tuple(maskable_mask.shape)} != x_0 shape {tuple(x_0.shape)}")

    low, high = noise_range
    device = x_0.device
    maskable_mask = maskable_mask.to(device=device, dtype=torch.bool)

    x_0_blocks = torch.split(x_0, block_size)
    mask_blocks = torch.split(maskable_mask, block_size)
    num_blocks = len(x_0_blocks)

    def _fixed_ratio_sample_mask_block(block: torch.Tensor, block_maskable: torch.Tensor, ratio: float) -> torch.Tensor:
        valid_local_ids = torch.nonzero(block_maskable).squeeze(-1)
        num_valid_ids = valid_local_ids.numel()
        num_to_mask = int(num_valid_ids * ratio)
        perm = torch.randperm(num_valid_ids, device=device)
        mask_local_ids = valid_local_ids[perm[:num_to_mask]]
        new_block = block.clone()
        new_block[mask_local_ids] = mask_token_id
        return new_block

    range_size = high - low
    if range_size > 1e-9 and noise_transition_margin_ratio > 0:
        effective_high = high - noise_transition_margin_ratio * range_size
    else:
        effective_high = high

    explicit_ranges: list[tuple[float, float]] = []
    if explicit_buffer_block_noise_ranges:
        for idx, pair in enumerate(explicit_buffer_block_noise_ranges):
            if len(pair) != 2:
                raise ValueError(f"explicit_buffer_block_noise_ranges[{idx}] must be [min, max], got {pair}")
            lo_i, hi_i = float(pair[0]), float(pair[1])
            if not (0.0 <= lo_i <= hi_i <= 1.0):
                raise ValueError(
                    f"explicit_buffer_block_noise_ranges[{idx}] must satisfy 0.0 <= min <= max <= 1.0, got {pair}"
                )
            explicit_ranges.append((lo_i, hi_i))

    probs = []
    active_blocks = int(maskable_mask.view(-1, block_size).any(dim=1).sum().item())
    block_offset = 0
    for buf_size in buffer_sizes:
        n = int(buf_size.item())
        active_in_buffer = max(min(active_blocks - block_offset, n), 0)
        if explicit_ranges and active_in_buffer > 0:
            if active_in_buffer > len(explicit_ranges):
                raise ValueError(
                    f"explicit_buffer_block_noise_ranges provides {len(explicit_ranges)} entries, "
                    f"but encountered an active buffer with {active_in_buffer} blocks. "
                    "Provide at least one [min, max] range per block position inside the buffer."
                )
            for i in range(active_in_buffer):
                lo_i, hi_i = explicit_ranges[i]
                probs.append(float(lo_i + (hi_i - lo_i) * torch.rand(1, device=device).item()))
            for _ in range(n - active_in_buffer):
                probs.append(0.0)
        else:
            if buffer_internal_ramp == "linear":
                if n <= 1:
                    t_i = torch.rand(1, device=device).item()
                    probs.extend([float(low + (effective_high - low) * t_i)] * n)
                else:
                    t = torch.linspace(0.0, 1.0, n, device=device)
                    probs.extend((low + (effective_high - low) * t).tolist())
            elif buffer_internal_ramp == "linear_random":
                if n <= 1:
                    t_i = torch.rand(1, device=device).item()
                    probs.extend([float(low + (effective_high - low) * t_i)] * n)
                else:
                    for i in range(n):
                        t_i = (i + torch.rand(1, device=device).item()) / n
                        probs.append(float(low + (effective_high - low) * t_i))
            elif buffer_internal_ramp == "chain_uniform":
                global_lower = low + (effective_high - low) * torch.rand(1, device=device).item()
                cur_lower = global_lower
                for _ in range(n):
                    span = max(effective_high - cur_lower, 1e-9)
                    cur_prob = cur_lower + span * torch.rand(1, device=device).item()
                    probs.append(float(cur_prob))
                    cur_lower = cur_prob
            elif buffer_internal_ramp == "chain_uniform_opt":
                # Block 1: pure uniform in [low, effective_high]
                # Block i: uniform in [prev_prob, effective_high]
                cur_lower = low + (effective_high - low) * torch.rand(1, device=device).item()
                probs.append(float(cur_lower))
                for _ in range(n - 1):
                    span = max(effective_high - cur_lower, 1e-9)
                    cur_lower = cur_lower + span * torch.rand(1, device=device).item()
                    probs.append(float(cur_lower))
            elif buffer_internal_ramp == "sorted_uniform":
                if n > 0:
                    sampled = low + (effective_high - low) * torch.rand(n, device=device)
                    probs.extend(torch.sort(sampled).values.tolist())
            elif buffer_internal_ramp == "linear_ascend":
                # For small block_size (e.g. 4): pick a random lower-bound noise L,
                # then linearly ascend toward 1.0 with equal spacing.
                # step = (1.0 - L) / n, block i gets L + i * step.
                global_lower = low + (effective_high - low) * torch.rand(1, device=device).item()
                step = (1.0 - global_lower) / max(n, 1)
                for i in range(n):
                    probs.append(float(min(global_lower + i * step, 1.0)))
            else:
                cur_low = low
                for _ in range(n):
                    u = torch.rand(1, device=device).item()
                    span = max(effective_high - cur_low, 1e-9)
                    cur_prob = cur_low + span * (1.0 - (1.0 - u) ** noise_transition_gamma)
                    cur_low = cur_prob
                    probs.append(cur_prob)
        block_offset += n

    out_blocks = []
    p_mask = torch.ones_like(x_0, dtype=torch.float)
    for i in range(num_blocks):
        out_blocks.append(
            _fixed_ratio_sample_mask_block(x_0_blocks[i], mask_blocks[i], probs[i])
        )
        block_start = i * block_size
        block_end = block_start + len(x_0_blocks[i])
        p_mask[block_start:block_end][mask_blocks[i]] = probs[i]

    return torch.cat(out_blocks, dim=0), p_mask


def preprocess_input_token_ids(
    input_ids: torch.Tensor,
    block_size: int,
    prompt_length: int,
    buffer_size: int,
    content_length: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build per-token buffer ids for Multi-BD blocks.

    Active block count is computed from content_length (already block-aligned by chat template),
    so pure max_seq_len tail padding is excluded from random buffer-size sampling.
    Sampling strategy: draw sizes in [2, buffer_size], truncate last size on overflow, allow 1
    only as remainder-closing fallback, then shuffle sampled sizes.
    """
    noisy_input_ids = input_ids.clone()
    num_blocks = len(noisy_input_ids) // block_size
    if buffer_size <= 1:
        raise ValueError(f"buffer_size must be > 1, got {buffer_size}")
    active_len = int(content_length) if content_length is not None else len(noisy_input_ids)
    active_len = max(0, min(active_len, len(noisy_input_ids)))
    if active_len % block_size != 0:
        raise ValueError(
            f"active_len ({active_len}) must be divisible by block_size ({block_size})"
        )
    active_blocks = active_len // block_size
    tail_blocks = num_blocks - active_blocks
    
    sampled_sizes = _sample_active_buffer_sizes(active_blocks, buffer_size, input_ids.device)
    if tail_blocks > 0:
        # Exclude max_seq_len tail padding from active sampling, but still assign a dedicated tail buffer id.
        sampled_sizes.append(tail_blocks)
    position_ids = torch.arange(0, len(input_ids), device=input_ids.device)
    maskable_ids = position_ids >= prompt_length
    buffer_ids, buffer_sizes = _build_buffer_ids_from_sizes(
        block_sizes=sampled_sizes,
        block_size=block_size,
        device=input_ids.device,
    )
    return position_ids, maskable_ids, buffer_ids, buffer_sizes


def preprocess_input_token_ids_global_linear(
    input_ids: torch.Tensor,
    block_size: int,
    prompt_length: int,
    content_length: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    No-buffer baseline metadata:
    - one active group containing all content blocks (for global low->high schedule),
    - optional tail group for max_seq_len-only padding blocks.
    """
    num_blocks = len(input_ids) // block_size
    active_len = int(content_length) if content_length is not None else len(input_ids)
    active_len = max(0, min(active_len, len(input_ids)))
    if active_len % block_size != 0:
        raise ValueError(f"active_len ({active_len}) must be divisible by block_size ({block_size})")

    active_blocks = active_len // block_size
    tail_blocks = num_blocks - active_blocks

    group_sizes: list[int] = []
    if active_blocks > 0:
        group_sizes.append(active_blocks)
    if tail_blocks > 0:
        group_sizes.append(tail_blocks)
    if not group_sizes:
        group_sizes = [0]

    buffer_id_chunks = []
    for gid, size in enumerate(group_sizes):
        if size > 0:
            buffer_id_chunks.append(torch.full((size * block_size,), gid, device=input_ids.device, dtype=torch.long))
    buffer_ids = (
        torch.cat(buffer_id_chunks, dim=0)
        if buffer_id_chunks
        else torch.empty((0,), device=input_ids.device, dtype=torch.long)
    )

    position_ids = torch.arange(0, len(input_ids), device=input_ids.device)
    maskable_ids = position_ids >= prompt_length
    buffer_sizes = torch.tensor(group_sizes, device=input_ids.device, dtype=torch.long)
    return position_ids, maskable_ids, buffer_ids, buffer_sizes


def process_dual_bd_sft(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_seq_len: int,
    block_size: int,
    text_keys: str | list[str] = "messages",
    noise_range: tuple[float, float] = [(0.1, 0.3), (0.5, 0.8)],
    mask_token_id: int = 156895,
    source_name: str | None = None,
) -> list[dict[str, torch.Tensor]]:
    if isinstance(text_keys, str):
        messages = example[text_keys]
    elif isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                messages = example[key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
        
    examples = []
    input_ids, prompt_length, _content_length = apply_chat_template_mdm(
        messages=messages, tokenizer=tokenizer, max_length=max_seq_len, block_size=block_size,
    )

    # overlap input_ids for the noisy part of input_ids
    overlapped_input_ids, overlapped_position_ids, maskable_ids = overlap_input_token_ids(input_ids, block_size, prompt_length)

    if maskable_ids.numel() == overlapped_input_ids.numel():
        maskable_mask = maskable_ids.to(dtype=torch.bool, device=overlapped_input_ids.device)
    else:
        maskable_mask = overlapped_position_ids.to(device=overlapped_input_ids.device) >= prompt_length

    noisy_input_ids = dual_bd_sft_noise_transition(
        x_0=overlapped_input_ids.clone(),
        noise_range=noise_range,
        maskable_mask=maskable_mask,
        mask_token_id=mask_token_id,
        block_size=block_size,
    )

    labels = overlapped_input_ids.clone()
    labels[~maskable_mask] = -100  # no loss for non-maskable tokens (prompt tokens)
    loss_mask = (noisy_input_ids == mask_token_id)
    labels[~loss_mask] = -100  # no loss for not masked tokens 
    
    # cat overlapped_position_ids of noisy input_ids with position_ids of the clean input_ids
    position_ids = torch.cat([overlapped_position_ids, torch.arange(max_seq_len)], dim=0)

    examples.append(
        {
            "input_ids": input_ids,
            "noisy_input_ids": noisy_input_ids,
            "attention_mask": torch.ones_like(noisy_input_ids),
            "gt_labels": labels,
            "position_ids": position_ids,
        }
    )

    return examples


def overlap_input_token_ids(
    input_ids: torch.Tensor,
    block_size: int,
    prompt_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Overlap noisy token ids to get the overlapped blocks.
    """
    noisy_input_ids = input_ids.clone()
    blocks = torch.split(noisy_input_ids, block_size)
    
    overlapped_blocks = []
    position_ids_list = []
    maskable_ids_list = []
    
    position_ids_fn = lambda b_idx: torch.arange(b_idx * block_size, (b_idx + 1) * block_size)
    maskable_ids_fn = lambda b_idx: position_ids_fn(b_idx) >= prompt_length
    
    for b_idx in range(len(blocks) - 1):
        # add left block
        overlapped_blocks.append(blocks[b_idx])
        position_ids_list.append(position_ids_fn(b_idx))
        position_ids_list.append(position_ids_fn(b_idx + 1))
        
        # add right block
        overlapped_blocks.append(blocks[b_idx + 1])
        maskable_ids_list.append(maskable_ids_fn(b_idx))
        maskable_ids_list.append(maskable_ids_fn(b_idx + 1))
    
    overlapped_input_ids = torch.cat(overlapped_blocks, dim=0)
    overlapped_position_ids = torch.cat(position_ids_list, dim=0)
    overlapped_maskable_ids = torch.cat(maskable_ids_list, dim=0)
    return overlapped_input_ids, overlapped_position_ids, overlapped_maskable_ids


def dual_bd_sft_noise_transition(
    x_0: torch.Tensor,
    noise_range: list[tuple[float, float]] | tuple[tuple[float, float], tuple[float, float]],
    maskable_mask: torch.Tensor,
    mask_token_id: int,
    block_size: int,
):
    if x_0.dim() != 1:
        raise ValueError(f"x_0 must be 1D, got shape={tuple(x_0.shape)}")
    if maskable_mask.shape != x_0.shape:
        raise ValueError(f"maskable_mask shape {tuple(maskable_mask.shape)} != x_0 shape {tuple(x_0.shape)}")
    if len(noise_range) != 2:
        raise ValueError(f"noise_range must be [(low_min, low_max), (high_min, high_max)], got {noise_range}")

    left_range, right_range = noise_range

    device = x_0.device
    maskable_mask = maskable_mask.to(device=device, dtype=torch.bool)

    x_0_blocks = torch.split(x_0, block_size)
    mask_blocks = torch.split(maskable_mask, block_size)

    def _bernoulli_sample_mask_block(block: torch.Tensor, block_maskable: torch.Tensor, p: float) -> torch.Tensor:
        move_indices = (torch.rand(block.shape, device=device) < p) & block_maskable
        return torch.where(move_indices, torch.full_like(block, mask_token_id), block)
    
    def _coupled_sample_mask_block(block: torch.Tensor, block_maskable: torch.Tensor, p_left: float, p_right: float) -> torch.Tensor:
        random_probs = torch.rand(block.shape, device=device)
        mask_indices_left = (random_probs < p_left) & block_maskable
        mask_indices_right = (random_probs < p_right) & block_maskable
        block_left = torch.where(mask_indices_left, torch.full_like(block, mask_token_id), block)
        block_right = torch.where(mask_indices_right, torch.full_like(block, mask_token_id), block)
        return block_left, block_right
    
    def _fixed_ratio_sample_mask_block(block: torch.Tensor, block_maskable: torch.Tensor, ratio: float) -> torch.Tensor:
        valid_local_ids = torch.nonzero(block_maskable).squeeze(-1)
        num_valid_ids = valid_local_ids.numel()
        num_to_mask = int(num_valid_ids * ratio)
        
        perm = torch.randperm(num_valid_ids, device=device)
        mask_local_ids = valid_local_ids[perm[:num_to_mask]]
        
        new_block = block.clone()
        new_block[mask_local_ids] = mask_token_id
        return new_block

    out_blocks: list[torch.Tensor] = []
    num_blocks = len(x_0_blocks)
    b_idx = 0
    while b_idx < num_blocks - 1:
        p_left = (torch.rand(1, device=device) * (left_range[1] - left_range[0]) + left_range[0]).item()
        p_right = (torch.rand(1, device=device) * (right_range[1] - right_range[0]) + right_range[0]).item()

        out_blocks.append(_fixed_ratio_sample_mask_block(x_0_blocks[b_idx], mask_blocks[b_idx], p_left))
        out_blocks.append(_fixed_ratio_sample_mask_block(x_0_blocks[b_idx + 1], mask_blocks[b_idx + 1], p_right))
        
        b_idx += 2

    return torch.cat(out_blocks, dim=0)


def apply_chat_template_mdm(
    messages,
    tokenizer,
    max_length,
    block_size=None,
    content_keep_trailing_pad_tokens: int = 0,
):
    inputs_str = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_str = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

    prompt_ids_unpadded = tokenizer(prompt_str, add_special_tokens=False)['input_ids']
    prompt_length = min(len(prompt_ids_unpadded), max_length)

    raw_ids = tokenizer(
        inputs_str,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=False,
    ).input_ids.squeeze(0)

    real_length = len(raw_ids)
    active_end = min(
        real_length + max(int(content_keep_trailing_pad_tokens), 0),
        max_length,
    )

    # Block-alignment padding: participates in training (labels = pad_token_id)
    if block_size is not None and block_size > 1:
        pad_to_block = (block_size - active_end % block_size) % block_size
        content_length = active_end + pad_to_block
    else:
        content_length = active_end

    content_length = min(content_length, max_length)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.full((max_length,), pad_id, dtype=raw_ids.dtype)
    input_ids[:real_length] = raw_ids

    return input_ids, prompt_length, content_length
