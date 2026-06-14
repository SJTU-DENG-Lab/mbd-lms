"""
Multi Block Diffusion (BD) attention mask utilities.

This module provides functions for generating attention masks for dual block diffusion
training, including both student and teacher mask generation.
"""

from typing import Tuple, Callable, Optional
import torch
from einops import rearrange


def _compute_multi_bd_params(seq_len: int, block_size: int) -> Tuple[int, int, int]:
    """
    Compute common parameters for multi BD mask generation.
    
    Args:
        seq_len: Sequence length
        block_size: Block size
        
    Returns:
        Tuple of (num_blocks, x_t_len, x_0_len)
    """
    num_blocks = seq_len // block_size
    x_t_len = seq_len
    x_0_len = seq_len
    return num_blocks, x_t_len, x_0_len


def multi_bd_attn_mask_student(
    batch_size: int,
    num_kv_heads: int,
    q_ids: torch.Tensor,
    kv_ids: torch.Tensor,
    buffer_ids: torch.Tensor,
    block_size: int,
    seq_len: int,
    active_blocks: Optional[int] = None,
    prompt_length: Optional[int] = None,
    content_length: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate attention mask for student model in dual block diffusion.
    
    Args:
        batch_size: Batch size (unused, kept for API compatibility)
        num_kv_heads: Number of KV heads (unused, kept for API compatibility)
        q_ids: Query position IDs, shape [q_len, 1]
        kv_ids: Key-Value position IDs, shape [1, kv_len]
        block_size: Block size
        seq_len: Sequence length
        
    Returns:
        Boolean mask tensor with shape [q_len, kv_len]
    """
    num_blocks, x_t_len, x_0_len = _compute_multi_bd_params(seq_len, block_size)
    
    if (x_t_len + x_0_len) != q_ids.shape[0]:
        raise ValueError(f"seq_len not match, x_t_len + x_0_len: {x_t_len + x_0_len}, q_ids.shape[0]: {q_ids.shape[0]}")
    
    x_0_flag_q = (q_ids >= x_t_len)
    x_0_flag_kv = (kv_ids >= x_t_len)
    
    x_0_block_mapping = torch.arange(num_blocks, dtype=torch.int32).repeat_interleave(block_size)
    block_mapping = torch.cat([x_0_block_mapping.clone(), x_0_block_mapping])
    
    block_mapping_q, block_group_mapping_q = [
        rearrange(mapping, "s -> s 1") for mapping in [block_mapping, buffer_ids]
    ]
    block_mapping_kv, block_group_mapping_kv = [
        rearrange(mapping, "s -> 1 s") for mapping in [block_mapping, buffer_ids]
    ]
    
    block_causal_template = (block_mapping_q >= block_mapping_kv)
    noisy_noisy = (~x_0_flag_q & ~x_0_flag_kv)
    noisy_clean = (~x_0_flag_q & x_0_flag_kv)
    clean_clean = (x_0_flag_q & x_0_flag_kv)

    if active_blocks is None:
        block_group_diagonal = (block_group_mapping_q == block_group_mapping_kv) & noisy_noisy
        offset_block_causal = (block_group_mapping_q > block_group_mapping_kv) & noisy_clean
        group_diagonal_inner_causal = block_causal_template & block_group_diagonal
        block_causal = block_causal_template & clean_clean
        return group_diagonal_inner_causal | offset_block_causal | block_causal

    # Mixed rule:
    # - response/content noisy region: buffer-aware multi-bd student rule
    # - prefix and global tail noisy region: block diagonal
    # - clean context interactions keep ordinary multi-bd offset/causal behavior
    active_blocks = int(active_blocks)
    active_blocks = max(0, min(active_blocks, num_blocks))
    if prompt_length is None and content_length is None and active_blocks is not None:
        prompt_length = 0
        content_length = int(active_blocks) * block_size
    prompt_length = int(prompt_length) if prompt_length is not None else 0
    content_length = int(content_length) if content_length is not None else x_t_len
    prompt_length = max(0, min(prompt_length, x_t_len))
    content_length = max(prompt_length, min(content_length, x_t_len))

    # Use block-level boundaries to avoid splitting one block into mixed rules.
    prompt_block = max(0, min(prompt_length // block_size, num_blocks))
    content_blocks = max(prompt_block, min(content_length // block_size, num_blocks))

    response_q = (~x_0_flag_q) & (block_mapping_q >= prompt_block) & (block_mapping_q < content_blocks)
    response_kv = (~x_0_flag_kv) & (block_mapping_kv >= prompt_block) & (block_mapping_kv < content_blocks)
    non_response_q = (~x_0_flag_q) & (~response_q)
    non_response_kv = (~x_0_flag_kv) & (~response_kv)

    # Response noisy-noisy: buffer-local inner causal
    response_group_diag = (block_group_mapping_q == block_group_mapping_kv) & response_q & response_kv
    response_noisy_noisy = block_causal_template & response_group_diag

    # Prefix/tail noisy-noisy: ordinary block diagonal
    non_response_noisy_noisy = (block_mapping_q == block_mapping_kv) & (non_response_q & non_response_kv)

    # Noisy(clean context) links:
    # - response noisy queries: keep group-based offset causal
    response_offset = (block_group_mapping_q > block_group_mapping_kv) & response_q & x_0_flag_kv
    # - prefix/tail noisy queries: ordinary block offset causal
    non_response_offset = (block_mapping_q > block_mapping_kv) & non_response_q & x_0_flag_kv

    # Clean-clean: ordinary block causal
    clean_causal = block_causal_template & clean_clean

    return response_noisy_noisy | non_response_noisy_noisy | response_offset | non_response_offset | clean_causal


def multi_bd_attn_mask_student_noisy_block_causal_only(
    batch_size: int,
    num_kv_heads: int,
    q_ids: torch.Tensor,
    kv_ids: torch.Tensor,
    block_size: int,
    seq_len: int
) -> torch.Tensor:
    """
    Student baseline mask:
    - only noisy-noisy (top-left) region is visible with block causal,
    - all cross-quadrant connections are masked out,
    - clean-clean is fully masked out.
    """
    num_blocks, x_t_len, x_0_len = _compute_multi_bd_params(seq_len, block_size)

    if (x_t_len + x_0_len) != q_ids.shape[0]:
        raise ValueError(f"seq_len not match, x_t_len + x_0_len: {x_t_len + x_0_len}, q_ids.shape[0]: {q_ids.shape[0]}")

    x_0_flag_q = (q_ids >= x_t_len)
    x_0_flag_kv = (kv_ids >= x_t_len)

    noisy_q = ~x_0_flag_q
    noisy_kv = ~x_0_flag_kv

    # Compute block ids directly from absolute positions so shapes match full [2*seq_len, 2*seq_len].
    q_block = (q_ids.clamp(max=x_t_len - 1) // block_size).to(torch.int32)
    kv_block = (kv_ids.clamp(max=x_t_len - 1) // block_size).to(torch.int32)

    # top-left noisy-noisy block-causal only
    noisy_block_causal = (q_block >= kv_block) & noisy_q & noisy_kv
    return noisy_block_causal


def multi_bd_attn_mask_teacher(
    batch_size: int,
    num_kv_heads: int,
    q_ids: torch.Tensor,
    kv_ids: torch.Tensor,
    block_size: int,
    seq_len: int
) -> torch.Tensor:
    """
    Generate attention mask for teacher model in dual block diffusion.
    
    Args:
        batch_size: Batch size (unused, kept for API compatibility)
        num_kv_heads: Number of KV heads (unused, kept for API compatibility)
        q_ids: Query position IDs, shape [q_len, 1]
        kv_ids: Key-Value position IDs, shape [1, kv_len]
        block_size: Block size
        seq_len: Sequence length
        
    Returns:
        Boolean mask tensor with shape [q_len, kv_len]
    """
    num_blocks, x_t_len, x_0_len = _compute_multi_bd_params(seq_len, block_size)
    
    x_0_flag_q = (q_ids >= x_t_len)
    x_0_flag_kv = (kv_ids >= x_t_len)
    
    x_0_block_mapping = torch.arange(num_blocks, dtype=torch.int32).repeat_interleave(block_size)
    block_mapping = torch.cat([x_0_block_mapping.clone(), x_0_block_mapping])
    
    block_mapping_q = rearrange(block_mapping, "s -> s 1")
    block_mapping_kv = rearrange(block_mapping, "s -> 1 s")
    
    block_diagonal = (
        (block_mapping_q == block_mapping_kv) 
        & (~x_0_flag_q & ~x_0_flag_kv)
    )
    offset_block_causal = (block_mapping_q > block_mapping_kv) & (x_0_flag_kv & ~x_0_flag_q)
    block_causal = (block_mapping_q >= block_mapping_kv) & (x_0_flag_kv & x_0_flag_q)
    
    return block_diagonal | offset_block_causal | block_causal


def multi_bd_attn_mask_generator(
    mask_flag_fn: Callable,
    enable_mixed_precision: bool
) -> torch.Tensor:
    """
    Generate attention mask prototype from a mask flag function.
    
    This function converts a boolean mask flag function into a float mask
    where False values are set to -inf and True values are set to 0.
    
    Args:
        mask_flag_fn: Function that returns a boolean mask tensor
        enable_mixed_precision: Whether to use float32 (True) or bfloat16 (False)
        
    Returns:
        Float mask tensor with shape [1, 1, q_len, kv_len]
    """
    multi_bd_attn_mask_flag = mask_flag_fn().unsqueeze(0).unsqueeze(0)
    multi_bd_attn_mask_prototype = torch.zeros_like(
        multi_bd_attn_mask_flag,
        dtype=torch.float32 if enable_mixed_precision else torch.bfloat16
    )
    multi_bd_attn_mask_prototype.masked_fill_(
        multi_bd_attn_mask_flag.logical_not(), float("-inf")
    )
    return multi_bd_attn_mask_prototype
