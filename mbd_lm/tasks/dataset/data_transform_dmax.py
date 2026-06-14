from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch

from mbd_lm.tasks.dataset.data_transform_multibd import process_multi_bd_sft


def process_mdm_sft_example_dmax_oput(
    example: Dict[str, Any],
    tokenizer,
    max_seq_len: int,
    text_keys: Union[str, List[str]] = "messages",
    noise_range: Tuple[float, float] = (0.75, 0.75),
    mask_token_id: int = 156895,
    source_name: Optional[str] = None,
) -> List[Dict[str, torch.Tensor]]:
    """DMax OPUT SFT transform.

    DMax differs from the normal BD transform in two ways:
      1. each row must carry a boolean ``flag`` selecting masked vs on-policy input;
      2. labels supervise the post-prompt answer span, not only masked positions.
    """
    del source_name

    messages = _get_messages(example, text_keys)
    if "flag" not in example:
        raise ValueError("DMax OPUT training expects each example to contain a boolean `flag` field.")

    input_ids, prompt_length = apply_chat_template_dmax(messages, tokenizer, max_seq_len)

    labels = input_ids.clone()
    labels[:prompt_length] = -100
    trim_trailing_pad_labels(labels, input_ids=input_ids, prompt_length=prompt_length, pad_id=_pad_id(tokenizer))

    maskable_mask = torch.arange(max_seq_len, device=input_ids.device) >= prompt_length
    noisy_input_ids = sft_noise_transition_dmax(
        input_ids.clone(),
        noise_range=noise_range,
        maskable_mask=maskable_mask,
        mask_token_id=mask_token_id,
    )

    return [
        {
            "input_ids": input_ids,
            "noisy_input_ids": noisy_input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
            "flag": torch.tensor(bool(example["flag"]), dtype=torch.bool),
        }
    ]


def process_multi_bd_sft_dmax_oput(
    example: Dict[str, Any],
    tokenizer,
    max_seq_len: int,
    block_size: int,
    chat_align_block_size: Optional[int] = None,
    text_keys: Union[str, List[str]] = "messages",
    noise_range: Tuple[Tuple[float, float], Tuple[float, float]] = ((0.1, 0.3), (0.5, 0.8)),
    mask_token_id: int = 156895,
    source_name: Optional[str] = None,
    buffer_size: int = 4,
    noise_transition_margin_ratio: float = 0.05,
    noise_transition_gamma: float = 2.0,
    buffer_internal_ramp: str = "linear_random",
    explicit_buffer_block_noise_ranges: Optional[List[List[float]]] = None,
    noise_schedule_mode: str = "buffer_ramp",
    coverage_enable: bool = False,
    coverage_random_n: int = 0,
    coverage_mode: Literal["full", "random_only"] = "full",
    keep_trailing_pad_tokens: int = 32,
) -> List[Dict[str, torch.Tensor]]:
    """DMax OPUT labels on top of the Multi-BD buffer/noise layout."""
    del source_name
    if "flag" not in example:
        raise ValueError("DMax OPUT Multi-BD training expects each example to contain a boolean `flag` field.")

    records = process_multi_bd_sft(
        example=example,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        block_size=block_size,
        chat_align_block_size=chat_align_block_size,
        text_keys=text_keys,
        noise_range=noise_range,
        mask_token_id=mask_token_id,
        buffer_size=buffer_size,
        noise_transition_margin_ratio=noise_transition_margin_ratio,
        noise_transition_gamma=noise_transition_gamma,
        buffer_internal_ramp=buffer_internal_ramp,
        explicit_buffer_block_noise_ranges=explicit_buffer_block_noise_ranges,
        noise_schedule_mode=noise_schedule_mode,
        coverage_enable=coverage_enable,
        coverage_random_n=coverage_random_n,
        coverage_mode=coverage_mode,
        content_keep_trailing_pad_tokens=keep_trailing_pad_tokens,
    )

    for record in records:
        labels = record["input_ids"].clone()
        prompt_length = int(record["prompt_length"].item())
        labels[:prompt_length] = -100
        trim_trailing_pad_labels(
            labels,
            input_ids=record["input_ids"],
            prompt_length=prompt_length,
            pad_id=_pad_id(tokenizer),
            keep_trailing_pad_tokens=keep_trailing_pad_tokens,
        )
        record["gt_labels"] = labels
        record["flag"] = torch.tensor(bool(example["flag"]), dtype=torch.bool)

    return records


def _get_messages(example: Dict[str, Any], text_keys: Union[str, List[str]]):
    if isinstance(text_keys, str):
        return example[text_keys]
    if isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                return example[key]
        raise ValueError(f"None of the keys {text_keys} are found in the example.")
    raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")


def _pad_id(tokenizer) -> int:
    return tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id


def apply_chat_template_dmax(messages, tokenizer, max_length: int) -> tuple[torch.Tensor, int]:
    inputs_str = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_str = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

    prompt_ids_unpadded = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
    prompt_length = min(len(prompt_ids_unpadded), max_length)

    raw_ids = tokenizer(
        inputs_str,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=False,
    ).input_ids.squeeze(0)

    real_length = min(len(raw_ids), max_length)
    input_ids = torch.full((max_length,), _pad_id(tokenizer), dtype=raw_ids.dtype)
    input_ids[:real_length] = raw_ids[:real_length]
    return input_ids, prompt_length


def trim_trailing_pad_labels(
    labels: torch.Tensor,
    *,
    input_ids: torch.Tensor,
    prompt_length: int,
    pad_id: int,
    keep_trailing_pad_tokens: int = 32,
) -> None:
    """Match DMax's tail handling: keep only the first 32 trailing pad/eos labels."""
    not_pad = input_ids != pad_id
    if torch.any(not_pad):
        last_not_pad = torch.nonzero(not_pad, as_tuple=False)[-1].item()
        run_start = last_not_pad + 1
    else:
        run_start = 0

    if run_start < input_ids.numel():
        start = max(run_start + keep_trailing_pad_tokens, prompt_length)
        if start < input_ids.numel():
            labels[start:] = -100


def sft_noise_transition_dmax(
    x_0: torch.Tensor,
    noise_range: Tuple[float, float],
    maskable_mask: torch.Tensor,
    mask_token_id: int,
) -> torch.Tensor:
    t_tensor = torch.rand(1, device=x_0.device) * (noise_range[1] - noise_range[0]) + noise_range[0]
    sigma = t_tensor.item()
    move_indices = (torch.rand(x_0.shape, device=x_0.device) < sigma) & maskable_mask
    return torch.where(move_indices, mask_token_id, x_0)
