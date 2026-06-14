import torch
from typing import Any, Dict, List, Optional, Union, Tuple


def process_mdm_sft_example(
    example: Dict[str, Any],
    tokenizer,
    max_seq_len: int,
    text_keys: Union[str, List[str]] = "messages",
    noise_range: Tuple[float, float] = (0.3, 0.8),
    mask_token_id: int = 156895,
    block_size: Optional[int] = None,
    source_name: Optional[str] = None,
) -> List[Dict[str, "torch.Tensor"]]:
    if isinstance(text_keys, str):
        messages = example[text_keys]
    elif isinstance(text_keys, list):
        for key in text_keys:
            if key in example:
                messages = example[key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")

    examples = []
    input_ids, prompt_length, content_length = apply_chat_template_mdm(
        messages=messages, tokenizer=tokenizer, max_length=max_seq_len, block_size=block_size,
    )

    labels = input_ids.clone()
    labels[:prompt_length] = -100
    labels[content_length:] = -100

    pos = torch.arange(max_seq_len)
    maskable_mask = (pos >= prompt_length) & (pos < content_length)

    noisy_input_ids, p_mask = sft_noise_transition(
        input_ids.clone(),
        noise_range=noise_range,
        maskable_mask=maskable_mask,
        mask_token_id=mask_token_id,
    )

    loss_mask = noisy_input_ids == mask_token_id
    labels[~loss_mask] = -100

    examples.append({
        "input_ids": input_ids,
        "noisy_input_ids": noisy_input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "p_mask": p_mask,
    })

    return examples


def process_mdm_tokenized_example(
    example: Dict[str, List[int]],
    max_seq_len: int,
    text_keys: Union[str, List[str]] = "input_ids",
    noise_range: Tuple[float, float] = (0.3, 0.8),
    mask_token_id: int = 156895,
    source_name: Optional[str] = None,
) -> List[Dict[str, "torch.Tensor"]]:
    examples = []
    if isinstance(text_keys, str):
        input_ids = example[text_keys]
    elif isinstance(text_keys, list):
        for text_key in text_keys:
            if text_key in example:
                input_ids = example[text_key]
                break
        else:
            raise ValueError(f"None of the keys {text_keys} are found in the example.")
    else:
        raise ValueError(f"text_keys must be a string or a list of strings, but got {type(text_keys)}")

    prompt_length = example['prompt_lengths']

    input_ids = torch.tensor(input_ids)
    labels = input_ids.clone()
    labels[:prompt_length] = -100

    maskable_mask = torch.arange(max_seq_len) >= prompt_length

    noisy_input_ids, p_mask = sft_noise_transition(
        input_ids.clone(),
        noise_range=noise_range,
        maskable_mask=maskable_mask,
        mask_token_id=mask_token_id,
    )

    loss_mask = noisy_input_ids == mask_token_id
    labels[~loss_mask] = -100

    examples.append({
        "input_ids": input_ids,
        "noisy_input_ids": noisy_input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "p_mask": p_mask,
    })

    return examples


def sft_noise_transition(x_0, noise_range, maskable_mask, mask_token_id):
    """
    Performs a noise transition by masking tokens.

    Args:
        x_0 (torch.Tensor): The input sequence (seq_len,).
        noise_range (tuple): (min, max) for the noise range.
        maskable_mask (torch.Tensor): Boolean mask for positions allowed to be masked.
        mask_token_id (int): The ID of the mask token.

    Returns:
        (torch.Tensor, torch.Tensor): noisy sequence, per-position p_mask (noise probability).
    """
    t_tensor = torch.rand(1, device=x_0.device) * (noise_range[1] - noise_range[0]) + noise_range[0]
    sigma = t_tensor.item()
    move_chance = sigma
    move_indices = (torch.rand(x_0.shape, device=x_0.device) < move_chance) & maskable_mask
    if maskable_mask.any() and not move_indices.any():
        maskable_indices = torch.nonzero(maskable_mask, as_tuple=True)[0]
        forced_index = maskable_indices[torch.randint(len(maskable_indices), (1,), device=x_0.device).item()]
        move_indices[forced_index] = True
    x_t = torch.where(move_indices, mask_token_id, x_0)

    p_mask = torch.ones_like(x_0, dtype=torch.float)
    p_mask[maskable_mask] = sigma

    return x_t, p_mask


def apply_chat_template_mdm(messages, tokenizer, max_length, block_size=None):
    inputs_str = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_str = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)

    prompt_ids_unpadded = tokenizer(prompt_str, add_special_tokens=False)['input_ids']
    prompt_length = len(prompt_ids_unpadded)

    raw_ids = tokenizer(
        inputs_str,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
        add_special_tokens=False,
    ).input_ids.squeeze(0)

    real_length = len(raw_ids)

    # Block-alignment padding: participates in training (labels = pad_token_id)
    if block_size is not None and block_size > 1:
        pad_to_block = (block_size - real_length % block_size) % block_size
        content_length = real_length + pad_to_block
    else:
        content_length = real_length

    content_length = min(content_length, max_length)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids = torch.full((max_length,), pad_id, dtype=raw_ids.dtype)
    input_ids[:real_length] = raw_ids

    return input_ids, prompt_length, content_length
