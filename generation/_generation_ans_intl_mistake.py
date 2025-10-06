# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, List, Optional, Tuple

import torch
from torchtune.modules.transformer import TransformerDecoder
from torchtune.modules.common_utils import disable_kv_cache

import torchtune
from .common import discretize_entropy, discretize_softmax, discretize_num, find_subsequence


def _find_subsequence(haystack: torch.Tensor, needle: torch.Tensor) -> int:
    return find_subsequence(haystack, needle)


def multinomial_sample_one(probs: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Samples from a multinomial distribution."""
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.int)


def sample(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    q: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Generic sample from a probability distribution. Includes support for Top-K sampling
    and Temperature.

    Args:
        logits (torch.Tensor): logits from which to sample
        temperature (float): value to scale the predicted logits by, default 1.0.
        top_k (Optional[int]): If specified, we prune the sampling to only token ids within the top_k probabilities
        q (Optional[torch.Tensor]): randomly sampled tensor for softmax sampling trick. If None,
            we use the default softmax sampling trick. Default None.

    Example:
        >>> from torchtune.generation import sample
        >>> logits = torch.empty(3, 3).uniform_(0, 1)
        >>> sample(logits)
        tensor([[1],
                [2],
                [0]], dtype=torch.int32)

    Returns:
        torch.Tensor: sampled token id
    """
    # scale the logits based on temperature
    logits = logits / max(temperature, 1e-5)
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        # select the very last value from the top_k above as the pivot
        pivot = v.select(-1, -1).unsqueeze(-1)
        # set everything smaller than pivot value to inf since these
        # should be pruned
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    # change logits into probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # if q is None, we use the default softmax sampling trick
    if q is None:
        q = torch.empty_like(probs).exponential_(1)

    return multinomial_sample_one(probs, q)


def generate_next_token(
    model: TransformerDecoder,
    input_pos: torch.Tensor,
    x: torch.Tensor,
    q: Optional[torch.Tensor] = None,
    *,
    mask: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates the next tokens given a prompt, and also returns the corresponding logits.

    Args:
        model (TransformerDecoder): model used for generation
        input_pos (torch.Tensor): tensor with the positional encodings associated with the given prompt,
            with shape [bsz x seq_length].
        x (torch.Tensor): tensor with the token IDs associated with the given prompt,
            with shape [bsz x seq_length].
        q (Optional[torch.Tensor]): randomly sampled tensor for softmax sampling trick.
            See https://github.com/pytorch-labs/gpt-fast/blob/32971d3129541c5bfb4f715abc33d1c5f408d204/generate.py#L40
        mask (Optional[torch.Tensor]): attention mask with shape [bsz x seq_length x seq_length],
            default None.
        temperature (float): value to scale the predicted logits by, default 1.0.
        top_k (Optional[int]): Top-k value to use for sampling, default None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: tuple of two tensors:
            - tokens (torch.Tensor): tensor with the generated tokens,
                with shape [bsz x 1].
            - logits (torch.Tensor): tensor with the logits associated with the generated tokens,
                with shape [bsz x 1 x vocab_size].

    """
    # model produces logits in [bsz, seq_length, vocab_size]
    # we want to take the last token's logits as the input to the next model call
    logits = model(x, input_pos=input_pos, mask=mask)[:, -1]
    return (
        sample(logits.clone(), temperature=temperature, top_k=top_k, q=q),
        logits.unsqueeze(1),
    )


def update_stop_tokens_tracker(
    tokens: torch.Tensor, stop_tokens: torch.Tensor, stop_token_reached: torch.Tensor
) -> torch.Tensor:
    """Updates which sequences have reached a stop token."""
    # tokens: [bsz, 1]
    # stop_tokens: [num_stop_tokens]
    # stop_token_reached: [bsz]
    stop_token_reached_curr = torch.isin(tokens, stop_tokens).flatten()
    stop_token_reached |= stop_token_reached_curr
    return stop_token_reached


def get_causal_mask_from_padding_mask(
    padding_mask: torch.Tensor, target_seq_len: Optional[int] = None
) -> torch.Tensor:
    """
    Converts a padding mask of shape ``[bsz, seq_len]`` to a ``[bsz, seq_len, seq_len]`` causal attention mask suitable for
    consumption by :func:`~torch.nn.functional.scaled_dot_product_attention`. If ``target_seq_len``
    is provided, this will return a mask of shape ``[bsz, seq_len, target_seq_len]``. This is useful
    when generating masks for static KV caches where the maximum length the caches have been setup with
    are longer than the current sequence.

    Args:
        padding_mask (torch.Tensor): Boolean tensor where False indicates the corresponding token in the sequence
            is a padding token and should be masked out in attention, with shape [bsz x seq_length]
        target_seq_len (Optional[int]): target sequence length to create attention mask with. Default None.

    Returns:
        torch.Tensor: Boolean causal mask with shape
            - [bsz, seq_length, seq_length] or
            - [bsz, seq_length, target_seq_len] if ``target_seq_len`` was specified.

    Raises:
        AssertionError: if ``target_seq_len < seq_len``, the sequence length of the padding mask.

    Example:
        >>> padding_mask = torch.tensor([[False, True, True, True]])
        >>> get_causal_mask_from_padding_mask(padding_mask, target_seq_len=5)
        tensor([[[ True, False, False, False, False],
                  [False,  True, False, False, False],
                  [False,  True,  True, False, False],
                  [False,  True,  True,  True, False]]])
        ])
    """
    bsz, seq_len = padding_mask.shape
    target_seq_len = seq_len if target_seq_len is None else target_seq_len

    if target_seq_len < seq_len:
        raise AssertionError(
            "target_seq_len cannot be shorter than the sequence length of the padding mask."
        )

    mask = torch.tril(
        torch.ones(seq_len, target_seq_len, device=padding_mask.device, dtype=bool),
        diagonal=0,
    ).repeat(bsz, 1, 1)
    mask.narrow(2, 0, seq_len).mul_(padding_mask[:, None, :].expand(-1, seq_len, -1))
    mask.diagonal(dim1=1, dim2=2).copy_(torch.Tensor([True]))
    return mask


def get_position_ids_from_padding_mask(
    padding_mask: torch.Tensor,
):
    """
    Calculates position ids given a padding mask which right-shifts position ids to start
    from the first valid token.

    Args:
        padding_mask (torch.Tensor): Boolean tensor where False indicates the corresponding token in the sequence
            is a padding token and should be masked out in attention. Shape [bsz, seq_len]

    Returns:
        torch.Tensor: position ids which are appropriately shifted according to any padding values.

    Example:
        >>> padding_mask = torch.tensor([False, False, False, True, True, True, True, True])
        >>> get_position_ids_from_padding_mask(padding_mask)
        torch.Tensor([0, 0, 0, 0, 1, 2, 3, 4])
    """
    return ((padding_mask.cumsum(-1) - 1) * padding_mask).to(torch.int)

def discretize_entropy(entropy) -> torch.Tensor:
    """Discretize entropy into reserved token IDs."""
    int_entropy = torch.round(entropy).to(dtype=torch.int)
    result = torch.zeros_like(int_entropy)

    result = torch.where(int_entropy >= 10, 128255, result)
    result = torch.where(int_entropy == 9, 128254, result)
    result = torch.where(int_entropy == 8, 128253, result)
    result = torch.where(int_entropy == 7, 128252, result)
    result = torch.where(int_entropy == 6, 128251, result)
    result = torch.where(int_entropy == 5, 128250, result)
    result = torch.where(int_entropy == 4, 128249, result)
    result = torch.where(int_entropy == 3, 128248, result)
    result = torch.where(int_entropy == 2, 128247, result)
    result = torch.where(int_entropy == 1, 128246, result)
    result = torch.where(int_entropy == 0, 128245, result)
    
    return result

def discretize_softmax(softmax) -> torch.Tensor:
    """Discretize softmax into reserved token IDs."""
    int_softmax = torch.round(softmax * 10).to(dtype=torch.int)
    result = torch.zeros_like(int_softmax)
    
    result = torch.where(int_softmax >= 10, 128255, result)
    result = torch.where(int_softmax == 9, 128254, result)
    result = torch.where(int_softmax == 8, 128253, result)
    result = torch.where(int_softmax == 7, 128252, result)
    result = torch.where(int_softmax == 6, 128251, result)
    result = torch.where(int_softmax == 5, 128250, result)
    result = torch.where(int_softmax == 4, 128249, result)
    result = torch.where(int_softmax == 3, 128248, result)
    result = torch.where(int_softmax == 2, 128247, result)
    result = torch.where(int_softmax == 1, 128246, result)
    result = torch.where(int_softmax == 0, 128245, result)
        
    return result

def discretize_num(softmax) -> torch.Tensor:
    """Discretize softmax values token IDs."""
    int_softmax = torch.round(softmax * 10).to(dtype=torch.int)
    result = torch.zeros_like(int_softmax)

    result = torch.where(int_softmax >= 10, 605, result)
    result = torch.where(int_softmax == 9, 24, result)
    result = torch.where(int_softmax == 8, 23, result)
    result = torch.where(int_softmax == 7, 22, result)
    result = torch.where(int_softmax == 6, 21, result)
    result = torch.where(int_softmax == 5, 20, result)
    result = torch.where(int_softmax == 4, 19, result)
    result = torch.where(int_softmax == 3, 18, result)
    result = torch.where(int_softmax == 2, 17, result)
    result = torch.where(int_softmax == 1, 16, result)
    result = torch.where(int_softmax == 0, 15, result)
    
    return result

@torch.inference_mode()
def generate(
    model: TransformerDecoder,
    prompt: torch.Tensor,
    *,
    max_generated_tokens: int,
    pad_id: int = 0,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    stop_tokens: Optional[List[int]] = None,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor: # Return only generated tokens
    """
    Generates tokens for an answer, assuming the prompt contains Question + Marker + Reasoning.
    It first calculates confidence tokens for the reasoning part, interleaves them,
    and then generates the answer tokens sequentially without further interleaving.

    Args:
        model (TransformerDecoder): model used for generation.
        prompt (torch.Tensor): Input prompt containing Question + Marker + Reasoning. Shape [bsz, prompt_len].
        max_generated_tokens (int): Maximum number of ANSWER tokens to generate.
        pad_id (int): Padding token ID. Defaults to 0.
        temperature (float): Sampling temperature. Defaults to 1.0.
        top_k (Optional[int]): Top-k sampling cutoff. Defaults to None.
        stop_tokens (Optional[List[int]]): List of token IDs that signal stopping generation. Defaults to None.
        rng (Optional[torch.Generator]): Random number generator for sampling. Defaults to None.
        marker_sequence (Optional[List[int]]): Token sequence indicating the end of the question / start of reasoning.
                                                If None or not found, generation proceeds without interleaving.

    Returns:
        torch.Tensor: Generated tokens, including the interleaved prompt and the generated answer.
                     Shape [bsz, final_length].
    """
    
    prompt = prompt.view(1, -1) if prompt.ndim == 1 else prompt
    bsz, prompt_length = prompt.size()
    device = prompt.device
    marker_tensor = torch.tensor([36287, 220, 16, 25], device=device, dtype=prompt.dtype)
    incremental_decoding = model.caches_are_enabled()
    total_response_length = prompt_length + max_generated_tokens
    max_seq_len = (
        total_response_length
        if not incremental_decoding
        else model.decoder_max_cache_seq_len
    )

    q_lens = []
    r_lens = []

    for i in range(bsz):
        reas_start = _find_subsequence(prompt[i], marker_tensor)
        if reas_start == -1: # Marker not found
            print(f"Marker not found for sample {i}")
            print(prompt[i])
            reas_start = 0 # Treat everything before answer as question
            q_len = prompt_length
            r_len = 0
        else:
            q_len = reas_start
            r_len = prompt_length - reas_start

        q_lens.append(q_len)
        r_lens.append(r_len)
            
    num_layers = len(model.layers)
    model.output_hidden_states = [num_layers]
    with disable_kv_cache(model):
        logits_orig = model(tokens=prompt)[0]

    probs = torch.nn.functional.softmax(logits_orig, dim=-1)
    discretized = discretize_num(torch.max(probs, dim=-1)[0])
    #entropy = -torch.sum(probs * torch.where(probs > 0, torch.log(probs), torch.zeros_like(probs)), dim=-1)  # [b, seq_len]
    #discretized = self.discretize_entropy(entropy)  # [b, seq_len]
    max_new_len = max(q_lens) + max(r_lens)*2 
    input_samples = torch.full((bsz, max_new_len), pad_id, device=device)
    for i in range(bsz):
        q_len, r_len = q_lens[i], r_lens[i]
        input_samples[i, max_new_len - q_len - r_len*2 : max_new_len - r_len*2] = prompt[i, :q_len]
        interleaved_reasoning = torch.stack((prompt[i, q_len:], discretized[i, q_len:]), dim=-1).flatten()
        input_samples[i, max_new_len - r_len*2 :] = interleaved_reasoning
    curr_pos = max_new_len

    padding_masks = input_samples != pad_id
    if not padding_masks.all():
        padding_masks = torch.nn.functional.pad(
            padding_masks, (0, max_generated_tokens), value=True
        )
        masks = get_causal_mask_from_padding_mask(
            padding_masks, target_seq_len=max_seq_len
        )
        input_pos = get_position_ids_from_padding_mask(padding_masks)
        
    else:
        masks = torch.tril(
            torch.ones(
                total_response_length,
                max_seq_len,
                dtype=torch.bool,
                device=prompt.device,
            )
        ).unsqueeze(0)
        input_pos = torch.arange(
            0, total_response_length, device=generated_tokens.device
        ).unsqueeze(0)
        

    if incremental_decoding:
        curr_masks = masks[:, :curr_pos]
    else:
        curr_masks = masks[:, :curr_pos, :curr_pos]
    q = None
    if rng is not None:
        q = torch.empty(
            (bsz, model.tok_embeddings.num_embeddings), device=prompt.device
        ).exponential_(1, generator=rng)
    model.output_hidden_states = []
    tokens, generated_logits = generate_next_token(
        model,
        input_pos=input_pos[:, :curr_pos].squeeze(),
        mask=curr_masks,
        x=input_samples,
        temperature=temperature,
        top_k=top_k,
        q=q,
    )
    generated_tokens = torch.cat([input_samples, tokens], dim=-1)

    stop_token_reached = torch.zeros(bsz, dtype=torch.bool, device=prompt.device)
    if stop_tokens is not None:
        stop_tokens = torch.tensor(stop_tokens, device=prompt.device, dtype=tokens.dtype)

    stop_token_mask = torch.ones(
        (bsz, max_new_len + 1), dtype=torch.int32, device=prompt.device
    )

    if stop_tokens is not None:
        stop_token_reached = update_stop_tokens_tracker(
            tokens, stop_tokens, stop_token_reached
        )
        if stop_token_reached.all().item():
            return generated_tokens, None

    for _ in range(max_generated_tokens - 1):
        if stop_tokens is not None:
            stop_token_mask = torch.cat(
                [stop_token_mask, ~stop_token_reached.reshape(bsz, 1)], dim=-1
            )

        # if incremental decoding is enabled, we can use the current position
        # otherwise, we take the whole sequence up to the current position
        if incremental_decoding:
            curr_input_pos = input_pos[:, curr_pos].contiguous()
            curr_masks = masks[:, curr_pos, None, :].contiguous()
        else:
            tokens = generated_tokens.clone()
            curr_input_pos = input_pos[:, : curr_pos + 1]
            curr_masks = masks[:, : curr_pos + 1, : curr_pos + 1]

        q = None
        if rng is not None:
            q = torch.empty(
                (bsz, model.tok_embeddings.num_embeddings), device=prompt.device
            ).exponential_(1, generator=rng)
        tokens, logits = generate_next_token(
            model,
            input_pos=curr_input_pos,
            x=tokens.clone(),
            mask=curr_masks,
            temperature=temperature,
            top_k=top_k,
            q=q,
        )
        generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
        curr_pos += 1

        if stop_tokens is not None:
            stop_token_reached = update_stop_tokens_tracker(
                tokens, stop_tokens, stop_token_reached
            )
            if stop_token_reached.all():
                break

    # mask out generated tokens in seqs that already hit a stop token
    if stop_tokens is not None:
        # Create a mask where True indicates positions after a stop token
        mask_to_pad = ~stop_token_mask.bool()
        # Replace these positions with the padding token
        generated_tokens = torch.where(
            mask_to_pad, 
            torch.tensor(pad_id, device=generated_tokens.device, dtype=generated_tokens.dtype), 
            generated_tokens
        )
    generated_tokens = torch.cat((prompt, generated_tokens[:, max_new_len:]), dim=-1)
    return generated_tokens, None