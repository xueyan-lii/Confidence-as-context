from typing import Optional

import torch


def discretize_entropy(entropy: torch.Tensor) -> torch.Tensor:
    """Discretize entropy into reserved token IDs in the 128245..128255 range.

    Rounds natural-log entropy to nearest integer and maps:
      0 -> 128245, 1 -> 128246, ..., 9 -> 128254, >=10 -> 128255
    """
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


def discretize_softmax(p: torch.Tensor) -> torch.Tensor:
    """Discretize max softmax probabilities to reserved token IDs (128245..128255).

    Uses round(p * 10) as bin; 10 -> 128255, 0..9 -> 128245..128254.
    """
    int_softmax = torch.round(p * 10).to(dtype=torch.int)
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


def discretize_num(p: torch.Tensor) -> torch.Tensor:
    """Discretize probabilities to numeric token IDs (15..605 mapping used by code).

    Uses round(p * 10) as bin; maps 0..10 -> {15,16,..,24,605}.
    """
    int_softmax = torch.round(p * 10).to(dtype=torch.int)
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


def find_subsequence(haystack: torch.Tensor, needle: torch.Tensor) -> int:
    """Return index in haystack AFTER the first needle match ends, else -1.

    Both tensors are 1D; function is device-safe.
    """
    n_len = int(needle.shape[0])
    h_len = int(haystack.shape[0])
    if n_len == 0 or h_len == 0 or n_len > h_len:
        return -1
    needle = needle.to(haystack.device)
    rolling_window = haystack.unfold(0, n_len, 1)
    matches = torch.all(rolling_window == needle, dim=1)
    match_indices = torch.nonzero(matches, as_tuple=True)[0]
    if len(match_indices) == 0:
        return -1
    return (match_indices[0] + n_len).item()



