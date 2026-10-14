"""Dataset utilities for next-character prediction."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset


class NextTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Return (input sequence, one-position-shifted target sequence) windows."""

    def __init__(self, token_ids: list[int], context_length: int = 64) -> None:
        if context_length < 1:
            raise ValueError("context_length must be at least 1")
        if len(token_ids) <= context_length:
            raise ValueError("Need more token IDs than context_length")
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        window = self.tokens[index : index + self.context_length + 1]
        return window[:-1], window[1:]
