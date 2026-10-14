"""Tiny baseline language model; intentionally not a Transformer yet."""
from __future__ import annotations

import torch
from torch import nn


class TinyLanguageModel(nn.Module):
    """Embedding + feed-forward baseline for next-token logits."""

    def __init__(self, vocab_size: int, embedding_dim: int = 128) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, vocab_size),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return logits with shape (batch, sequence, vocab_size)."""
        embedded = self.embedding(token_ids)
        return self.network(embedded)
