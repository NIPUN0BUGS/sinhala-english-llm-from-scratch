"""Small decoder-only causal Transformer for CPU-friendly experiments."""
from __future__ import annotations

import torch
from torch import nn


class CausalTransformerLanguageModel(nn.Module):
    """Decoder-only Transformer with learned positions and a causal attention mask."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if embedding_dim < 1 or n_heads < 1 or n_layers < 1 or max_seq_len < 1:
            raise ValueError("model dimensions must be positive")
        if embedding_dim % n_heads:
            raise ValueError("embedding_dim must be divisible by n_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        batch_size, seq_len = token_ids.shape
        if seq_len < 1 or seq_len > self.max_seq_len:
            raise ValueError(f"sequence length must be between 1 and {self.max_seq_len}")
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        hidden = self.embedding_dropout(hidden)
        # Boolean True entries are masked: a position cannot attend to future tokens.
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=token_ids.device),
            diagonal=1,
        )
        hidden = self.blocks(hidden, mask=causal_mask, is_causal=True)
        return self.lm_head(self.final_norm(hidden))


# Backwards-compatible import name; new training uses the explicit class above.
TinyLanguageModel = CausalTransformerLanguageModel
