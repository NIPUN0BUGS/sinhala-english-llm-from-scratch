"""Small decoder-only Transformer language model for CPU-first experiments."""
from __future__ import annotations

import torch
from torch import nn


class TinyLanguageModel(nn.Module):
    """A compact GPT-style causal Transformer that predicts the next character."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 96,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if max_seq_len < 1:
            raise ValueError("max_seq_len must be at least 1")

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return next-character logits shaped (batch, sequence, vocab_size)."""
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")
        _, sequence_length = token_ids.shape
        if sequence_length < 1:
            raise ValueError("Input sequence cannot be empty")
        if sequence_length > self.max_seq_len:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds max_seq_len "
                f"{self.max_seq_len}"
            )

        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)
        # In PyTorch's Transformer API, True marks positions that attention must mask.
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=token_ids.device,
            ),
            diagonal=1,
        )
        hidden = self.blocks(hidden, mask=causal_mask)
        return self.lm_head(self.final_norm(hidden))
