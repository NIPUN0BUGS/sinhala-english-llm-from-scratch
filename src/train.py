"""Minimal training entry point for the starter model."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import NextTokenDataset
from src.model import TinyLanguageModel
from src.tokenizer import CharacterTokenizer

ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "raw" / "train.txt"
CHECKPOINT_PATH = ROOT / "checkpoints" / "tiny_model.pt"


def main() -> None:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Training corpus not found: {CORPUS_PATH}")

    text = CORPUS_PATH.read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer().fit(text)
    token_ids = tokenizer.encode(text)
    context_length = min(64, max(1, len(token_ids) // 4))
    dataset = NextTokenDataset(token_ids, context_length=context_length)
    loader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)

    model = TinyLanguageModel(tokenizer.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    epochs = 10

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        batches = 0
        for inputs, targets in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, tokenizer.vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        mean_loss = total_loss / max(1, batches)
        print(f"Epoch {epoch + 1:02d}/{epochs} - loss: {mean_loss:.4f}")

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": tokenizer.vocab_size,
            "embedding_dim": 128,
            "context_length": context_length,
            "vocabulary": tokenizer._id_to_char,
        },
        CHECKPOINT_PATH,
    )
    print(f"Saved starter checkpoint to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
