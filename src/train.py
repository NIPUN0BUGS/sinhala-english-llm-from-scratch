"""Train a small Sinhala-English causal Transformer locally."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import NextTokenDataset
from src.model import TinyLanguageModel
from src.tokenizer import CharacterTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_PATH = ROOT / "data" / "raw" / "train.txt"
DEFAULT_CHECKPOINT_PATH = ROOT / "checkpoints" / "tiny_transformer.pt"


def evaluate(
    model: TinyLanguageModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
            total_loss += float(loss)
            batches += 1
    model.train()
    return total_loss / max(1, batches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps must be positive when provided")
    if not args.corpus.exists():
        raise FileNotFoundError(f"Training corpus not found: {args.corpus}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Keep CPU training predictable on modest machines.
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cpu")

    text = args.corpus.read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer().fit(text)
    token_ids = tokenizer.encode(text)
    if len(token_ids) < 40:
        raise ValueError(
            "Corpus is too short for a train/validation split. Add more text first."
        )

    split_at = int(len(token_ids) * 0.9)
    train_ids, val_ids = token_ids[:split_at], token_ids[split_at:]
    context_length = min(args.context_length, max(4, len(train_ids) // 10))
    if len(train_ids) <= context_length or len(val_ids) <= context_length:
        raise ValueError(
            "Not enough tokens for this context length and validation split. "
            "Add more text or lower --context-length."
        )

    train_data = NextTokenDataset(train_ids, context_length=context_length)
    val_data = NextTokenDataset(val_ids, context_length=context_length)
    train_loader = DataLoader(
        train_data, batch_size=min(args.batch_size, len(train_data)), shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=min(args.batch_size, len(val_data)), shuffle=False
    )

    config = {
        "vocab_size": tokenizer.vocab_size,
        "embedding_dim": 96,
        "num_heads": 4,
        "num_layers": 2,
        "max_seq_len": context_length,
        "dropout": 0.1,
    }
    model = TinyLanguageModel(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Characters: {len(token_ids):,}; vocabulary: {tokenizer.vocab_size}; "
        f"train windows: {len(train_data):,}; validation windows: {len(val_data):,}"
    )
    import time

    started_at = time.perf_counter()
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
            global_step += 1
            if args.max_steps is not None and global_step >= args.max_steps:
                break

        train_loss = total_loss / max(1, batches)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | val loss {val_loss:.4f}"
        )
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    print(
        f"Optimizer steps: {global_step}; "
        f"elapsed seconds: {time.perf_counter() - started_at:.2f}"
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": config,
            "vocabulary": tokenizer.vocabulary,
            "unknown_token": tokenizer.unknown_token,
            "context_length": context_length,
            "training": {
                "epochs": args.epochs,
                "seed": args.seed,
                "corpus_path": str(args.corpus),
                "train_characters": len(train_ids),
                "validation_characters": len(val_ids),
            },
        },
        args.checkpoint,
    )
    print(f"Saved checkpoint: {args.checkpoint}")
    print(
        "Reminder: this sample corpus only verifies the pipeline; it is far too small "
        "to produce useful general-purpose chat responses."
    )


if __name__ == "__main__":
    main()
