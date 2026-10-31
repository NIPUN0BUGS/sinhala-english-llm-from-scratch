"""CPU-friendly training entry point for the Sinhala-English language model.

This trainer keeps the repository's Safetensors-only checkpoint policy.  It
supports the current model implementation and records enough metadata to
reconstruct its character vocabulary safely.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import NextTokenDataset
from src.model import TinyLanguageModel
from src.safe_checkpoint import save_safetensors
from src.tokenizer import CharacterTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_PATH = ROOT / "data" / "raw" / "train.txt"
DEFAULT_CHECKPOINT_PATH = ROOT / "checkpoints" / "tiny_model.safetensors"
DEFAULT_METADATA_PATH = ROOT / "checkpoints" / "tiny_model.metadata.json"


def evaluate(
    model: TinyLanguageModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Return mean next-token loss without updating model parameters."""
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
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="UTF-8 corpus file; the final 10%% is used for validation.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=128,
        help="Must match the current TinyLanguageModel implementation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.context_length < 1:
        raise ValueError("epochs, batch-size, and context-length must be positive")
    if args.embedding_dim < 1:
        raise ValueError("--embedding-dim must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive when provided")
    if not args.corpus.is_file():
        raise FileNotFoundError(f"Training corpus not found: {args.corpus}")

    text = args.corpus.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Training corpus is empty: {args.corpus}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cpu")

    tokenizer = CharacterTokenizer().fit(text)
    token_ids = tokenizer.encode(text)
    if len(token_ids) < 40:
        raise ValueError("Corpus is too short for a train/validation split.")

    split_at = int(len(token_ids) * 0.9)
    train_ids, validation_ids = token_ids[:split_at], token_ids[split_at:]
    context_length = min(args.context_length, max(1, len(train_ids) // 10))
    if len(train_ids) <= context_length or len(validation_ids) <= context_length:
        raise ValueError("Not enough tokens for the selected context length and split.")

    train_data = NextTokenDataset(train_ids, context_length=context_length)
    validation_data = NextTokenDataset(validation_ids, context_length=context_length)
    train_loader = DataLoader(
        train_data,
        batch_size=min(args.batch_size, len(train_data)),
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=min(args.batch_size, len(validation_data)),
        shuffle=False,
    )

    model = TinyLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = nn.CrossEntropyLoss()

    best_validation_loss = float("inf")
    best_epoch = 0
    global_step = 0
    started_at = time.perf_counter()
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Device: {device}; parameters: "
        f"{sum(parameter.numel() for parameter in model.parameters()):,}"
    )
    print(
        f"Train tokens: {len(train_ids):,}; validation tokens: "
        f"{len(validation_ids):,}; vocabulary: {tokenizer.vocab_size}"
    )

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        batches = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(
                logits.reshape(-1, model.vocab_size),
                targets.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.detach())
            batches += 1
            global_step += 1
            if args.max_steps is not None and global_step >= args.max_steps:
                break

        train_loss = total_loss / max(1, batches)
        validation_loss = evaluate(
            model, validation_loader, criterion, device
        )
        scheduler.step(validation_loss)
        improved = validation_loss < best_validation_loss

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"validation loss {validation_loss:.4f}"
            + (" | new best" if improved else "")
        )

        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            save_safetensors(model.state_dict(), args.checkpoint)
            metadata = {
                "format": "safetensors",
                "model_type": "TinyLanguageModel",
                "vocab_size": tokenizer.vocab_size,
                "embedding_dim": args.embedding_dim,
                "context_length": context_length,
                "vocabulary": tokenizer._id_to_char,
                "unknown_token": tokenizer.unknown_token,
                "best_validation_loss": best_validation_loss,
                "best_epoch": best_epoch,
                "training": {
                    "epochs_requested": args.epochs,
                    "optimizer_steps": global_step,
                    "seed": args.seed,
                    "corpus_path": str(args.corpus),
                    "train_tokens": len(train_ids),
                    "validation_tokens": len(validation_ids),
                },
            }
            args.metadata.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        if args.max_steps is not None and global_step >= args.max_steps:
            break

    print(
        f"Optimizer steps: {global_step}; best epoch: {best_epoch}; "
        f"best validation loss: {best_validation_loss:.4f}; "
        f"elapsed seconds: {time.perf_counter() - started_at:.2f}"
    )
    print(f"Saved safe checkpoint: {args.checkpoint}")
    print(f"Saved tokenizer/model metadata: {args.metadata}")
    print(
        "Note: this checkpoint uses the repository's current feed-forward "
        "baseline; it is not a causal Transformer or a general-purpose chatbot."
    )


if __name__ == "__main__":
    main()
