"""Reproducible CPU-first training for the Sinhala-English causal Transformer."""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.dataset import NextTokenDataset
from src.model import CausalTransformerLanguageModel
from src.safe_checkpoint import save_safetensors
from src.tokenizer import CharacterTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_PATH = ROOT / "data" / "raw" / "train.txt"
DEFAULT_CHECKPOINT_PATH = ROOT / "checkpoints" / "tiny_transformer.safetensors"
DEFAULT_METADATA_PATH = ROOT / "checkpoints" / "tiny_transformer.metadata.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate(
    model: CausalTransformerLanguageModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
            token_count = targets.numel()
            total_loss += float(loss) * token_count
            total_tokens += token_count
    return total_loss / max(1, total_tokens)


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.batch_size, args.context_length, args.embedding_dim,
           args.heads, args.layers, args.patience, args.threads) < 1:
        raise ValueError("epochs, batch size, context, dimensions, patience and threads must be positive")
    if args.embedding_dim % args.heads:
        raise ValueError("--embedding-dim must be divisible by --heads")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    if not args.corpus.is_file():
        raise FileNotFoundError(f"Training corpus not found: {args.corpus}")
    text = args.corpus.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Training corpus is empty: {args.corpus}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    tokenizer = CharacterTokenizer().fit(text)
    token_ids = tokenizer.encode(text)
    if len(token_ids) < 20:
        raise ValueError("Corpus is too short; provide at least 20 characters.")
    split_at = int(len(token_ids) * 0.9)
    train_ids, validation_ids = token_ids[:split_at], token_ids[split_at:]
    if len(train_ids) < 2 or len(validation_ids) < 2:
        raise ValueError("Corpus is too short for a train/validation split.")
    context_length = min(args.context_length, len(train_ids) - 1, len(validation_ids) - 1)

    train_data = NextTokenDataset(train_ids, context_length=context_length)
    validation_data = NextTokenDataset(validation_ids, context_length=context_length)
    train_loader = DataLoader(
        train_data, batch_size=min(args.batch_size, len(train_data)),
        shuffle=True, num_workers=0,
    )
    validation_loader = DataLoader(
        validation_data, batch_size=min(args.batch_size, len(validation_data)),
        shuffle=False, num_workers=0,
    )

    model = CausalTransformerLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
        n_heads=args.heads,
        n_layers=args.layers,
        max_seq_len=context_length,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    criterion = nn.CrossEntropyLoss()
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)

    best_validation_loss = math.inf
    best_epoch = 0
    steps = 0
    stale_epochs = 0
    started = time.perf_counter()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Device=cpu parameters={parameter_count:,} vocab={tokenizer.vocab_size} "
        f"train_tokens={len(train_ids)} validation_tokens={len(validation_ids)} "
        f"context={context_length}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at step {steps + 1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * targets.numel()
            total_tokens += targets.numel()
            steps += 1
            if args.max_steps is not None and steps >= args.max_steps:
                break

        train_loss = total_loss / max(1, total_tokens)
        validation_loss = evaluate(model, validation_loader, criterion, device)
        if not math.isfinite(validation_loss):
            raise FloatingPointError(f"Non-finite validation loss at epoch {epoch}")
        scheduler.step(validation_loss)
        improved = validation_loss < best_validation_loss
        print(
            f"epoch={epoch}/{args.epochs} step={steps} train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} val_ppl={math.exp(min(validation_loss, 20)):.2f}"
            + (" best" if improved else "")
        )

        if improved:
            best_validation_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            save_safetensors(model.state_dict(), args.checkpoint)
            metadata = {
                "format": "safetensors",
                "model_type": "CausalTransformerLanguageModel",
                "vocab_size": tokenizer.vocab_size,
                "vocabulary": tokenizer._id_to_char,
                "unknown_token": tokenizer.unknown_token,
                "embedding_dim": args.embedding_dim,
                "heads": args.heads,
                "layers": args.layers,
                "dropout": args.dropout,
                "context_length": context_length,
                "best_validation_loss": best_validation_loss,
                "best_epoch": best_epoch,
                "training": {
                    "seed": args.seed,
                    "steps": steps,
                    "epochs_requested": args.epochs,
                    "corpus": str(args.corpus),
                    "train_tokens": len(train_ids),
                    "validation_tokens": len(validation_ids),
                    "optimizer": "AdamW",
                    "scheduler": "ReduceLROnPlateau",
                    "gradient_clip_norm": 1.0,
                },
            }
            args.metadata.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            stale_epochs += 1

        if args.max_steps is not None and steps >= args.max_steps:
            break
        if stale_epochs >= args.patience:
            print(f"Early stopping: validation loss did not improve for {args.patience} epochs.")
            break

    if best_epoch == 0:
        raise RuntimeError("Training completed without producing a checkpoint.")
    print(
        f"best_epoch={best_epoch} best_val_loss={best_validation_loss:.4f} "
        f"steps={steps} elapsed_seconds={time.perf_counter() - started:.1f}"
    )
    print(f"checkpoint={args.checkpoint}")
    print(f"metadata={args.metadata}")


if __name__ == "__main__":
    main()
