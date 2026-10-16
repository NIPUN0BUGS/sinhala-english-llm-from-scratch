"""Train a small Sinhala-English causal Transformer locally."""
from __future__ import annotations

import argparse
import random
import time
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


def read_ids(path: Path, tokenizer: CharacterTokenizer) -> list[int]:
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Corpus file is empty: {path}")
    return tokenizer.encode(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=DEFAULT_CORPUS_PATH,
        help="Legacy single-file mode; splits the file sequentially into train/validation.",
    )
    parser.add_argument("--train-corpus", type=Path, default=None)
    parser.add_argument("--validation-corpus", type=Path, default=None)
    parser.add_argument("--test-corpus", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Resume from a checkpoint created by this project (same tokenizer/model config required).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.context_length < 4:
        raise ValueError("epochs/batch-size must be positive and context-length must be at least 4")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive when provided")

    split_paths = (args.train_corpus, args.validation_corpus, args.test_corpus)
    if any(path is not None for path in split_paths) and not all(
        path is not None for path in split_paths
    ):
        raise ValueError("Provide all three: --train-corpus, --validation-corpus, --test-corpus")
    explicit_splits = all(path is not None for path in split_paths)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cpu")

    previous = None
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        try:
            previous = torch.load(args.resume, map_location=device, weights_only=True)
        except TypeError:
            previous = torch.load(args.resume, map_location=device)
        if "model_state_dict" not in previous or "model_config" not in previous:
            raise ValueError("Resume checkpoint is missing model_state_dict/model_config.")

    if explicit_splits:
        assert args.train_corpus and args.validation_corpus and args.test_corpus
        train_text = args.train_corpus.read_text(encoding="utf-8")
        if not train_text.strip():
            raise ValueError(f"Training corpus is empty: {args.train_corpus}")
        # Freeze the checkpoint vocabulary when resuming so new corpus characters map
        # to the unknown token instead of invalidating the model's embedding sizes.
        if previous is not None and previous.get("vocabulary"):
            tokenizer = CharacterTokenizer.from_vocabulary(
                previous["vocabulary"], previous.get("unknown_token", "�")
            )
        else:
            tokenizer = CharacterTokenizer().fit(train_text)
        train_ids = tokenizer.encode(train_text)
        val_ids = read_ids(args.validation_corpus, tokenizer)
        test_ids = read_ids(args.test_corpus, tokenizer)
        corpus_paths = {
            "train": str(args.train_corpus),
            "validation": str(args.validation_corpus),
            "test": str(args.test_corpus),
        }
    else:
        if not args.corpus.is_file():
            raise FileNotFoundError(f"Training corpus not found: {args.corpus}")
        text = args.corpus.read_text(encoding="utf-8")
        if previous is not None and previous.get("vocabulary"):
            tokenizer = CharacterTokenizer.from_vocabulary(
                previous["vocabulary"], previous.get("unknown_token", "�")
            )
        else:
            tokenizer = CharacterTokenizer().fit(text)
        token_ids = tokenizer.encode(text)
        if len(token_ids) < 40:
            raise ValueError("Corpus is too short for a train/validation split.")
        split_at = int(len(token_ids) * 0.9)
        train_ids, val_ids = token_ids[:split_at], token_ids[split_at:]
        test_ids = []
        corpus_paths = {"train": str(args.corpus), "validation": "sequential 10% split"}

    context_length = min(args.context_length, max(4, len(train_ids) // 10))
    if len(train_ids) <= context_length or len(val_ids) <= context_length:
        raise ValueError("Not enough tokens for the selected context length and data splits.")
    if test_ids and len(test_ids) <= context_length:
        raise ValueError("Test corpus must contain more tokens than the context length.")

    train_data = NextTokenDataset(train_ids, context_length=context_length)
    val_data = NextTokenDataset(val_ids, context_length=context_length)
    train_loader = DataLoader(
        train_data, batch_size=min(args.batch_size, len(train_data)), shuffle=True
    )
    val_loader = DataLoader(
        val_data, batch_size=min(args.batch_size, len(val_data)), shuffle=False
    )
    test_loader = None
    if test_ids:
        test_data = NextTokenDataset(test_ids, context_length=context_length)
        test_loader = DataLoader(
            test_data, batch_size=min(args.batch_size, len(test_data)), shuffle=False
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
    resumed_from = None
    best_val_loss = float("inf")
    best_epoch = 0
    if previous is not None:
        if previous["model_config"] != config:
            raise ValueError(
                "Checkpoint model configuration does not match this run. "
                "Use the same context length and model settings."
            )
        if previous.get("vocabulary") != tokenizer.vocabulary:
            raise ValueError(
                "Checkpoint vocabulary differs from the current corpus tokenizer. "
                "Resume with the original corpus/tokenizer, or start a fresh run."
            )
        model.load_state_dict(previous["model_state_dict"])
        if "optimizer_state_dict" in previous:
            optimizer.load_state_dict(previous["optimizer_state_dict"])
        best_val_loss = float(previous.get("best_validation_loss", float("inf")))
        best_epoch = int(previous.get("best_epoch", 0))
        resumed_from = str(args.resume)

    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(
        f"Train tokens: {len(train_ids):,}; validation tokens: {len(val_ids):,}; "
        f"test tokens: {len(test_ids):,}; vocabulary: {tokenizer.vocab_size}"
    )

    started_at = time.perf_counter()
    global_step = 0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)

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
        improved = val_loss < best_val_loss
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f}" + (" | new best" if improved else "")
        )
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "model_config": config,
                    "vocabulary": tokenizer.vocabulary,
                    "unknown_token": tokenizer.unknown_token,
                    "context_length": context_length,
                    "best_validation_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "training": {
                        "epochs_requested": args.epochs,
                        "resumed_from": resumed_from,
                        "optimizer_steps": global_step,
                        "seed": args.seed,
                        "corpus_paths": corpus_paths,
                        "train_tokens": len(train_ids),
                        "validation_tokens": len(val_ids),
                        "test_tokens": len(test_ids),
                    },
                },
                args.checkpoint,
            )
        if args.max_steps is not None and global_step >= args.max_steps:
            break

    print(
        f"Optimizer steps: {global_step}; best epoch: {best_epoch}; "
        f"best validation loss: {best_val_loss:.4f}; "
        f"elapsed seconds: {time.perf_counter() - started_at:.2f}"
    )
    if test_loader is not None:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"Held-out test loss (best checkpoint): {test_loss:.4f}")
    print(f"Saved best-validation checkpoint: {args.checkpoint}")
    print("Note: a small character-level model and tiny corpus are not enough for general-purpose chat.")


if __name__ == "__main__":
    main()
