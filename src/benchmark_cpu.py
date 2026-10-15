"""Benchmark a fresh randomly initialized tiny model on CPU; saves timing JSON only."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from src.model import TinyLanguageModel
from src.tokenizer import CharacterTokenizer

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/processed/train.txt")
    parser.add_argument("--tokens", type=int, default=100_000, help="Character tokens to process (current tokenizer is character-level).")
    parser.add_argument("--steps", type=int, default=100, help="Maximum optimizer steps; each step processes batch_size * context_length target tokens.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark-results/cpu-benchmark.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tokens < 1 or args.steps < 1 or args.batch_size < 1 or args.context_length < 4:
        raise ValueError("tokens/steps/batch-size must be positive; context-length must be >= 4")
    if not args.corpus.is_file():
        raise FileNotFoundError(f"Corpus not found: {args.corpus}. Run src.prepare_corpus first.")
    text = args.corpus.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Corpus is empty.")
    tokenizer = CharacterTokenizer().fit(text)
    ids = tokenizer.encode(text)
    if len(ids) < args.context_length + 1:
        raise ValueError("Corpus is shorter than context length.")
    ids = ids[:min(args.tokens, len(ids))]
    if len(ids) < args.context_length + 1:
        raise ValueError("Not enough characters for requested benchmark.")

    torch.manual_seed(42)
    torch.set_num_threads(4)
    model = TinyLanguageModel(
        vocab_size=tokenizer.vocab_size, embedding_dim=64, num_heads=4,
        num_layers=2, max_seq_len=args.context_length, dropout=0.1
    ).cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()
    usable = len(ids) - args.context_length - 1
    steps = min(args.steps, max(1, math.ceil(usable / (args.batch_size * args.context_length))))
    generator = torch.Generator().manual_seed(42)
    started = time.perf_counter()
    processed = 0
    for step in range(steps):
        starts = torch.randint(0, usable, (args.batch_size,), generator=generator)
        x = torch.stack([torch.tensor(ids[int(s):int(s)+args.context_length]) for s in starts])
        y = torch.stack([torch.tensor(ids[int(s)+1:int(s)+args.context_length+1]) for s in starts])
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits.reshape(-1, model.vocab_size), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        processed += y.numel()
    elapsed = time.perf_counter() - started
    tokens_per_second = processed / max(elapsed, 1e-9)
    result = {
        "device": "cpu",
        "model": "fresh random initialization; no checkpoint loaded",
        "tokenizer": "Unicode character tokenizer (not BPE)",
        "parameters": sum(p.numel() for p in model.parameters()),
        "corpus_characters_available": len(text),
        "requested_character_tokens": args.tokens,
        "training_target_tokens_processed": processed,
        "optimizer_steps": steps,
        "elapsed_seconds": round(elapsed, 3),
        "tokens_per_second": round(tokens_per_second, 2),
        "estimated_hours_for_5m_tokens_at_measured_speed": round(5_000_000 / max(tokens_per_second, 1e-9) / 3600, 2),
        "estimated_hours_for_10m_tokens_at_measured_speed": round(10_000_000 / max(tokens_per_second, 1e-9) / 3600, 2),
        "note": "Estimate is a rough throughput extrapolation, not a prediction of convergence time. Character tokens differ from BPE tokens."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
