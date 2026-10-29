"""Generate sample text from a saved local Transformer checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.model import TinyLanguageModel
from src.tokenizer import CharacterTokenizer, SentencePieceBPETokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = ROOT / "checkpoints" / "tiny_transformer.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--prompt", type=str, default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be zero or greater")
    if args.temperature <= 0:
        raise ValueError("--temperature must be greater than zero")
    if args.top_k < 0:
        raise ValueError("--top-k cannot be negative")
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Run `python -m src.train` first."
        )

    torch.manual_seed(args.seed)
    try:
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=True
        )
    except TypeError:  # Compatibility with older PyTorch releases.
        checkpoint = torch.load(args.checkpoint, map_location="cpu")

    tokenizer_type = checkpoint.get("tokenizer_type", "character")
    if tokenizer_type == "bpe":
        tokenizer = SentencePieceBPETokenizer(checkpoint["tokenizer_model"])
    else:
        tokenizer = CharacterTokenizer.from_vocabulary(
            checkpoint["vocabulary"],
            unknown_token=checkpoint.get("unknown_token", "�"),
        )
    model = TinyLanguageModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        prompt_ids = tokenizer.encode(tokenizer.unknown_token)
    generated = torch.tensor([prompt_ids], dtype=torch.long)

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            context = generated[:, -model.max_seq_len :]
            logits = model(context)[:, -1, :] / args.temperature
            if args.top_k:
                k = min(args.top_k, logits.shape[-1])
                threshold = torch.topk(logits, k, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            probabilities = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1)
            generated = torch.cat((generated, next_id), dim=1)

    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
