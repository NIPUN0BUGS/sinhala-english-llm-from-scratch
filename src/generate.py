"""Generate text from a locally saved Safetensors causal Transformer checkpoint."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from src.model import CausalTransformerLanguageModel
from src.safe_checkpoint import load_safetensors
from src.tokenizer import CharacterTokenizer


def load_model(checkpoint: Path, metadata_path: Path):
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "safetensors":
        raise ValueError("Unsupported checkpoint format; expected safetensors.")
    if metadata.get("model_type") != "CausalTransformerLanguageModel":
        raise ValueError("Checkpoint is not a causal Transformer model.")
    tokenizer = CharacterTokenizer(unknown_token=metadata["unknown_token"])
    tokenizer._id_to_char = list(metadata["vocabulary"])
    tokenizer._char_to_id = {char: index for index, char in enumerate(tokenizer._id_to_char)}
    model = CausalTransformerLanguageModel(
        vocab_size=metadata["vocab_size"],
        embedding_dim=metadata["embedding_dim"],
        n_heads=metadata["heads"],
        n_layers=metadata["layers"],
        max_seq_len=metadata["context_length"],
        dropout=metadata["dropout"],
    )
    model.load_state_dict(load_safetensors(checkpoint))
    model.eval()
    return model, tokenizer, metadata


def generate(
    model: CausalTransformerLanguageModel,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 20,
) -> str:
    if max_new_tokens < 0 or temperature <= 0 or top_k < 0:
        raise ValueError("max_new_tokens/top_k must be non-negative and temperature positive")
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [0]
    tokens = torch.tensor([ids], dtype=torch.long)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            context = tokens[:, -model.max_seq_len:]
            logits = model(context)[:, -1, :] / temperature
            if top_k:
                k = min(top_k, logits.size(-1))
                threshold = torch.topk(logits, k).values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
    return tokenizer.decode(tokens[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metadata_path = args.metadata or args.checkpoint.with_name(
        args.checkpoint.name.replace(".safetensors", ".metadata.json")
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, tokenizer, _ = load_model(args.checkpoint, metadata_path)
    print(generate(
        model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_k
    ))


if __name__ == "__main__":
    main()
