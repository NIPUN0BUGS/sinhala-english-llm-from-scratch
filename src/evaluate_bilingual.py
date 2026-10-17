"""Evaluate a checkpoint on a small fixed Sinhala-English translation/QA smoke benchmark.

Scores are lexical similarity indicators, not proof of factual correctness or semantic quality.
"""
from __future__ import annotations

import argparse
import difflib
import json
import time
from pathlib import Path

import torch

from src.model import TinyLanguageModel
from src.tokenizer import CharacterTokenizer, SentencePieceBPETokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "hourly-bpe-model.pt"
DEFAULT_BENCHMARK = ROOT / "data" / "evaluation" / "bilingual_benchmark.jsonl"
DEFAULT_OUTPUT = ROOT / "evaluation-results" / "latest.json"


def load_checkpoint(path: Path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("tokenizer_type", "character") == "bpe":
        tokenizer = SentencePieceBPETokenizer(checkpoint["tokenizer_model"])
    else:
        tokenizer = CharacterTokenizer.from_vocabulary(
            checkpoint["vocabulary"],
            unknown_token=checkpoint.get("unknown_token", "�"),
        )
    model = TinyLanguageModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer, checkpoint


def generate_greedy(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = [0]
    generated = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            context = generated[:, -model.max_seq_len:]
            next_id = int(torch.argmax(model(context)[:, -1, :], dim=-1).item())
            generated = torch.cat(
                [generated, torch.tensor([[next_id]], dtype=torch.long)], dim=1
            )
    decoded = tokenizer.decode(generated[0].tolist())
    decoded_prompt = tokenizer.decode(prompt_ids)
    return decoded[len(decoded_prompt):].strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not args.benchmark.is_file():
        raise FileNotFoundError(f"Benchmark not found: {args.benchmark}")
    if not 1 <= args.max_new_tokens <= 200:
        raise ValueError("--max-new-tokens must be between 1 and 200")

    cases = [
        json.loads(line)
        for line in args.benchmark.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError("Benchmark has no cases.")
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint)
    results = []
    started = time.perf_counter()
    for case in cases:
        prediction = generate_greedy(
            model, tokenizer, case["prompt"], args.max_new_tokens
        )
        reference = case["reference"].strip()
        predicted_normalized = " ".join(prediction.casefold().split())
        reference_normalized = " ".join(reference.casefold().split())
        similarity = difflib.SequenceMatcher(
            a=reference_normalized, b=predicted_normalized
        ).ratio()
        results.append({
            "id": case["id"],
            "task": case["task"],
            "prompt": case["prompt"],
            "reference": reference,
            "prediction": prediction,
            "exact_match": predicted_normalized == reference_normalized,
            "character_similarity": round(similarity, 4),
        })

    grouped = {}
    for task in sorted({case["task"] for case in results}):
        subset = [item for item in results if item["task"] == task]
        grouped[task] = {
            "cases": len(subset),
            "exact_match_rate": round(
                sum(item["exact_match"] for item in subset) / len(subset), 4
            ),
            "mean_character_similarity": round(
                sum(item["character_similarity"] for item in subset) / len(subset), 4
            ),
        }
    report = {
        "checkpoint": str(args.checkpoint),
        "tokenizer_type": checkpoint.get("tokenizer_type", "character"),
        "best_validation_loss": checkpoint.get("best_validation_loss"),
        "best_epoch": checkpoint.get("best_epoch"),
        "benchmark": str(args.benchmark),
        "case_count": len(results),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "overall": {
            "exact_match_rate": round(
                sum(item["exact_match"] for item in results) / len(results), 4
            ),
            "mean_character_similarity": round(
                sum(item["character_similarity"] for item in results) / len(results), 4
            ),
        },
        "by_task": grouped,
        "metric_note": (
            "Character similarity is lexical overlap only; it does not assess factual "
            "correctness, meaning, or translation adequacy. The benchmark is a small smoke test."
        ),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in (
        "tokenizer_type", "case_count", "elapsed_seconds", "overall", "by_task"
    )}, ensure_ascii=False, indent=2))
    print(f"Detailed evaluation report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
