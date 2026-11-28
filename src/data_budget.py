"""Measure progress toward the first one-million-token training corpus."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def measure(manifest_path: Path, target: int, min_per_language: int) -> dict:
    counts = {"si": 0, "en": 0}
    records = 0
    if manifest_path.is_file():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            language = item.get("language")
            text = item.get("text", "")
            if language not in counts or not isinstance(text, str):
                raise ValueError("Manifest contains an invalid language or text record.")
            counts[language] += len(text)
            records += 1
    total = sum(counts.values())
    return {
        "tokenizer": "Unicode character-level (one Unicode code point per token)",
        "target_tokens": target,
        "total_tokens": total,
        "progress": min(1.0, total / target) if target else 1.0,
        "language_tokens": counts,
        "minimum_tokens_per_language": min_per_language,
        "records": records,
        "ready": total >= target and min(counts.values()) >= min_per_language,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/raw/self_learning_manifest.jsonl")
    parser.add_argument("--target", type=int, default=1_000_000)
    parser.add_argument("--min-per-language", type=int, default=100_000)
    parser.add_argument("--report", type=Path, default=ROOT / "checkpoints/data_budget.json")
    args = parser.parse_args()
    if args.target < 1 or args.min_per_language < 0:
        raise ValueError("Token target must be positive and per-language minimum non-negative.")
    result = measure(args.manifest, args.target, args.min_per_language)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Training tokens: {result['total_tokens']:,} / {args.target:,} ({result['progress']:.1%})")
    print(f"Sinhala: {result['language_tokens']['si']:,}; English: {result['language_tokens']['en']:,}")
    print(f"Source records: {result['records']:,}; ready for training: {result['ready']}")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as stream:
            stream.write(f"ready={'true' if result['ready'] else 'false'}\n")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
