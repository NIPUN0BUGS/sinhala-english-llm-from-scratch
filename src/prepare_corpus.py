"""Prepare a clean, reproducible corpus split from UTF-8 text records.

Input format: one independent training record per non-empty line. For bilingual
pairs, keep both languages on the same line (separated by a tab or space).
The script does not read parquet files or download data automatically.
"""
from __future__ import annotations

import argparse
import random
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def clean_records(text: str) -> tuple[list[str], dict[str, int]]:
    """Normalize Unicode/whitespace, remove blank and duplicate records."""
    records: list[str] = []
    seen: set[str] = set()
    stats = {"blank": 0, "duplicates": 0, "replacement_chars": text.count("\ufffd")}
    for raw in text.splitlines():
        record = unicodedata.normalize("NFC", " ".join(raw.split()))
        if not record:
            stats["blank"] += 1
            continue
        if record in seen:
            stats["duplicates"] += 1
            continue
        seen.add(record)
        records.append(record)
    return records, stats


def split_records(
    records: list[str],
    validation_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Create deterministic, non-overlapping train/validation/test splits."""
    if not 0 <= validation_ratio < 1 or not 0 <= test_ratio < 1:
        raise ValueError("Split ratios must be in [0, 1).")
    if validation_ratio + test_ratio >= 1:
        raise ValueError("Validation and test ratios must sum to less than 1.")
    if len(records) < 3:
        raise ValueError("Need at least 3 unique records to create three splits.")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * test_ratio)) if test_ratio else 0
    val_count = max(1, round(len(shuffled) * validation_ratio)) if validation_ratio else 0
    if test_count + val_count >= len(shuffled):
        raise ValueError("Not enough records for the requested split ratios.")
    test = shuffled[:test_count]
    validation = shuffled[test_count:test_count + val_count]
    train = shuffled[test_count + val_count:]
    return train, validation, test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/train.txt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input text file not found: {args.input}")
    text = args.input.read_text(encoding="utf-8")
    records, stats = clean_records(text)
    train, validation, test = split_records(
        records, args.validation_ratio, args.test_ratio, args.seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train.txt", train), ("validation.txt", validation), ("test.txt", test)):
        (args.output_dir / name).write_text("\n".join(subset) + "\n", encoding="utf-8")
    print(f"Input records: {len(text.splitlines())}")
    print(f"Unique cleaned records: {len(records)}")
    print(f"Removed blank lines: {stats['blank']}; duplicates: {stats['duplicates']}")
    print(f"Unicode replacement characters found: {stats['replacement_chars']}")
    print(f"Train: {len(train)}; validation: {len(validation)}; test: {len(test)}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
