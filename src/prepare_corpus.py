"""Prepare clean UTF-8 records and keep adjacent Sinhala/English pairs together."""
from __future__ import annotations

import argparse
import random
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LANGUAGE_PREFIX = re.compile(r"^(Sinhala|English):\s*", re.IGNORECASE)


def clean_records(text: str) -> tuple[list[str], dict[str, int]]:
    """Normalize text, pair adjacent Sinhala/English lines, and remove duplicates."""
    lines = text.splitlines()
    normalized: list[str] = []
    stats = {"blank": 0, "duplicates": 0, "replacement_chars": text.count("\ufffd"), "paired_records": 0}
    for raw in lines:
        line = unicodedata.normalize("NFC", " ".join(raw.split()))
        if not line:
            stats["blank"] += 1
            continue
        normalized.append(line)

    records: list[str] = []
    i = 0
    while i < len(normalized):
        current = normalized[i]
        match = LANGUAGE_PREFIX.match(current)
        if match and match.group(1).lower() == "sinhala" and i + 1 < len(normalized):
            next_match = LANGUAGE_PREFIX.match(normalized[i + 1])
            if next_match and next_match.group(1).lower() == "english":
                si = LANGUAGE_PREFIX.sub("", current, count=1).strip()
                en = LANGUAGE_PREFIX.sub("", normalized[i + 1], count=1).strip()
                current = f"Sinhala: {si}\tEnglish: {en}"
                stats["paired_records"] += 1
                i += 1
        records.append(current)
        i += 1

    unique: list[str] = []
    seen: set[str] = set()
    for record in records:
        if record in seen:
            stats["duplicates"] += 1
            continue
        seen.add(record)
        unique.append(record)
    return unique, stats


def split_records(records: list[str], validation_ratio: float = 0.05,
                  test_ratio: float = 0.05, seed: int = 42) -> tuple[list[str], list[str], list[str]]:
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
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/train_from_dev.txt")
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
    train, validation, test = split_records(records, args.validation_ratio, args.test_ratio, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in (("train.txt", train), ("validation.txt", validation), ("test.txt", test)):
        (args.output_dir / name).write_text("\n".join(subset) + "\n", encoding="utf-8")
    print(f"Input lines: {len(text.splitlines())}")
    print(f"Unique cleaned records: {len(records)}")
    print(f"Paired Sinhala-English records: {stats['paired_records']}")
    print(f"Removed blank lines: {stats['blank']}; duplicates: {stats['duplicates']}")
    print(f"Unicode replacement characters found: {stats['replacement_chars']}")
    print(f"Train: {len(train)}; validation: {len(validation)}; test: {len(test)}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
