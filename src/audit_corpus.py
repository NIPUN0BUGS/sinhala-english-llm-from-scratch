"""Audit UTF-8 Sinhala-English corpus quality and inspect training checkpoints.

Examples (PowerShell, from repository root):
  python -m src.audit_corpus --input data/raw/train_from_dev.txt
  python -m src.audit_corpus --input data/processed/train.txt --validation data/processed/validation.txt --test data/processed/test.txt
  python -m src.audit_corpus --input data/raw/train_from_dev.txt --checkpoint checkpoints/steps2000.pt
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SINHALA_RANGES = ((0x0D80, 0x0DFF),)
LATIN_RANGES = ((0x0041, 0x005A), (0x0061, 0x007A))


def in_ranges(char: str, ranges: tuple[tuple[int, int], ...]) -> bool:
    code = ord(char)
    return any(start <= code <= end for start, end in ranges)


def audit_text(text: str) -> dict:
    lines = text.splitlines()
    nonempty = [line.strip() for line in lines if line.strip()]
    normalized = [unicodedata.normalize("NFC", line) for line in nonempty]
    duplicates = len(normalized) - len(set(normalized))
    chars = [char for char in text if not char.isspace()]
    sinhala_chars = sum(in_ranges(char, SINHALA_RANGES) for char in chars)
    latin_chars = sum(any(in_ranges(char, ((a, b),)) for a, b in LATIN_RANGES) for char in chars)
    replacement_count = text.count("\ufffd")
    suspicious_mojibake = sum(text.count(marker) for marker in ("ï¿½", "Ã", "â€™", "â€œ", "â€", "�"))
    line_lengths = [len(line) for line in nonempty]
    return {
        "lines_total": len(lines),
        "lines_nonempty": len(nonempty),
        "blank_lines": len(lines) - len(nonempty),
        "exact_duplicate_lines": duplicates,
        "unique_nonempty_lines": len(set(normalized)),
        "replacement_characters": replacement_count,
        "suspicious_mojibake_markers": suspicious_mojibake,
        "characters_non_whitespace": len(chars),
        "sinhala_characters": sinhala_chars,
        "latin_letters": latin_chars,
        "sinhala_share_of_nonspace_chars": round(sinhala_chars / len(chars), 4) if chars else 0.0,
        "latin_share_of_nonspace_chars": round(latin_chars / len(chars), 4) if chars else 0.0,
        "lines_with_sinhala": sum(any(in_ranges(c, SINHALA_RANGES) for c in line) for line in nonempty),
        "lines_with_latin": sum(any(in_ranges(c, LATIN_RANGES) for c in line) for line in nonempty),
        "line_length_min": min(line_lengths, default=0),
        "line_length_median": statistics.median(line_lengths) if line_lengths else 0,
        "line_length_p95": sorted(line_lengths)[min(len(line_lengths)-1, math.ceil(0.95*len(line_lengths))-1)] if line_lengths else 0,
        "line_length_max": max(line_lengths, default=0),
        "unicode_not_nfc_lines": sum(line != unicodedata.normalize("NFC", line) for line in nonempty),
    }


def read_utf8(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8")


def checkpoint_summary(path: Path) -> dict:
    import torch

    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    summary = {
        "path": str(path),
        "model_config": data.get("model_config"),
        "vocabulary_size": len(data.get("vocabulary", [])),
        "best_validation_loss_saved": data.get("best_validation_loss"),
        "best_epoch": data.get("best_epoch"),
        "training_metadata": data.get("training", {}),
        "has_optimizer_state": "optimizer_state_dict" in data,
        "has_scheduler_state": "scheduler_state_dict" in data,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Corpus file to audit")
    parser.add_argument("--validation", type=Path, help="Optional validation split")
    parser.add_argument("--test", type=Path, help="Optional held-out test split")
    parser.add_argument("--checkpoint", type=Path, help="Optional checkpoint to inspect")
    args = parser.parse_args()

    report = {"corpora": {"input": {"path": str(args.input), **audit_text(read_utf8(args.input))}}}
    if args.validation:
        report["corpora"]["validation"] = {
            "path": str(args.validation), **audit_text(read_utf8(args.validation))
        }
    if args.test:
        report["corpora"]["test"] = {
            "path": str(args.test), **audit_text(read_utf8(args.test))
        }
    if args.validation and args.test:
        input_lines = {unicodedata.normalize("NFC", x.strip()) for x in read_utf8(args.input).splitlines() if x.strip()}
        validation_lines = {unicodedata.normalize("NFC", x.strip()) for x in read_utf8(args.validation).splitlines() if x.strip()}
        test_lines = {unicodedata.normalize("NFC", x.strip()) for x in read_utf8(args.test).splitlines() if x.strip()}
        report["split_overlap"] = {
            "input_validation_exact_line_overlap": len(input_lines & validation_lines),
            "input_test_exact_line_overlap": len(input_lines & test_lines),
            "validation_test_exact_line_overlap": len(validation_lines & test_lines),
        }
    if args.checkpoint:
        report["checkpoint"] = checkpoint_summary(args.checkpoint)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
