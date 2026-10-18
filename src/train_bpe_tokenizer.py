"""Train a SentencePiece BPE tokenizer from scratch on the cleaned training split.

Install optional dependency with: python -m pip install sentencepiece
This creates tokenizer artifacts only; it does not load or use a pretrained model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPECIAL_TOKENS = [
    "<|sinhala|>",
    "<|english|>",
    "<|code|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/train.txt")
    parser.add_argument("--model-prefix", type=Path, default=ROOT / "artifacts/sinhala_english_bpe")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument(
        "--user-symbol",
        action="append",
        default=None,
        help="Add a special token; repeat this option to add multiple tokens.",
    )
    args = parser.parse_args()
    if args.vocab_size < 256:
        raise ValueError("--vocab-size should be at least 256")
    if not args.input.is_file():
        raise FileNotFoundError(f"Training text not found: {args.input}")
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise SystemExit(
            "Install the optional dependency first: python -m pip install sentencepiece"
        ) from exc

    text = args.input.read_text(encoding="utf-8")
    if len(text.strip()) < 100:
        raise ValueError("Training text is too small for a useful tokenizer experiment.")
    special_tokens = list(dict.fromkeys(DEFAULT_SPECIAL_TOKENS + (args.user_symbol or [])))
    if any(not token.strip() for token in special_tokens):
        raise ValueError("Special tokens cannot be empty.")
    args.model_prefix.parent.mkdir(parents=True, exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=str(args.input),
        model_prefix=str(args.model_prefix),
        model_type="bpe",
        vocab_size=args.vocab_size,
        character_coverage=1.0,
        hard_vocab_limit=False,
        shuffle_input_sentence=False,
        input_sentence_size=0,
        num_threads=4,
        user_defined_symbols=special_tokens,
    )
    model_path = Path(str(args.model_prefix) + ".model")
    processor = spm.SentencePieceProcessor(model_file=str(model_path))
    print(f"Tokenizer model: {model_path}")
    print(f"Requested vocabulary size: {args.vocab_size}")
    print(f"Actual vocabulary size: {processor.get_piece_size()}")
    print(f"Special tokens: {', '.join(special_tokens)}")
    print(f"Example tokenization: {processor.encode('සිංහල Python code', out_type=str)}")
    print("BPE tokenizer is ready for training with: python -m src.train --tokenizer bpe")


if __name__ == "__main__":
    main()
