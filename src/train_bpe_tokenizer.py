"""Train a SentencePiece BPE tokenizer from scratch on the cleaned training split.

Install optional dependency with: python -m pip install sentencepiece
This creates tokenizer artifacts only; it does not load or use a pretrained model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/train.txt")
    parser.add_argument("--model-prefix", type=Path, default=ROOT / "artifacts/sinhala_english_bpe")
    parser.add_argument("--vocab-size", type=int, default=8000)
    args = parser.parse_args()
    if args.vocab_size < 256:
        raise ValueError("--vocab-size should be at least 256")
    if not args.input.is_file():
        raise FileNotFoundError(f"Training text not found: {args.input}")
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise SystemExit("Install the optional dependency first: python -m pip install sentencepiece") from exc
    text = args.input.read_text(encoding="utf-8")
    if len(text.strip()) < 100:
        raise ValueError("Training text is too small for a useful tokenizer experiment.")
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
        user_defined_symbols=["<|sinhala|>", "<|english|>", "<|code|>", "<|user|>", "<|assistant|>"],
    )
    processor = spm.SentencePieceProcessor(model_file=str(args.model_prefix) + ".model")
    print(f"Tokenizer model: {args.model_prefix}.model")
    print(f"Vocabulary size: {processor.get_piece_size()}")
    print(f"Example tokenization: {processor.encode('සිංහල Python code', out_type=str)}")
    print("Note: the existing TinyLanguageModel currently uses character IDs; this BPE tokenizer is a separate artifact until model/training code is migrated.")


if __name__ == "__main__":
    main()
