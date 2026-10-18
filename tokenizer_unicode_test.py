from pathlib import Path
from src.tokenizer import CharacterTokenizer

sample = "\u0dc3\u0dd2\u0d82\u0dc4\u0dbd: \u0db8\u0d9c\u0dda \u0db1\u0db8"
text = Path("data/raw/train_from_dev.txt").read_text(encoding="utf-8")

tokenizer = CharacterTokenizer().fit(text)
decoded = tokenizer.decode(tokenizer.encode(sample))
vocabulary = tokenizer.vocabulary

sinhala_chars = [ch for ch in sample if "\u0D80" <= ch <= "\u0DFF"]

print("Original:", repr(sample))
print("Decoded:", repr(decoded))
print("Round-trip OK:", sample == decoded)
print("Sinhala characters in vocabulary:",
      all(ch in vocabulary for ch in sinhala_chars))
print("Vocabulary size:", tokenizer.vocab_size)
print("Unknown token:", repr(tokenizer.unknown_token))
