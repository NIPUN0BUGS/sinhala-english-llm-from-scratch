"""Simple Unicode character tokenizer for Sinhala-English text."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CharacterTokenizer:
    """Map each Unicode character to an integer ID, preserving round trips."""

    unknown_token: str = "�"

    def __post_init__(self) -> None:
        self._char_to_id: dict[str, int] = {}
        self._id_to_char: list[str] = []

    def fit(self, text: str) -> "CharacterTokenizer":
        if not text:
            raise ValueError("Cannot fit tokenizer on empty text.")
        chars = sorted(set(text) | {self.unknown_token})
        self._id_to_char = chars
        self._char_to_id = {char: idx for idx, char in enumerate(chars)}
        return self

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_char)

    def encode(self, text: str) -> list[int]:
        if not self._char_to_id:
            raise RuntimeError("Call fit(text) before encode(text).")
        unknown_id = self._char_to_id[self.unknown_token]
        return [self._char_to_id.get(char, unknown_id) for char in text]

    def decode(self, token_ids: list[int]) -> str:
        if not self._id_to_char:
            raise RuntimeError("Call fit(text) before decode(token_ids).")
        chars: list[str] = []
        for token_id in token_ids:
            if token_id < 0 or token_id >= len(self._id_to_char):
                raise ValueError(f"Token ID out of range: {token_id}")
            chars.append(self._id_to_char[token_id])
        return "".join(chars)
