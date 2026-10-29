import unittest

from src.tokenizer import CharacterTokenizer


class CharacterTokenizerTests(unittest.TestCase):
    def test_round_trip_with_sinhala_and_english(self) -> None:
        sample = "Hello ආයුබෝවන්!"
        tokenizer = CharacterTokenizer().fit(sample)
        self.assertEqual(tokenizer.decode(tokenizer.encode(sample)), sample)

    def test_unknown_character_uses_unknown_token(self) -> None:
        tokenizer = CharacterTokenizer().fit("abc")
        self.assertEqual(tokenizer.decode(tokenizer.encode("z")), tokenizer.unknown_token)

    def test_encode_before_fit_fails(self) -> None:
        with self.assertRaises(RuntimeError):
            CharacterTokenizer().encode("hi")


if __name__ == "__main__":
    unittest.main()
