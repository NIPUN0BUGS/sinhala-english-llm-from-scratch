import unittest
from src.tokenizer import CharacterTokenizer
from src.collect_wikipedia import LICENSE, LICENSE_URL

class SelfLearningSupportTests(unittest.TestCase):
    def test_saved_vocabulary_handles_unseen_characters(self):
        original = CharacterTokenizer().fit("Hello ආයුබෝවන්")
        restored = CharacterTokenizer(unknown_token=original.unknown_token)
        restored._id_to_char = list(original._id_to_char)
        restored._char_to_id = {char: i for i, char in enumerate(restored._id_to_char)}
        encoded = restored.encode("Hello 🧪")
        self.assertEqual(encoded[-1], restored._char_to_id[restored.unknown_token])

    def test_wikipedia_attribution_license_is_explicit(self):
        self.assertEqual(LICENSE, "CC BY-SA 4.0")
        self.assertTrue(LICENSE_URL.startswith("https://creativecommons.org/"))

if __name__ == "__main__":
    unittest.main()
