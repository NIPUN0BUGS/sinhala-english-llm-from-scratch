import unittest
from unittest.mock import patch
import json
from src.tokenizer import CharacterTokenizer
from src.collect_wikipedia import LICENSE, LICENSE_URL, fetch_random


class SelfLearningSupportTests(unittest.TestCase):
    def test_saved_vocabulary_handles_unseen_characters(self):
        original = CharacterTokenizer().fit("Hello ආයුබෝවන්")
        restored = CharacterTokenizer(unknown_token=original.unknown_token)
        restored._id_to_char = list(original._id_to_char)
        restored._char_to_id = {char: i for i, char in enumerate(restored._id_to_char)}
        encoded = restored.encode("Hello 🧪")
        self.assertEqual(encoded[-1], restored._char_to_id[restored.unknown_token])

    def test_vocabulary_extension_preserves_existing_ids(self):
        tokenizer = CharacterTokenizer().fit("Sinhala")
        old_ids = dict(tokenizer._char_to_id)
        tokenizer.extend("English සිංහල")
        for char, token_id in old_ids.items():
            self.assertEqual(tokenizer._char_to_id[char], token_id)
        self.assertIn("E", tokenizer._char_to_id)
        self.assertIn("ි", tokenizer._char_to_id)

    def test_wikipedia_attribution_license_is_explicit(self):
        self.assertEqual(LICENSE, "CC BY-SA 4.0")
        self.assertTrue(LICENSE_URL.startswith("https://creativecommons.org/"))

    @patch("src.collect_wikipedia.urllib.request.urlopen")
    def test_fetch_random_returns_english_page(self, mocked_open):
        payload = {"query": {"pages": [{
            "title": "Language model",
            "fullurl": "https://en.wikipedia.org/wiki/Language_model",
            "extract": "A language model is a probability distribution over sequences of words. " * 5,
        }]}}
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return json.dumps(payload).encode("utf-8")
        mocked_open.return_value = Response()
        record = fetch_random("en")
        self.assertEqual(record["language"], "en")
        self.assertTrue(record["url"].startswith("https://en.wikipedia.org/"))
        self.assertEqual(record["license"], "CC BY-SA 4.0")

    @patch("src.collect_wikipedia.urllib.request.urlopen")
    def test_fetch_random_rejects_wrong_language_host(self, mocked_open):
        payload = {"query": {"pages": [{
            "title": "Wrong host", "fullurl": "https://si.wikipedia.org/wiki/Wrong_host",
            "extract": "A sufficiently long extract. " * 20,
        }]}}
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def read(self): return json.dumps(payload).encode("utf-8")
        mocked_open.return_value = Response()
        self.assertIsNone(fetch_random("en"))


if __name__ == "__main__":
    unittest.main()
