import unittest

from src.collect_open_datasets import (
    LICENSES,
    SOURCES,
    language_ratio,
    normalize_text,
    text_hash,
)


class OpenDatasetCollectorTests(unittest.TestCase):
    def test_sources_are_allowlisted_with_explicit_licenses(self):
        self.assertEqual(
            {(item["dataset"], item["split"], item["language"], item["license"]) for item in SOURCES},
            {
                ("Minuri/sinhala-corpus-culturax", "train", "si", "CC BY-SA 4.0"),
                ("keshan/lk_hansard_trilingual", "sinhala", "si", "CC BY 4.0"),
                ("keshan/lk_hansard_trilingual", "english", "en", "CC BY 4.0"),
            },
        )
        self.assertEqual(set(LICENSES), {"CC BY-SA 4.0", "CC BY 4.0"})

    def test_sinhala_script_filter(self):
        text = "සිංහල භාෂාව සහ සාහිත්‍යය " * 10
        self.assertGreater(language_ratio(text, "si"), 0.8)
        self.assertLess(language_ratio("This is an English paragraph. " * 10, "si"), 0.1)

    def test_english_script_filter(self):
        text = "The Parliament of Sri Lanka publishes official reports. " * 10
        self.assertGreater(language_ratio(text, "en"), 0.8)
        self.assertLess(language_ratio("සිංහල භාෂාව සහ සාහිත්‍යය " * 10, "en"), 0.1)

    def test_normalization_and_hash_are_stable(self):
        self.assertEqual(normalize_text("  hello\n\tworld  "), "hello world")
        self.assertEqual(text_hash("Hello   WORLD"), text_hash("hello world"))


if __name__ == "__main__":
    unittest.main()
