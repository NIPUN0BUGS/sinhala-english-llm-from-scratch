import unittest

from src.prepare_corpus import clean_records, split_records


class PrepareCorpusTests(unittest.TestCase):
    def test_normalizes_whitespace_and_removes_duplicates(self) -> None:
        records, stats = clean_records("  Hello   world  \n\nHello world\nආයුබෝවන්\n")
        self.assertEqual(records, ["Hello world", "ආයුබෝවන්"])
        self.assertEqual(stats["blank"], 1)
        self.assertEqual(stats["duplicates"], 1)

    def test_pairs_adjacent_sinhala_and_english_lines(self) -> None:
        records, stats = clean_records(
            "Sinhala: ආයුබෝවන්\nEnglish: Hello\n\nSinhala: ස්තුතියි\nEnglish: Thank you\n"
        )
        self.assertEqual(len(records), 2)
        self.assertIn("Sinhala: ආයුබෝවන්\tEnglish: Hello", records[0])
        self.assertNotIn("\\t", records[0])
        self.assertEqual(stats["paired_records"], 2)

    def test_split_is_deterministic_and_non_overlapping(self) -> None:
        records = [f"record {index}" for index in range(100)]
        first = split_records(records, seed=7)
        second = split_records(records, seed=7)
        self.assertEqual(first, second)
        train, validation, test = map(set, first)
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(len(train | validation | test), len(records))

    def test_rejects_too_few_records(self) -> None:
        with self.assertRaises(ValueError):
            split_records(["one", "two"])


if __name__ == "__main__":
    unittest.main()
