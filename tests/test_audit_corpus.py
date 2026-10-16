import unittest

from src.audit_corpus import audit_text


class AuditCorpusTests(unittest.TestCase):
    def test_reports_duplicates_blanks_and_language_content(self):
        report = audit_text("Hello සිංහල\n\nHello සිංහල\n")
        self.assertEqual(report["lines_nonempty"], 2)
        self.assertEqual(report["blank_lines"], 1)
        self.assertEqual(report["exact_duplicate_lines"], 1)
        self.assertGreater(report["sinhala_characters"], 0)
        self.assertGreater(report["latin_letters"], 0)

    def test_reports_bad_encoding_markers(self):
        report = audit_text("ï¿½ text �")
        self.assertGreaterEqual(report["suspicious_mojibake_markers"], 1)
        self.assertEqual(report["replacement_characters"], 1)


if __name__ == "__main__":
    unittest.main()
