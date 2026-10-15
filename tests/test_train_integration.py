import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


class TrainIntegrationTests(unittest.TestCase):
    def test_explicit_splits_save_best_checkpoint_and_evaluate_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            train = directory / "train.txt"
            validation = directory / "validation.txt"
            test = directory / "test.txt"
            checkpoint = directory / "best.pt"
            train.write_text(("Hello Sinhala ආයුබෝවන්. " * 8), encoding="utf-8")
            validation.write_text(("Validation සිංහල text. " * 4), encoding="utf-8")
            test.write_text(("Test set පරීක්ෂණය. " * 4), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable, "-m", "src.train",
                    "--train-corpus", str(train),
                    "--validation-corpus", str(validation),
                    "--test-corpus", str(test),
                    "--checkpoint", str(checkpoint),
                    "--epochs", "1",
                    "--max-steps", "1",
                    "--batch-size", "2",
                    "--context-length", "4",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
            self.assertIn("Held-out test loss (best checkpoint):", result.stdout)
            self.assertTrue(checkpoint.is_file())
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.assertIn("best_validation_loss", saved)
            self.assertEqual(saved["training"]["optimizer_steps"], 1)
            self.assertEqual(saved["training"]["test_tokens"], len(test.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
