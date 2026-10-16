import unittest

import torch

from src.model import TinyLanguageModel


class TinyLanguageModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = TinyLanguageModel(
            vocab_size=32,
            embedding_dim=16,
            num_heads=4,
            num_layers=2,
            max_seq_len=8,
            dropout=0.0,
        )
        self.model.eval()

    def test_logits_have_expected_shape(self) -> None:
        tokens = torch.randint(0, 32, (2, 5))
        logits = self.model(tokens)
        self.assertEqual(tuple(logits.shape), (2, 5, 32))

    def test_future_tokens_do_not_change_earlier_logits(self) -> None:
        original = torch.tensor([[1, 2, 3, 4]])
        changed_future = torch.tensor([[1, 2, 9, 8]])
        with torch.no_grad():
            original_logits = self.model(original)
            changed_logits = self.model(changed_future)
        # The first two positions cannot attend to tokens at positions 2 or 3.
        self.assertTrue(
            torch.allclose(original_logits[:, :2], changed_logits[:, :2], atol=1e-6)
        )

    def test_rejects_sequence_longer_than_context(self) -> None:
        tokens = torch.randint(0, 32, (1, 9))
        with self.assertRaises(ValueError):
            self.model(tokens)


if __name__ == "__main__":
    unittest.main()
