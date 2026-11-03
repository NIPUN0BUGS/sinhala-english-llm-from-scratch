import unittest

import torch

from src.model import CausalTransformerLanguageModel


class CausalTransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = CausalTransformerLanguageModel(
            vocab_size=32, embedding_dim=16, n_heads=4, n_layers=2,
            max_seq_len=12, dropout=0.0,
        ).eval()

    def test_logits_shape(self):
        logits = self.model(torch.randint(0, 32, (2, 8)))
        self.assertEqual(tuple(logits.shape), (2, 8, 32))

    def test_causal_mask_prevents_future_token_leakage(self):
        first = torch.tensor([[1, 2, 3, 4, 5]])
        changed_future = torch.tensor([[1, 2, 3, 9, 8]])
        with torch.inference_mode():
            a = self.model(first)
            b = self.model(changed_future)
        self.assertTrue(torch.allclose(a[:, :3], b[:, :3], atol=1e-6))

    def test_rejects_context_longer_than_config(self):
        with self.assertRaises(ValueError):
            self.model(torch.ones((1, 13), dtype=torch.long))


if __name__ == "__main__":
    unittest.main()
