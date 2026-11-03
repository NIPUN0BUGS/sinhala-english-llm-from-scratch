import tempfile
import unittest
from pathlib import Path

import torch

from src.model import CausalTransformerLanguageModel
from src.provenance import sha256_file
from src.safe_checkpoint import load_safetensors, save_safetensors
from src.tokenizer import CharacterTokenizer


class SecurityAndQualityTests(unittest.TestCase):
    def test_sinhala_english_round_trip(self):
        text = "Hello! ආයුබෝවන්! How are you? ඔයාට කොහොමද?"
        tokenizer = CharacterTokenizer().fit(text)
        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_safe_checkpoint_round_trip_and_reject_pickle_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            expected = {"weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
            save_safetensors(expected, path)
            actual = load_safetensors(path)
            self.assertTrue(torch.equal(expected["weight"], actual["weight"]))
            with self.assertRaises(ValueError):
                load_safetensors(Path(directory) / "checkpoint.pt")

    def test_safe_checkpoint_supports_tied_model_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tied-model.safetensors"
            model = CausalTransformerLanguageModel(
                vocab_size=16, embedding_dim=16, n_heads=4, n_layers=1,
                max_seq_len=8, dropout=0.0,
            )
            self.assertEqual(
                model.token_embedding.weight.data_ptr(),
                model.lm_head.weight.data_ptr(),
            )
            save_safetensors(model.state_dict(), path)
            restored = CausalTransformerLanguageModel(
                vocab_size=16, embedding_dim=16, n_heads=4, n_layers=1,
                max_seq_len=8, dropout=0.0,
            )
            restored.load_state_dict(load_safetensors(path))
            self.assertTrue(torch.equal(
                model.token_embedding.weight, restored.token_embedding.weight
            ))
            self.assertTrue(torch.equal(
                model.lm_head.weight, restored.lm_head.weight
            ))

    def test_dataset_hash_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.txt"
            path.write_text("සිංහල English", encoding="utf-8")
            self.assertEqual(sha256_file(path), sha256_file(path))
            self.assertEqual(len(sha256_file(path)), 64)

    def test_cpu_memory_smoke_budget(self):
        model = torch.nn.Embedding(128, 16)
        tokens = torch.zeros((2, 32), dtype=torch.long)
        output = model(tokens)
        self.assertEqual(tuple(output.shape), (2, 32, 16))
        self.assertLess(output.numel() * output.element_size(), 1_000_000)


if __name__ == "__main__":
    unittest.main()
