"""Local HTTP API for testing the tiny model from the deployed ZEE website.

Run after training:
    python -m src.serve --checkpoint checkpoints/best.pt --host 127.0.0.1 --port 8766

Only the configured website origin and localhost development origins are allowed.
"""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch

from src.model import TinyLanguageModel
from src.tokenizer import CharacterTokenizer, SentencePieceBPETokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = ROOT / "checkpoints" / "best.pt"
ALLOWED_ORIGINS = {
    "https://zee-bay.vercel.app",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
}


def load_model(checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train a model first."
        )
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("tokenizer_type", "character") == "bpe":
        tokenizer = SentencePieceBPETokenizer(checkpoint["tokenizer_model"])
    else:
        tokenizer = CharacterTokenizer.from_vocabulary(
            checkpoint["vocabulary"],
            unknown_token=checkpoint.get("unknown_token", "�"),
        )
    model = TinyLanguageModel(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, top_k: int) -> str:
    if not prompt.strip():
        raise ValueError("prompt cannot be empty")
    if not 1 <= max_new_tokens <= 200:
        raise ValueError("max_new_tokens must be between 1 and 200")
    if not 0.1 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0.1 and 2.0")
    if not 0 <= top_k <= 100:
        raise ValueError("top_k must be between 0 and 100")

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        prompt_ids = tokenizer.encode(tokenizer.unknown_token)
    generated = torch.tensor([prompt_ids], dtype=torch.long)
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            context = generated[:, -model.max_seq_len :]
            logits = model(context)[:, -1, :] / temperature
            if top_k:
                k = min(top_k, logits.shape[-1])
                threshold = torch.topk(logits, k, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_id), dim=1)
    all_text = tokenizer.decode(generated[0].tolist())
    decoded_prompt = tokenizer.decode(prompt_ids)
    return all_text[len(decoded_prompt):].strip() or "(The model did not generate new text.)"


def make_handler(model, tokenizer):
    class Handler(BaseHTTPRequestHandler):
        server_version = "TinySinhalaEnglishLLM/0.1"

        def _headers(self, status: int, origin: str | None = None):
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            if origin in ALLOWED_ORIGINS:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Private-Network", "true")
            self.end_headers()

        def _json(self, status: int, payload: dict, origin: str | None = None):
            self._headers(status, origin)
            self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

        def do_OPTIONS(self):
            origin = self.headers.get("Origin")
            if origin not in ALLOWED_ORIGINS:
                self._json(403, {"error": "Origin not allowed"})
                return
            self._headers(204, origin)

        def do_GET(self):
            origin = self.headers.get("Origin")
            if urlparse(self.path).path == "/health":
                self._json(200, {"status": "ok", "model": "tiny-transformer"}, origin)
            else:
                self._json(404, {"error": "Not found"}, origin)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if origin not in ALLOWED_ORIGINS:
                self._json(403, {"error": "Origin not allowed"})
                return
            if urlparse(self.path).path != "/generate":
                self._json(404, {"error": "Not found"}, origin)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > 16_384:
                    raise ValueError("Request body must be between 1 and 16384 bytes")
                payload = json.loads(self.rfile.read(length))
                prompt = payload.get("prompt")
                if not isinstance(prompt, str) or len(prompt) > 2000:
                    raise ValueError("prompt must be a string of at most 2000 characters")
                answer = generate(
                    model,
                    tokenizer,
                    prompt,
                    int(payload.get("max_new_tokens", 120)),
                    float(payload.get("temperature", 0.8)),
                    int(payload.get("top_k", 20)),
                )
                self._json(200, {
                    "message": answer,
                    "intent": "answer_question",
                    "actionExecuted": False,
                    "model": "tiny-transformer",
                }, origin)
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self._json(400, {"error": str(error)}, origin)
            except Exception:
                self._json(500, {"error": "Generation failed. Check the local server terminal."}, origin)

        def log_message(self, format, *args):
            print(f"[tiny-llm-api] {self.address_string()} - {format % args}")

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    model, tokenizer = load_model(args.checkpoint)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model, tokenizer))
    print(f"Tiny LLM API ready at http://{args.host}:{args.port}")
    print("Health check: http://127.0.0.1:%d/health" % args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping tiny LLM API.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
