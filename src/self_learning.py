"""Continue from previous weights and compare checkpoints on an identical bilingual vocabulary."""
from __future__ import annotations
import argparse, json, math, shutil, subprocess, sys
from pathlib import Path
import torch
from torch import nn
from src.dataset import NextTokenDataset
from src.generate import load_model
from src.model import CausalTransformerLanguageModel
from src.safe_checkpoint import load_safetensors
from src.tokenizer import CharacterTokenizer

ROOT = Path(__file__).resolve().parents[1]


def validation_loss(checkpoint: Path, metadata: Path, corpus: Path, threads: int,
                    target_vocabulary: list[str] | None = None) -> float:
    saved_model, saved_tokenizer, config = load_model(checkpoint, metadata)
    vocabulary = target_vocabulary or saved_tokenizer._id_to_char
    if not set(saved_tokenizer._id_to_char).issubset(vocabulary):
        raise ValueError("Evaluation vocabulary must preserve all checkpoint tokens.")
    tokenizer = CharacterTokenizer(unknown_token=config["unknown_token"])
    tokenizer._id_to_char = list(vocabulary)
    tokenizer._char_to_id = {char: index for index, char in enumerate(vocabulary)}
    if len(vocabulary) == saved_model.vocab_size:
        model = saved_model
    else:
        model = CausalTransformerLanguageModel(
            vocab_size=len(vocabulary), embedding_dim=config["embedding_dim"],
            n_heads=config["heads"], n_layers=config["layers"],
            max_seq_len=config["context_length"], dropout=config["dropout"],
        )
        state = load_safetensors(checkpoint)
        for key in ("token_embedding.weight", "lm_head.weight"):
            expanded = model.state_dict()[key].clone()
            expanded[:state[key].shape[0]] = state[key]
            state[key] = expanded
        model.load_state_dict(state)
    ids = tokenizer.encode(corpus.read_text(encoding="utf-8"))
    validation_ids = ids[int(len(ids) * 0.9):]
    context = min(config["context_length"], len(validation_ids) - 1)
    if context < 1:
        raise ValueError("Validation corpus is too short.")
    dataset = NextTokenDataset(validation_ids, context)
    torch.set_num_threads(threads)
    criterion = nn.CrossEntropyLoss()
    total, count = 0.0, 0
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            x, y = dataset[index]
            logits = model(x.unsqueeze(0))
            loss = criterion(logits.reshape(-1, model.vocab_size), y.reshape(-1))
            total += float(loss) * y.numel()
            count += y.numel()
    return total / max(1, count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=ROOT / "data/raw/self_learning_corpus.txt")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/self_learning.safetensors")
    parser.add_argument("--metadata", type=Path, default=ROOT / "checkpoints/self_learning.metadata.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--min-improvement", type=float, default=0.0005)
    args = parser.parse_args()
    corpus_text = args.corpus.read_text(encoding="utf-8") if args.corpus.is_file() else ""
    if len(corpus_text.strip()) < 1000:
        raise SystemExit("At least 1,000 corpus characters are required before training.")
    candidate = args.checkpoint.with_name(args.checkpoint.stem + ".candidate.safetensors")
    candidate_meta = args.metadata.with_name(args.metadata.stem + ".candidate.json")
    baseline = args.checkpoint.is_file() and args.metadata.is_file()
    old_meta = json.loads(args.metadata.read_text(encoding="utf-8")) if baseline else None
    target_vocabulary = list(old_meta["vocabulary"]) if old_meta else sorted(set(corpus_text) | {"�"})
    if old_meta:
        target_vocabulary.extend(sorted(set(corpus_text).difference(target_vocabulary)))
        old_loss = validation_loss(args.checkpoint, args.metadata, args.corpus, args.threads, target_vocabulary)
    else:
        old_loss = math.inf
    command = [sys.executable, "-m", "src.train", "--corpus", str(args.corpus),
        "--checkpoint", str(candidate), "--metadata", str(candidate_meta),
        "--epochs", str(args.epochs), "--threads", str(args.threads)]
    if baseline:
        command += ["--resume-checkpoint", str(args.checkpoint), "--resume-metadata", str(args.metadata)]
    subprocess.run(command, cwd=ROOT, check=True)
    candidate_metadata = json.loads(candidate_meta.read_text(encoding="utf-8"))
    new_loss = validation_loss(candidate, candidate_meta, args.corpus, args.threads,
                               candidate_metadata["vocabulary"])
    print(f"Baseline validation loss: {old_loss:.6f}")
    print(f"Candidate validation loss: {new_loss:.6f}")
    accepted = not baseline or new_loss < old_loss - args.min_improvement
    if accepted:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(candidate), str(args.checkpoint))
        shutil.move(str(candidate_meta), str(args.metadata))
        result = "promoted"
        print(f"Promoted improved checkpoint: {args.checkpoint}")
    else:
        candidate.unlink(missing_ok=True)
        candidate_meta.unlink(missing_ok=True)
        result = "rejected"
        print("No improvement; previous checkpoint retained.")
    report = {"result": result, "baseline_validation_loss": old_loss if baseline else None,
        "candidate_validation_loss": new_loss, "vocabulary_size": len(candidate_metadata["vocabulary"]),
        "corpus": str(args.corpus)}
    report_path = ROOT / "checkpoints/self_learning_latest.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Evaluation report: {report_path}")


if __name__ == "__main__":
    main()
