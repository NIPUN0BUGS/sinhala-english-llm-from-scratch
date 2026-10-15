# Sinhala–English Language Model From Scratch

A low-cost, CPU-first educational project for experimenting with a bilingual Sinhala-English language model.

## Current implementation

- Unicode character tokenizer for Sinhala and English
- Next-character training windows
- Small decoder-only causal Transformer (token + position embeddings, masked self-attention, feed-forward blocks)
- Train/validation split and per-epoch loss reporting
- Checkpoint saving and command-line text generation
- Unit tests for tokenization, model output shape, and causal masking

> **Important:** This is a learning prototype, not a capable general-purpose chatbot yet. The included corpus is a tiny smoke-test dataset. It is only useful for checking that the pipeline runs; it is not enough to teach the model Sinhala or English. Real capability requires a much larger, legally usable, carefully cleaned bilingual corpus and systematic evaluation.

## Hardware and cost

The default configuration is intentionally small for CPU experiments (96-dimensional embeddings, 2 Transformer layers, 4 attention heads). Training on a CPU will still be slow compared with a GPU, and increasing the dataset or model size increases runtime. No paid API is required for local experiments.

## Requirements

- Python 3.12 recommended
- PyTorch

## Setup on Windows

From PowerShell, in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that same terminal, then activate again.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Train with separate corpus splits

After preparing `data/processed/train.txt`, `validation.txt`, and `test.txt`,
train using the explicit splits:

```powershell
python -m src.train --train-corpus data/processed/train.txt --validation-corpus data/processed/validation.txt --test-corpus data/processed/test.txt --checkpoint checkpoints/best.pt --epochs 20 --batch-size 8 --context-length 64
```

The tokenizer is fitted only on the training text. Validation loss chooses which
checkpoint is saved; the held-out test loss is reported once after training using
that best checkpoint. Do not tune model settings against the test loss. The older
`--corpus` option remains available for quick smoke tests, but it does not use a
separate held-out test set.

## Train a smoke-test model

```powershell
python -m src.train
```

Optional smaller CPU run:

```powershell
python -m src.train --epochs 3 --batch-size 4 --context-length 32
```

The default checkpoint is saved to `checkpoints/tiny_transformer.pt`. Checkpoints are excluded from Git by `.gitignore`.

## Prepare a clean corpus

Use one independent record per non-empty line. Keep a Sinhala-English pair on the
same line (for example, separated by a tab) so the pair cannot be split apart.
The preparation tool normalizes Unicode to NFC, collapses whitespace, removes
blank lines and exact duplicate records, then creates deterministic train,
validation, and test files. It reports Unicode replacement characters so encoding
problems are visible.

```powershell
python -m src.prepare_corpus --input data/raw/train_from_dev.txt --output-dir data/processed --seed 42
```

Default split: 90% train, 5% validation, 5% test. The test split must remain held
out: do not use it for training or tuning. The tool reads UTF-8 text only; convert
licensed source datasets to the documented line-based format separately. Do not
use a dataset's devtest/test partition as training material.

Run the corpus utility tests with:

```powershell
python -m unittest discover -s tests -p "test_prepare_corpus.py" -v
```

## Generate text

After training:

```powershell
python -m src.generate --prompt "ආයුබෝවන්" --max-new-tokens 100
python -m src.generate --prompt "Hello" --max-new-tokens 100
```

Useful options include `--temperature 0.8` and `--top-k 20`. The generated text from the toy dataset will likely be incoherent; generation is included to test the end-to-end pipeline.

## Project layout

```text
src/
  tokenizer.py
  dataset.py
  model.py
  train.py
  generate.py
data/raw/
  train.txt
tests/
  test_tokenizer.py
  test_model.py
```

## Roadmap

1. Run unit tests and a short local smoke test.
2. Replace the toy corpus with a larger, permission-cleared Sinhala-English dataset.
3. Add corpus validation, deduplication, data documentation, and held-out evaluation.
4. Improve tokenization beyond character-level if experiments justify the added complexity.
5. Tune model size to available hardware and compare checkpoints using measured validation results.
6. Build an inference API and website only after the local model has demonstrated useful quality.

## Data and responsible use

Use only data you have permission to process and redistribute. Record dataset sources and licenses. Keep secrets, tokens, personal information, local environments, model checkpoints, and large datasets out of Git.
