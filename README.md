# Sinhala–English Language Model From Scratch

An educational project for building a small Sinhala–English language-model pipeline from the ground up.

## Current scope

- Character-level tokenizer with Unicode Sinhala support
- Dataset preparation for next-character prediction
- A small PyTorch neural language model (embedding + feed-forward layers)
- Training loop with checkpoint saving
- Basic tokenizer tests

> **Important:** This is a learning baseline, not a production chatbot or a Transformer yet. The included corpus is only a tiny smoke-test dataset. Useful generation requires a much larger, properly licensed bilingual corpus and further model work.

## Requirements

- Python 3.12 recommended
- PyTorch
- NumPy

## Setup on Windows

From PowerShell:

```powershell
cd D:\llm\my-llm
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch numpy
```

Copy or merge these project files into your local project folder first. Do not overwrite existing files without reviewing them.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

## Train

```powershell
python -m src.train
```

The starter trainer saves a checkpoint under `checkpoints/`.

## Project layout

```text
src/
  tokenizer.py
  dataset.py
  model.py
  train.py
data/raw/
  train.txt
tests/
  test_tokenizer.py
```

## Roadmap

1. Validate tokenizer and dataset pipeline.
2. Collect and clean a larger licensed Sinhala–English dataset.
3. Add a proper train/validation split and evaluation.
4. Implement a causal Transformer with attention and positional embeddings.
5. Track checkpoints and experiments; only then build an inference API or chat UI.

## Data and responsible use

Use only data you have permission to process and redistribute. Keep secrets, tokens, personal information, local environments, model checkpoints, and large datasets out of Git.
