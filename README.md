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

## Resume training from a checkpoint

The trainer can continue from a checkpoint produced by this project. Resume requires
the same model configuration and exact training vocabulary. The optimizer state is
restored when the checkpoint contains it; older checkpoints without optimizer state
resume the model weights with a fresh AdamW optimizer.

For example, to continue the existing character-level run using the same source corpus:

```powershell
python -m src.train --corpus data/raw/train_from_dev.txt --checkpoint checkpoints/steps2000.pt --resume checkpoints/steps2000.pt --epochs 20 --batch-size 8 --context-length 64
```

Use the same corpus and model settings used to create the checkpoint. If the script
reports a vocabulary or configuration mismatch, stop rather than forcing incompatible
weights into the model. `--epochs` is the number of additional epochs in this run.
The checkpoint is updated only when validation loss improves. The trainer does not
yet implement automatic early stopping.

## Train a smoke-test model

```powershell
python -m src.train
```

Optional smaller CPU run:

```powershell
python -m src.train --epochs 3 --batch-size 4 --context-length 32
```

The default checkpoint is saved to `checkpoints/tiny_transformer.pt`. Checkpoints are excluded from Git by `.gitignore`.

## Audit corpus quality and checkpoints before training

Run the audit before spending more time training. It reports line counts, duplicate
records, blank lines, Unicode replacement/mojibake markers, Sinhala/Latin character
shares, line-length distribution, and Unicode normalization. It can also compare
exact line overlap between supplied splits and inspect saved checkpoint metadata.
It does not upload or modify your data.

Audit the original corpus and current 2,000-step checkpoint:

```powershell
python -m src.audit_corpus --input data/raw/train_from_dev.txt --checkpoint checkpoints/steps2000.pt
```

If processed splits already exist, audit them and check for exact duplicate lines
across splits:

```powershell
python -m src.audit_corpus --input data/processed/train.txt --validation data/processed/validation.txt --test data/processed/test.txt --checkpoint checkpoints/steps2000.pt
```

Keep the JSON output. Do not train more steps until the report has been reviewed.
The current checkpoint's saved validation loss is useful only if it was trained on
a genuine held-out validation set; a sequential split from one concatenated text
file can produce misleadingly optimistic results.

Run the audit tests:

```powershell
python -m unittest discover -s tests -p "test_audit_corpus.py" -v
```

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

## Test the model in the deployed ZEE website

The ZEE chat UI can send messages to this model running on your own Windows PC.
This keeps PyTorch and the checkpoint off Vercel and avoids uploading the model.

1. Train a checkpoint first. If you used the split-training command above, it is
   `checkpoints/best.pt`. If you used the default smoke test, use
   `checkpoints/tiny_transformer.pt` in the next command instead.
2. In PowerShell, from the LLM repository root, start the local API:

   ```powershell
   python -m src.serve --checkpoint checkpoints/best.pt --host 127.0.0.1 --port 8766
   ```

3. Open [ZEE Tiny LLM test mode](https://zee-bay.vercel.app/?tinyLLM=1) in the same
   computer/browser where the API is running. Send a message in the chat.
4. To return to normal ZEE mode, open [ZEE](https://zee-bay.vercel.app/) without
   the `?tinyLLM=1` query parameter.

The local API binds only to loopback and allows browser requests from the ZEE
production origin and Vite localhost origins. Keep the terminal running while
testing. Stop it with Ctrl+C. This is an integration smoke test: the current
small character-level model may produce incoherent text until it is trained on a
much larger, high-quality Sinhala-English corpus.

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


## From-scratch tokenizer and CPU benchmark

The corpus cleaner keeps adjacent `Sinhala:` and `English:` lines together as a single record before splitting, avoiding bilingual pair leakage across train/validation/test. Outputs are local and ignored by Git.

Run from the repository root in PowerShell:

```powershell
python -m src.prepare_corpus --input data/raw/train_from_dev.txt --output-dir data/processed --seed 42
python -m pip install sentencepiece
python -m src.train_bpe_tokenizer --input data/processed/train.txt --vocab-size 8000
python -m src.benchmark_cpu --corpus data/processed/train.txt --tokens 100000
```

The BPE tokenizer is trained from scratch and does not use a pretrained model. It is currently a separate artifact: the existing training model still uses a character tokenizer, so don't treat BPE token counts as equivalent to the benchmark's character-token counts.

The CPU benchmark initializes a fresh random model and loads no checkpoint. It measures training throughput and writes `benchmark-results/cpu-benchmark.json`; its 5M/10M-hour figures are rough throughput extrapolations, not estimates of how long convergence will take. Use the benchmark's actual processed-token count and tokens/second when deciding whether to scale up.

Run tests:

```powershell
python -m unittest discover -s tests -v
```


## Hourly corpus growth and training (Windows)

The PowerShell runner now attempts to collect up to two new article extracts per language
from Sinhala and English Wikipedia on each hourly cycle, using the MediaWiki API. It
stores unique page IDs, source URLs, fetch timestamps, and license metadata in
`data/raw/wikipedia_manifest.jsonl`. Text is stored in
`data/raw/external_wikipedia.txt`; the seed corpus and collected text are combined into
`data/raw/hourly_corpus.txt`, then the train/validation/test splits are rebuilt before
tests and training. New content is only collected when internet access and the Wikimedia
API are available. If no new pages are found, existing data is retained.

Wikipedia text is generally available under CC BY-SA 4.0; preserve attribution and review
the [Wikimedia Terms of Use](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use)
and [CC BY-SA 4.0 license](https://creativecommons.org/licenses/by-sa/4.0/) before
redistributing collected data or model outputs. The manifest keeps source URLs and license
information; it is not a substitute for reviewing each source's applicable terms.

Before resuming a checkpoint while the corpus grows, the trainer now reuses the checkpoint's
existing character vocabulary. Previously unseen characters are mapped to the unknown
token; this avoids a vocabulary-size mismatch but means the current character-level model
does not learn new character IDs. SentencePiece BPE is still not integrated into the model.

Run the hourly loop manually if Task Scheduler registration is blocked by Windows permissions:

```powershell
cd D:\llm\my-llm-git
git pull --ff-only origin feature/tokenizer-cleaning-cpu-benchmark
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\hourly_train_and_test.ps1
```

Keep that PowerShell window open and the PC awake. The script sleeps for one hour between
cycles. It adds data and retries on the next cycle, but it does **not** automatically repair
source-code failures. Logs are written to `training-logs`.
