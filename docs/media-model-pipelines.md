# ZEE media dataset and model-training plan

This plan keeps text LLM training separate from image, video/cartoon, and voice workflows.

## Rules before using a media asset

1. Store metadata and source URLs in Git; do not commit large media collections or redistribute datasets whose terms prohibit re-hosting.
2. Record the exact asset URL, creator, licence, attribution text, retrieval date, and allowed uses.
3. Require commercial-use permission if ZEE may be monetized. Exclude noncommercial-only assets from commercial training.
4. Respect each dataset's terms, privacy restrictions, robots/terms where applicable, and takedown requests.
5. Do not identify people from voice datasets or clone a real person's voice without permission.
6. Never automatically promote a fine-tuned model. Evaluate quality, safety, memorization, and licence compliance first.

## Candidate sources

- Images: Wikimedia Commons. Licence is per asset; discover with Commons APIs and verify metadata before downloading.
- Cartoon artwork: Anita Dataset is useful for research, but includes CC BY-NC-SA material. Do not use noncommercial items in a commercial ZEE pipeline. Review each asset's individual licence.
- Video: Prelinger Archives Open dataset provides a curated subset described as public domain, CC0, CC BY, or CC BY-SA. Verify item-level metadata and any restrictions before use.
- Voice/audio: Mozilla Common Voice. Use only the relevant language release and follow its dataset terms. The Sinhala Spontaneous Speech dataset is CC0 but prohibits speaker identification and re-hosting/redistribution.
- Additional sources: Openverse and other public repositories may help discover assets, but discovery is not proof of licence. Always validate the original item.

## Separate pipelines

### 1. Image pipeline
Discover licensed image metadata -> filter by licence and quality -> optionally download to temporary storage -> create captions/metadata -> fine-tune an existing image model only on a GPU runner/service -> evaluate -> publish model version and manifest.

### 2. Video/cartoon pipeline
Start with licensed video clips and animation frames. Extract frames and captions only when permitted. Begin with data curation, scene/character consistency evaluation, and Blender-based assembly. Fine-tuning a video foundation model requires substantially more GPU memory and storage; do not run it on the current CPU-only machine.

### 3. Voice pipeline
Use licensed speech with its permitted transcript/metadata for speech recognition or general speech research. For text-to-speech, use a model explicitly licensed for that purpose and a voice with suitable consent/licensing. Keep speaker identification disabled. Do not assume an ASR dataset is suitable for TTS training.

## CI cadence and compute

- Daily/weekly: discover and validate metadata; keep a small manifest in Git.
- On demand: run curated dataset preparation and GPU fine-tuning.
- Every training run must have a budget, pinned model/dataset versions, fixed evaluation set, and a candidate-versus-baseline comparison.
- Publish only model artifacts that pass evaluation. Keep the last approved model if a run fails.
- Do not put API keys or GPU credentials in the repository. Use GitHub Actions secrets and least-privilege permissions.

## Current hardware constraint

The development computer (Intel i3-10100, 12 GB RAM, Intel UHD 630, no NVIDIA GPU) can curate small datasets and run lightweight checks, but is not suitable for training image/video foundation models from scratch. GitHub Actions can orchestrate remote GPU jobs; it does not itself guarantee GPU availability.

## Initial implementation scope

The first safe milestone is metadata discovery and licence auditing, not downloading entire collections or automatically training from arbitrary web content. Add GPU training only after choosing the exact base model, training method, dataset licences, and compute provider.
