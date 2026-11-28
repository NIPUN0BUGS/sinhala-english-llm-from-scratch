"""Fast, attributed collection from reviewed Hugging Face datasets.

This runs on GitHub Actions; it never reads files from a developer's PC.
The Hugging Face Dataset Viewer API serves at most 100 rows per request.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://datasets-server.huggingface.co"
USER_AGENT = "ZEE-Sinhala-English-LLM-data-collector/1.0 (GitHub Actions)"
LICENSES = {
    "CC BY-SA 4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC BY 4.0": "https://creativecommons.org/licenses/by/4.0/",
}

# Dataset names, splits and licences are deliberately allow-listed after review.
# Keep each source's original licence and attribution with every extracted row.
SOURCES = (
    {
        "dataset": "Minuri/nsina-sentences-raw",
        "config": "default",
        "split": "train",
        "language": "si",
        "license": "CC BY-SA 4.0",
        "attribution": "Minuri; original source sinhala-nlp/NSINA (Hettiarachchi et al., 2024)",
        "description": "Sinhala NSINA-derived corpus",
        "target_chars": 550_000,
    },
    {
        "dataset": "keshan/lk_hansard_trilingual",
        "config": "default",
        "split": "sinhala",
        "language": "si",
        "license": "CC BY 4.0",
        "attribution": "keshan; Sri Lanka Parliament Hansard source material",
        "description": "Sri Lanka Parliament Hansard (Sinhala)",
        "target_chars": 150_000,
    },
    {
        "dataset": "keshan/lk_hansard_trilingual",
        "config": "default",
        "split": "english",
        "language": "en",
        "license": "CC BY 4.0",
        "attribution": "keshan; Sri Lanka Parliament Hansard source material",
        "description": "Sri Lanka Parliament Hansard (English)",
        "target_chars": 550_000,
    },
)


def request_json(url: str, retries: int = 5) -> dict:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt >= retries:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                delay = min(60.0, max(1.0, float(retry_after))) if retry_after else min(30.0, 2 ** (attempt + 1))
            except (TypeError, ValueError):
                delay = min(30.0, 2 ** (attempt + 1))
            print(f"Dataset API HTTP {exc.code}; retrying in {delay:.0f}s")
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError):
            if attempt >= retries:
                raise
            delay = min(30.0, 2 ** (attempt + 1))
            time.sleep(delay)
    raise RuntimeError("Dataset API retries exhausted")


def dataset_splits(dataset: str) -> set[tuple[str, str]]:
    url = f"{API}/splits?{urllib.parse.urlencode({'dataset': dataset})}"
    payload = request_json(url)
    return {(item["config"], item["split"]) for item in payload.get("splits", [])}


def fetch_page(source: dict, offset: int) -> dict:
    params = {
        "dataset": source["dataset"],
        "config": source["config"],
        "split": source["split"],
        "offset": str(offset),
        "length": "100",
    }
    url = f"{API}/rows?{urllib.parse.urlencode(params)}"
    return request_json(url)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


def language_ratio(text: str, language: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    if language == "si":
        relevant = sum("\u0D80" <= char <= "\u0DFF" for char in letters)
    else:
        relevant = sum("A" <= char <= "Z" or "a" <= char <= "z" for char in letters)
    return relevant / len(letters)


def text_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).casefold().encode("utf-8")).hexdigest()


def load_existing(manifest_path: Path) -> tuple[set[str], set[str]]:
    hashes, urls = set(), set()
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if isinstance(item.get("text"), str):
                    hashes.add(text_hash(item["text"]))
                if item.get("url"):
                    urls.add(item["url"])
            except (json.JSONDecodeError, TypeError):
                continue
    return hashes, urls


def collect_source(
    source: dict,
    target_chars: int,
    max_rows: int,
    hashes: set[str],
    output,
    manifest,
    workers: int,
) -> int:
    dataset = source["dataset"]
    expected = (source["config"], source["split"])
    available = dataset_splits(dataset)
    if expected not in available:
        print(f"SKIP {dataset}/{expected}: split not published; found {sorted(available)}")
        return 0

    collected_chars = 0
    collected_rows = 0
    source_url = f"https://huggingface.co/datasets/{dataset}"
    print(f"Collecting {source['language']} from {dataset}/{source['split']} ({source['license']})")

    for batch_start in range(0, max_rows, workers * 100):
        offsets = list(range(batch_start, min(batch_start + workers * 100, max_rows), 100))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            pages = list(pool.map(lambda offset: fetch_page(source, offset), offsets))
        for payload in pages:
            for wrapped in payload.get("rows", []):
                row = wrapped.get("row", {})
                raw = row.get("text")
                if not isinstance(raw, str):
                    continue
                text = normalize_text(raw)
                if len(text) < 80 or language_ratio(text, source["language"]) < 0.35:
                    continue
                text = text[:8000]
                digest = text_hash(text)
                if digest in hashes:
                    continue
                hashes.add(digest)
                row_id = wrapped.get("row_idx", collected_rows)
                title = f"{source['description']} row {row_id}"
                record = {
                    "language": source["language"],
                    "title": title,
                    "url": source_url,
                    "license": source["license"],
                    "license_url": LICENSES[source["license"]],
                    "text": text,
                    "collected_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_dataset": dataset,
                    "source_split": source["split"],
                    "source_row_id": row_id,
                    "source_attribution": source["attribution"],
                    "source_metadata": {
                        key: str(row[key])[:500]
                        for key in ("source", "file_name", "source_folder")
                        if row.get(key) is not None
                    },
                }
                output.write(f"\n\n[{source['language']}] {title}\n{text}\n")
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                collected_chars += len(text)
                collected_rows += 1
                if collected_chars >= target_chars:
                    break
            if collected_chars >= target_chars:
                break
        output.flush()
        manifest.flush()
        print(f"  {dataset}/{source['split']}: {collected_chars:,} new chars, {collected_rows:,} rows")
        if collected_chars >= target_chars:
            break
    print(f"Finished {dataset}/{source['split']}: {collected_chars:,} chars")
    return collected_chars


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-chars-per-source", type=int, default=550_000)
    parser.add_argument("--max-rows-per-source", type=int, default=12_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/self_learning_corpus.txt")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/raw/self_learning_manifest.jsonl")
    args = parser.parse_args()
    if args.target_chars_per_source < 1 or args.max_rows_per_source < 100 or args.workers not in range(1, 17):
        raise ValueError("Invalid target, max rows, or worker count (1-8).")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    hashes, _ = load_existing(args.manifest)
    totals = {"si": 0, "en": 0}
    with args.output.open("a", encoding="utf-8") as output, args.manifest.open("a", encoding="utf-8") as manifest:
        for source in SOURCES:
            added = collect_source(
                source, source.get("target_chars", args.target_chars_per_source), args.max_rows_per_source,
                hashes, output, manifest, args.workers,
            )
            totals[source["language"]] += added

    print(f"Open-dataset collection totals: Sinhala={totals['si']:,}; English={totals['en']:,} characters")
    if sum(totals.values()) == 0:
        print("No new Hugging Face rows were collected; existing corpus remains intact.")


if __name__ == "__main__":
    main()
