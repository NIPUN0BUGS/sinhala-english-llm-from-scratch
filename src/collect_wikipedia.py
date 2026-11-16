"""Collect attributed Sinhala and English Wikipedia extracts for scheduled training."""
from __future__ import annotations
import argparse, json, time, urllib.error, urllib.request
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
LICENSE = "CC BY-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"

def fetch_random(language: str) -> dict | None:
    url = f"https://{language}.wikipedia.org/api/rest_v1/page/random/summary"
    request = urllib.request.Request(url, headers={"User-Agent": "SinhalaEnglishLLMResearch/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Skip {language} page: {exc}")
        return None
    text = " ".join(payload.get("extract", "").split())
    title = payload.get("title")
    page_url = payload.get("content_urls", {}).get("desktop", {}).get("page")
    if not title or not page_url or len(text) < 200:
        return None
    return {"language": language, "title": title, "url": page_url, "license": LICENSE,
            "license_url": LICENSE_URL, "text": text,
            "collected_at_utc": datetime.now(timezone.utc).isoformat()}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages-per-language", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/self_learning_corpus.txt")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/raw/self_learning_manifest.jsonl")
    args = parser.parse_args()
    if not 1 <= args.pages_per_language <= 100:
        raise ValueError("pages-per-language must be between 1 and 100")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    known = set()
    if args.manifest.exists():
        for line in args.manifest.read_text(encoding="utf-8").splitlines():
            try: known.add(json.loads(line)["url"])
            except (json.JSONDecodeError, KeyError): continue
    added = 0
    with args.output.open("a", encoding="utf-8") as corpus, args.manifest.open("a", encoding="utf-8") as manifest:
        for language in ("si", "en"):
            for _ in range(args.pages_per_language):
                item = fetch_random(language)
                if item and item["url"] not in known:
                    corpus.write(f"\n\n[{language}] {item['title']}\n{item['text']}\n")
                    manifest.write(json.dumps(item, ensure_ascii=False) + "\n")
                    known.add(item["url"]); added += 1
                time.sleep(0.4)
    print(f"Added {added} new attributed extracts.")
    if added == 0: print("No new pages collected; existing corpus unchanged.")

if __name__ == "__main__":
    main()
