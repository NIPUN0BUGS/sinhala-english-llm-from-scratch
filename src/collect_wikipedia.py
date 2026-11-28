"""Collect attributed Sinhala and English Wikipedia extracts for scheduled training."""
from __future__ import annotations
import argparse, json, time, urllib.error, urllib.parse, urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LICENSE = "CC BY-SA 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
USER_AGENT = "SinhalaEnglishLLMResearch/1.1 (dataset collection)"


def fetch_random(language: str) -> dict | None:
    """Fetch one namespace-zero article using the MediaWiki Action API."""
    if language not in {"si", "en"}:
        raise ValueError("language must be 'si' or 'en'")
    params = {
        "action": "query", "generator": "random", "grnnamespace": "0",
        "grnlimit": "1", "prop": "extracts|info", "inprop": "url",
        "explaintext": "1", "exintro": "1", "exchars": "5000",
        "format": "json", "formatversion": "2",
    }
    url = f"https://{language}.wikipedia.org/w/api.php?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"Skip {language} page: {exc}")
        return None
    pages = payload.get("query", {}).get("pages", [])
    if isinstance(pages, dict):
        pages = list(pages.values())
    for page in pages:
        text = " ".join(page.get("extract", "").split())
        title, page_url = page.get("title"), page.get("fullurl")
        if not title or not page_url or len(text) < 200:
            continue
        if urllib.parse.urlparse(page_url).hostname != f"{language}.wikipedia.org":
            continue
        return {
            "language": language, "title": title, "url": page_url,
            "license": LICENSE, "license_url": LICENSE_URL, "text": text,
            "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages-per-language", type=int, default=5)
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/self_learning_corpus.txt")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/raw/self_learning_manifest.jsonl")
    parser.add_argument("--max-attempts-per-page", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.pages_per_language <= 100:
        raise ValueError("pages-per-language must be between 1 and 100")
    if not 1 <= args.max_attempts_per_page <= 20:
        raise ValueError("max-attempts-per-page must be between 1 and 20")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    known = set()
    if args.manifest.exists():
        for line in args.manifest.read_text(encoding="utf-8").splitlines():
            try:
                known.add(json.loads(line)["url"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    counts, added = {"si": 0, "en": 0}, 0
    with args.output.open("a", encoding="utf-8") as corpus, args.manifest.open("a", encoding="utf-8") as manifest:
        for language in ("si", "en"):
            attempts = 0
            limit = args.pages_per_language * args.max_attempts_per_page
            while counts[language] < args.pages_per_language and attempts < limit:
                attempts += 1
                item = fetch_random(language)
                if item and item["url"] not in known:
                    corpus.write(f"\n\n[{language}] {item['title']}\n{item['text']}\n")
                    manifest.write(json.dumps(item, ensure_ascii=False) + "\n")
                    known.add(item["url"])
                    counts[language] += 1
                    added += 1
                time.sleep(0.5)
            print(f"{language}: collected {counts[language]}/{args.pages_per_language} new pages in {attempts} attempts")
    print(f"Added {added} new attributed extracts.")
    if min(counts.values()) == 0:
        raise SystemExit("Bilingual collection failed: no new pages for at least one language.")


if __name__ == "__main__":
    main()
