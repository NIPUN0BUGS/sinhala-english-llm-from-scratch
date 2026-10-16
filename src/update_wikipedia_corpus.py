"""Incrementally collect short text extracts from Sinhala and English Wikipedia.

Wikipedia text is CC BY-SA; keep this manifest for attribution and license tracking.
The script stores unique pages locally and never overwrites previously collected pages.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "raw" / "wikipedia_manifest.jsonl"
DEFAULT_TEXT = ROOT / "data" / "raw" / "external_wikipedia.txt"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
USER_AGENT = "SinhalaEnglishLLMResearchBot/1.0 (local educational dataset collector)"


def fetch_random_pages(language: str, limit: int, timeout: int = 20) -> list[dict[str, str]]:
    """Fetch a small number of random main-namespace article extracts."""
    endpoint = f"https://{language}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "random",
        "grnnamespace": "0",
        "grnlimit": str(limit),
        "prop": "extracts|info",
        "explaintext": "1",
        "exchars": "2500",
        "inprop": "url",
        "format": "json",
        "formatversion": "2",
    }
    request = urllib.request.Request(
        endpoint + "?" + urllib.parse.urlencode(params),
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    pages = payload.get("query", {}).get("pages", [])
    results: list[dict[str, str]] = []
    for page in pages:
        title = " ".join(str(page.get("title", "")).split())
        extract = " ".join(str(page.get("extract", "")).split())
        page_id = str(page.get("pageid", ""))
        url = str(page.get("fullurl", ""))
        if not page_id or len(extract) < 200 or not url.startswith("https://"):
            continue
        results.append({
            "page_id": page_id,
            "language": language,
            "title": title,
            "url": url,
            "license": "CC BY-SA 4.0; verify article attribution and applicable terms",
            "license_url": LICENSE_URL,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "text": extract,
        })
    return results


def load_manifest(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                if record.get("page_id") and record.get("text"):
                    records.append(record)
            except (json.JSONDecodeError, TypeError):
                continue
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--pages-per-language", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.pages_per_language <= 5:
        parser.error("--pages-per-language must be between 1 and 5")

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    existing = load_manifest(args.manifest)
    known_ids = {record["page_id"] for record in existing}
    added: list[dict[str, str]] = []
    errors: list[str] = []
    for language in ("si", "en"):
        try:
            for record in fetch_random_pages(language, args.pages_per_language):
                if record["page_id"] not in known_ids:
                    known_ids.add(record["page_id"])
                    added.append(record)
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"{language} Wikipedia: {type(exc).__name__}: {exc}")
        # Be polite to Wikimedia; never fire both language requests back-to-back.
        time.sleep(1)

    if added:
        with args.manifest.open("a", encoding="utf-8", newline="\n") as handle:
            for record in added:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        existing.extend(added)

    # One normalized article per line; preparation code deduplicates records.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in existing:
            language_name = "Sinhala" if record.get("language") == "si" else "English"
            handle.write(f"{language_name} Wikipedia: {record['text']}\n")

    print(f"New Wikipedia pages added this cycle: {len(added)}")
    print(f"Unique stored Wikipedia pages: {len(existing)}")
    print(f"External text corpus: {args.output}")
    print(f"Attribution/license manifest: {args.manifest}")
    for error in errors:
        print(f"WARNING: {error}")
    if not existing:
        print("WARNING: No Wikipedia extracts available yet; check internet access.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
