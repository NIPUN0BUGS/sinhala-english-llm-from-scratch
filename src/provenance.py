"""Write a small dataset provenance manifest with source/licence fields and SHA-256."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(dataset: Path) -> dict:
    try:
        relative_path = str(dataset.resolve().relative_to(ROOT))
    except ValueError:
        relative_path = dataset.name
    existing = {}
    manifest_path = ROOT / "data" / "provenance.json"
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}
    previous = existing.get("datasets", {}).get(relative_path, {})
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": {
            relative_path: {
                "sha256": sha256_file(dataset),
                "bytes": dataset.stat().st_size,
                "source_url": previous.get("source_url", "REQUIRED: add the original source URL"),
                "license": previous.get("license", "REQUIRED: verify and record the dataset licence"),
                "version": previous.get("version", "local-file; record upstream version or commit"),
                "notes": previous.get("notes", "Do not claim provenance/licensing until verified."),
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    manifest = build_manifest(args.dataset)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote provenance and SHA-256 manifest: {args.manifest}")


if __name__ == "__main__":
    main()
