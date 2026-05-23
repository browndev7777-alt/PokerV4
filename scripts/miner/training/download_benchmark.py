#!/usr/bin/env python3
"""Download all labeled benchmark chunks from the Poker44 API.

Saves each outer-chunk as a JSON file under data/benchmark/.
Each file contains the inner chunks (hands) and groundTruth labels aligned 1-to-1.

Usage (run from repo root):
    python scripts/miner/training/download_benchmark.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests is not installed. Run: pip install requests")

BASE = "https://api.poker44.net/api/v1/benchmark"
OUT_DIR = Path(__file__).resolve().parents[3] / "data" / "benchmark"


def fetch_releases() -> list[dict]:
    resp = requests.get(f"{BASE}/releases", timeout=30)
    resp.raise_for_status()
    return resp.json()["data"]["releases"]


def fetch_chunks_for_date(source_date: str) -> list[dict]:
    resp = requests.get(f"{BASE}/chunks", params={"sourceDate": source_date}, timeout=60)
    resp.raise_for_status()
    return resp.json()["data"]["chunks"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    releases = fetch_releases()
    print(f"Found {len(releases)} benchmark releases.")

    total_outer = 0
    total_inner = 0

    for release in releases:
        source_date = release["sourceDate"]
        chunk_count = release.get("chunkCount", "?")
        print(f"\nDownloading {source_date}  (outer chunks: {chunk_count}) ...")

        try:
            outer_chunks = fetch_chunks_for_date(source_date)
        except Exception as exc:
            print(f"  ERROR: {exc} — skipping {source_date}")
            continue

        for outer in outer_chunks:
            chunk_id = outer["chunkId"]
            out_file = OUT_DIR / f"{source_date}_{chunk_id[:8]}.json"

            if out_file.exists():
                print(f"  skip (exists): {out_file.name}")
                continue

            inner_chunks = outer.get("chunks", [])
            ground_truth = outer.get("groundTruth", [])

            if len(inner_chunks) != len(ground_truth):
                print(f"  WARNING: chunk/label mismatch for {chunk_id} — skipping")
                continue

            out_file.write_text(
                json.dumps(outer, separators=(",", ":")), encoding="utf-8"
            )
            total_inner += len(inner_chunks)
            print(f"  saved {out_file.name}  ({len(inner_chunks)} inner chunks)")

        total_outer += len(outer_chunks)
        time.sleep(0.5)

    print(f"\nDone. {total_outer} outer chunks, {total_inner} inner scoring units saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
