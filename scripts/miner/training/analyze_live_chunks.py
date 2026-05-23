#!/usr/bin/env python3
"""Analyze captured live validator chunks and compare to benchmark.

Run after live chunks have been saved (requires POKER44_SAVE_RAW_CHUNKS=1 on miners).
Shows which features differ between live and benchmark data so you can retrain.

Usage:
    python scripts/miner/training/analyze_live_chunks.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np
from poker44.score.features_v1 import extract_v1_features

LIVE_DIR = REPO / "data" / "miner_training" / "raw_validator_chunks"
BENCH_DIR = REPO / "data" / "benchmark"


def load_live_chunks() -> list[dict]:
    files = sorted(LIVE_DIR.glob("chunks_*.json"))
    if not files:
        return []
    chunks = []
    for f in files:
        data = json.loads(f.read_text())
        for chunk in data.get("chunks", []):
            chunks.append(chunk)
    return chunks


def load_benchmark_chunks() -> tuple[list, list]:
    """Returns (bot_chunks, human_chunks)."""
    bots, humans = [], []
    for f in sorted(BENCH_DIR.glob("*.json"))[:20]:
        outer = json.loads(f.read_text())
        for chunk, label in zip(outer["chunks"], outer["groundTruth"]):
            if label == 1:
                bots.append(chunk)
            else:
                humans.append(chunk)
    return bots, humans


def feature_stats(chunks: list) -> dict[str, float]:
    if not chunks:
        return {}
    all_feats = [extract_v1_features(c) for c in chunks]
    keys = set().union(*all_feats)
    return {k: float(np.mean([f.get(k, 0) for f in all_feats])) for k in sorted(keys)}


def main() -> None:
    live_chunks = load_live_chunks()
    bench_bots, bench_humans = load_benchmark_chunks()

    print(f"Live chunks captured: {len(live_chunks)}")
    print(f"Benchmark bot chunks: {len(bench_bots)}")
    print(f"Benchmark human chunks: {len(bench_humans)}")

    if not live_chunks:
        print("\nNo live chunks yet. Wait for validators to query your miners.")
        print(f"They will be saved to: {LIVE_DIR}")
        return

    live_stats = feature_stats(live_chunks)
    bot_stats = feature_stats(bench_bots)
    human_stats = feature_stats(bench_humans)

    key_features = [
        "other_ratio", "global_other_ratio", "hand_other_r_mean",
        "diversity", "fingerprint_unique_ratio", "inv_diversity",
        "bigram_entropy", "fingerprint_entropy", "chunk_size",
        "global_fold_ratio", "global_raise_ratio", "global_call_ratio",
        "hand_n_actions_std", "hand_n_actions_mean",
    ]

    print(f"\n{'Feature':<30}  {'Bench BOT':>10}  {'Bench HUM':>10}  {'LIVE':>10}  {'live~bot?':>10}")
    print("-" * 78)
    for feat in key_features:
        b = bot_stats.get(feat, 0)
        h = human_stats.get(feat, 0)
        l = live_stats.get(feat, 0)
        # Is live closer to bot or human?
        closer = "BOT" if abs(l - b) < abs(l - h) else "human"
        print(f"{feat:<30}  {b:>10.4f}  {h:>10.4f}  {l:>10.4f}  {closer:>10}")

    print(f"\nLive data n_hands per chunk stats:")
    sizes = [len(c) for c in live_chunks]
    print(f"  min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.1f}")

    print("\nAction type distribution in live data:")
    from collections import Counter
    all_types = Counter()
    for chunk in live_chunks:
        for hand in chunk:
            for action in hand.get("actions", []):
                all_types[action.get("action_type", "?")] += 1
    total = max(sum(all_types.values()), 1)
    for t, cnt in all_types.most_common():
        print(f"  {t:<12} {cnt:>6}  ({cnt/total:.1%})")


if __name__ == "__main__":
    main()
