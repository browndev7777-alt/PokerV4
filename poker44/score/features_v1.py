"""V1 feature extraction — optimized for Poker44 V1 eval format.

Key differences from V0:
- starting_stack is always 1.0 (normalized) → stack features are DEAD
- Variable chunk sizes (1-70 hands)
- Variable actions per hand (not always 12)
- hero_seat is preserved (new signal)
- "other" action_type is the dominant bot signal

Focus: within-chunk homogeneity + action patterns + hero_seat consistency.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np


def _safe_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "range": 0.0, "cv": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    mn = float(arr.min())
    mx = float(arr.max())
    return {
        "mean": mean,
        "std": std,
        "min": mn,
        "max": mx,
        "range": mx - mn,
        "cv": std / (abs(mean) + 1e-9),
    }


def extract_v1_features(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    """Extract V1-optimized features from a sanitized chunk."""
    if not hands:
        return {}

    feats: Dict[str, float] = {}
    n_hands = len(hands)

    # === Chunk-level action statistics ===
    all_actions = []
    per_hand_action_counts = []
    per_hand_unique_actions = []
    per_hand_other_ratios = []
    per_hand_fold_ratios = []
    per_hand_raise_ratios = []
    per_hand_call_ratios = []
    per_hand_check_ratios = []
    per_hand_bet_ratios = []
    per_hand_allin_ratios = []
    per_hand_n_streets = []
    per_hand_n_players = []
    per_hand_amount_stds = []
    per_hand_pot_after_maxs = []
    hero_seats = []
    action_seqs = []

    for h in hands:
        actions = h.get("actions") or []
        players = h.get("players") or []
        streets = h.get("streets") or []
        metadata = h.get("metadata") or {}

        all_actions.extend(actions)
        per_hand_action_counts.append(len(actions))
        per_hand_n_streets.append(len(streets))
        per_hand_n_players.append(len(players))

        types = [a.get("action_type", "") for a in actions]
        cnt = Counter(types)
        n = max(len(actions), 1)

        per_hand_unique_actions.append(len(set(types)))
        per_hand_other_ratios.append(cnt.get("other", 0) / n)
        per_hand_fold_ratios.append(cnt.get("fold", 0) / n)
        per_hand_raise_ratios.append(cnt.get("raise", 0) / n)
        per_hand_call_ratios.append(cnt.get("call", 0) / n)
        per_hand_check_ratios.append(cnt.get("check", 0) / n)
        per_hand_bet_ratios.append(cnt.get("bet", 0) / n)
        per_hand_allin_ratios.append(cnt.get("all_in", 0) / n)

        amounts = [float(a.get("normalized_amount_bb", 0.0)) for a in actions]
        per_hand_amount_stds.append(float(np.std(amounts)) if amounts else 0.0)

        pots = [float(a.get("pot_after", 0.0)) for a in actions]
        per_hand_pot_after_maxs.append(max(pots) if pots else 0.0)

        hero_seats.append(int(metadata.get("hero_seat", 0) or 0))
        action_seqs.append(tuple(types))

    total_actions = len(all_actions)
    if total_actions == 0:
        return {}

    # === Dominant V1 signal: other_ratio ===
    total_other = sum(1 for a in all_actions if a.get("action_type") == "other")
    feats["other_ratio"] = total_other / total_actions
    feats["other_ratio_x2"] = min(1.0, feats["other_ratio"] * 2.0)

    # === Diversity signal ===
    unique_seqs = len(set(action_seqs))
    feats["diversity"] = unique_seqs / max(n_hands, 1)
    feats["inv_diversity"] = 1.0 - feats["diversity"]

    # === Per-hand variance signals (within-chunk homogeneity) ===
    for name, vals in [
        ("n_actions", per_hand_action_counts),
        ("unique_actions", per_hand_unique_actions),
        ("other_r", per_hand_other_ratios),
        ("fold_r", per_hand_fold_ratios),
        ("raise_r", per_hand_raise_ratios),
        ("call_r", per_hand_call_ratios),
        ("check_r", per_hand_check_ratios),
        ("bet_r", per_hand_bet_ratios),
        ("allin_r", per_hand_allin_ratios),
        ("n_streets", per_hand_n_streets),
        ("n_players", per_hand_n_players),
        ("amount_std", per_hand_amount_stds),
        ("pot_after_max", per_hand_pot_after_maxs),
    ]:
        stats = _safe_stats(vals)
        for stat_name, val in stats.items():
            feats[f"hand_{name}_{stat_name}"] = val

    # === Hero seat consistency ===
    hero_stats = _safe_stats([float(x) for x in hero_seats])
    for stat_name, val in hero_stats.items():
        feats[f"hero_seat_{stat_name}"] = val
    feats["hero_seat_unique_ratio"] = len(set(hero_seats)) / max(n_hands, 1)

    # === Global ratios ===
    cnt = Counter(a.get("action_type", "") for a in all_actions)
    total = max(sum(cnt.values()), 1)
    feats["global_fold_ratio"] = cnt.get("fold", 0) / total
    feats["global_raise_ratio"] = cnt.get("raise", 0) / total
    feats["global_call_ratio"] = cnt.get("call", 0) / total
    feats["global_check_ratio"] = cnt.get("check", 0) / total
    feats["global_bet_ratio"] = cnt.get("bet", 0) / total
    feats["global_allin_ratio"] = cnt.get("all_in", 0) / total
    feats["global_other_ratio"] = cnt.get("other", 0) / total

    # === Amount statistics ===
    amounts = [float(a.get("normalized_amount_bb", 0.0)) for a in all_actions]
    if amounts:
        feats["global_amount_mean"] = float(np.mean(amounts))
        feats["global_amount_std"] = float(np.std(amounts))
        feats["global_amount_max"] = float(max(amounts))
        # Bucket entropy
        buckets = [int(min(a / 2.0, 10)) for a in amounts]
        bc = Counter(buckets)
        probs = np.asarray(list(bc.values()), dtype=np.float64) / len(buckets)
        feats["amount_bucket_entropy"] = float(-(probs * np.log(probs + 1e-12)).sum())
    else:
        feats["global_amount_mean"] = 0.0
        feats["global_amount_std"] = 0.0
        feats["global_amount_max"] = 0.0
        feats["amount_bucket_entropy"] = 0.0

    # === Pot progression ===
    pot_afters = [float(a.get("pot_after", 0.0)) for a in all_actions]
    if pot_afters:
        feats["global_pot_max"] = max(pot_afters)
        feats["global_pot_mean"] = float(np.mean(pot_afters))
    else:
        feats["global_pot_max"] = 0.0
        feats["global_pot_mean"] = 0.0

    # === Chunk size ===
    feats["chunk_size"] = float(n_hands)
    feats["chunk_size_log"] = math.log1p(n_hands)

    # === Street distribution ===
    street_counts = Counter(len(h.get("streets") or []) for h in hands)
    for i in range(5):
        feats[f"street_count_{i}_frac"] = street_counts.get(i, 0) / n_hands

    # === Player count distribution ===
    player_counts = Counter(len(h.get("players") or []) for h in hands)
    for i in range(2, 8):
        feats[f"player_count_{i}_frac"] = player_counts.get(i, 0) / n_hands

    # === Action sequence entropy (bigram) ===
    bigrams = []
    for h in hands:
        types = [a.get("action_type", "") for a in h.get("actions", [])]
        for a, b in zip(types[:-1], types[1:]):
            bigrams.append((a, b))
    if bigrams:
        bc = Counter(bigrams)
        probs = np.asarray(list(bc.values()), dtype=np.float64) / len(bigrams)
        feats["bigram_entropy"] = float(-(probs * np.log(probs + 1e-12)).sum())
    else:
        feats["bigram_entropy"] = 0.0

    # === Fingerprint entropy (first 8 actions per hand) ===
    fingerprints = []
    for h in hands:
        actions = h.get("actions") or []
        fp = tuple((a.get("action_type", ""), int(float(a.get("normalized_amount_bb", 0)) * 10)) for a in actions[:8])
        fingerprints.append(fp)
    if fingerprints:
        fc = Counter(fingerprints)
        probs = np.asarray(list(fc.values()), dtype=np.float64) / len(fingerprints)
        feats["fingerprint_entropy"] = float(-(probs * np.log(probs + 1e-12)).sum())
        feats["fingerprint_unique_ratio"] = len(fc) / len(fingerprints)
    else:
        feats["fingerprint_entropy"] = 0.0
        feats["fingerprint_unique_ratio"] = 0.0

    return feats


def chunks_to_matrix(chunks: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Convert list of {"hands": [...], "is_bot": bool} to (X, y, feature_names)."""
    rows: List[Dict[str, float]] = []
    labels: List[int] = []
    for c in chunks:
        feats = extract_v1_features(c["hands"])
        rows.append(feats)
        labels.append(1 if c.get("is_bot", False) else 0)

    if not rows:
        return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.int64), []

    all_names = sorted(rows[0].keys())
    X = np.zeros((len(rows), len(all_names)), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, name in enumerate(all_names):
            X[i, j] = float(r.get(name, 0.0))
    y = np.asarray(labels, dtype=np.int64)
    return X, y, all_names
