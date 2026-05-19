"""Feature extraction for Poker44 sanitized chunks.

Focus: features that survive validator sanitization AND capture within-chunk
homogeneity (the fundamental design weakness — bot chunks are one TableSession,
human chunks are a mix of real games).
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np

FEATURE_NAMES: List[str] = []


def _clear() -> None:
    FEATURE_NAMES.clear()


def _register(name: str, value: float) -> Tuple[str, float]:
    FEATURE_NAMES.append(name)
    return name, value


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


def _hand_level_signals(hand: Dict[str, Any]) -> Dict[str, float]:
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    counts = Counter(a.get("action_type") for a in actions)

    unique_action_tuples = len({
        (a.get("action_type"), a.get("actor_seat"), round(float(a.get("normalized_amount_bb", 0.0)), 2))
        for a in actions
    })
    unique_amount_bb = len({round(float(a.get("normalized_amount_bb", 0.0)), 2) for a in actions})
    stacks = [float(p.get("starting_stack", 0.0)) for p in players]

    amount_bb_vals = [float(a.get("normalized_amount_bb", 0.0)) for a in actions]
    pot_after_vals = [float(a.get("pot_after", 0.0)) for a in actions]

    return {
        "n_actions": float(len(actions)),
        "n_players": float(len(players)),
        "n_streets": float(len(streets)),
        "uniq_action_tuples": float(unique_action_tuples),
        "uniq_amount_bb": float(unique_amount_bb),
        "dup_ratio_action_tuples": 1.0 - unique_action_tuples / max(len(actions), 1),
        "call_ratio": counts.get("call", 0) / max(len(actions), 1),
        "check_ratio": counts.get("check", 0) / max(len(actions), 1),
        "raise_ratio": counts.get("raise", 0) / max(len(actions), 1),
        "fold_ratio": counts.get("fold", 0) / max(len(actions), 1),
        "bet_ratio": counts.get("bet", 0) / max(len(actions), 1),
        "allin_ratio": counts.get("all_in", 0) / max(len(actions), 1),
        "stack_mean": float(np.mean(stacks)) if stacks else 0.0,
        "stack_std": float(np.std(stacks)) if stacks else 0.0,
        "stack_max": float(max(stacks)) if stacks else 0.0,
        "stack_min": float(min(stacks)) if stacks else 0.0,
        "amount_bb_mean": float(np.mean(amount_bb_vals)) if amount_bb_vals else 0.0,
        "amount_bb_std": float(np.std(amount_bb_vals)) if amount_bb_vals else 0.0,
        "amount_bb_max": float(max(amount_bb_vals)) if amount_bb_vals else 0.0,
        "pot_after_max": float(max(pot_after_vals)) if pot_after_vals else 0.0,
        "pot_after_mean": float(np.mean(pot_after_vals)) if pot_after_vals else 0.0,
    }


def _street_distribution(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    bucket = {"preflop_only": 0, "flop": 0, "turn": 0, "river": 0}
    for h in hands:
        n = len(h.get("streets") or [])
        if n <= 0:
            bucket["preflop_only"] += 1
        elif n == 1:
            bucket["flop"] += 1
        elif n == 2:
            bucket["turn"] += 1
        else:
            bucket["river"] += 1
    total = max(len(hands), 1)
    return {f"street_{k}_frac": v / total for k, v in bucket.items()}


def _player_count_distribution(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    bucket = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    for h in hands:
        n = len(h.get("players") or [])
        if n >= 7:
            bucket[7] += 1
        elif n in bucket:
            bucket[n] += 1
    total = max(len(hands), 1)
    return {f"players_{k}_frac": v / total for k, v in bucket.items()}


def _action_ngrams(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    """Per-chunk distribution of action bigrams. Bots have more predictable transitions."""
    transitions = Counter()
    total = 0
    for h in hands:
        actions = h.get("actions") or []
        types = [a.get("action_type", "") for a in actions]
        for a, b in zip(types[:-1], types[1:]):
            transitions[(a, b)] += 1
            total += 1
    total = max(total, 1)
    core = ["call", "check", "raise", "fold", "bet"]
    out: Dict[str, float] = {}
    for a in core:
        for b in core:
            out[f"bigram_{a}_{b}"] = transitions.get((a, b), 0) / total
    out["bigram_entropy"] = 0.0
    if transitions:
        probs = np.asarray(list(transitions.values()), dtype=np.float64) / total
        out["bigram_entropy"] = float(-(probs * np.log(probs + 1e-12)).sum())
    return out


def _amount_bucket_distribution(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    """Coarse histogram of normalized_amount_bb across chunk.

    Bots cluster around pot-fraction-derived sizes; humans are more scattered.
    """
    buckets = [0, 1, 2, 3, 5, 8, 15, 30, 75, 200]
    counts = [0] * (len(buckets) + 1)
    total = 0
    for h in hands:
        for a in h.get("actions") or []:
            v = float(a.get("normalized_amount_bb", 0.0))
            idx = 0
            for i, edge in enumerate(buckets):
                if v <= edge:
                    idx = i
                    break
            else:
                idx = len(buckets)
            counts[idx] += 1
            total += 1
    total = max(total, 1)
    return {f"amt_bucket_{i}": c / total for i, c in enumerate(counts)}


def _amount_second_decimal_entropy(hands: List[Dict[str, Any]]) -> float:
    """Bots often produce round pot fractions → discrete second-decimal distribution."""
    vals = []
    for h in hands:
        for a in h.get("actions") or []:
            v = float(a.get("normalized_amount_bb", 0.0))
            vals.append(round((v * 100) % 10, 0))
    if not vals:
        return 0.0
    counts = Counter(vals)
    total = len(vals)
    probs = np.asarray(list(counts.values()), dtype=np.float64) / total
    return float(-(probs * np.log(probs + 1e-12)).sum())


def _aggregate_stats(per_hand: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_hand:
        return {}
    keys = per_hand[0].keys()
    out: Dict[str, float] = {}
    for k in keys:
        values = [h[k] for h in per_hand]
        stats = _safe_stats(values)
        for stat_name, val in stats.items():
            out[f"hand_{k}_{stat_name}"] = val
    return out


def _per_hand_bot_scores(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    """Per-hand mini bot score → then variance across chunk.

    A bot chunk (one TableSession) should have CONSISTENT per-hand scores.
    A human chunk (mixed players) should have HIGH variance in per-hand scores.
    """
    if not hands:
        return {"hand_bot_score_mean": 0.5, "hand_bot_score_std": 0.0,
                "hand_bot_score_range": 0.0, "hand_bot_score_cv": 0.0}

    scores = []
    for h in hands:
        actions = h.get("actions") or []
        players = h.get("players") or []
        streets = h.get("streets") or []

        n_act = len(actions)
        n_players = len(players)
        n_streets = len(streets)

        # Mini heuristic per hand (richer than the old one)
        types = [a.get("action_type", "") for a in actions]
        amounts = [float(a.get("normalized_amount_bb", 0)) for a in actions]

        # Unique action types used
        unique_types = len(set(types))
        # Amount variance (bots: consistent sizing, humans: chaotic)
        amt_std = float(np.std(amounts)) if amounts else 0.0
        # Repeat ratio in 12-action window (bots: more repeats due to stride sampling)
        action_tuples = [(a.get("action_type"), a.get("actor_seat")) for a in actions]
        repeat_ratio = 1.0 - len(set(action_tuples)) / max(len(action_tuples), 1)

        # Combine into per-hand score [0, 1]
        s = 0.0
        s += 0.25 * min(unique_types / 5.0, 1.0)      # more unique = more human
        s += 0.25 * min(amt_std / 3.0, 1.0)            # higher variance = more human
        s += 0.25 * (1.0 - repeat_ratio)                # less repeats = more human
        s += 0.15 * min(n_streets / 3.0, 1.0)          # deeper = more varied
        s += 0.10 * min(n_players / 6.0, 1.0)          # more players = more complex
        scores.append(max(0.0, min(1.0, s)))

    arr = np.asarray(scores)
    mean = float(arr.mean())
    std = float(arr.std())
    return {
        "hand_bot_score_mean": mean,
        "hand_bot_score_std": std,
        "hand_bot_score_range": float(arr.max() - arr.min()),
        "hand_bot_score_cv": std / (abs(mean) + 1e-9),
    }


def _pot_progression(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    """Pot growth pattern across actions within each hand.

    Bots tend to have CONSISTENT pot progression ratios (algorithmic sizing).
    Humans have ERRATIC pot growth (emotional, tilted, creative).
    """
    all_ratios = []
    per_hand_ratio_stds = []

    for h in hands:
        actions = h.get("actions") or []
        pots = [float(a.get("pot_after", 0)) for a in actions]
        ratios = []
        for i in range(1, len(pots)):
            if pots[i - 1] > 0:
                ratios.append(pots[i] / pots[i - 1])
        all_ratios.extend(ratios)
        per_hand_ratio_stds.append(float(np.std(ratios)) if ratios else 0.0)

    if not all_ratios:
        return {"pot_ratio_mean": 0.0, "pot_ratio_std": 0.0,
                "pot_ratio_per_hand_std_mean": 0.0, "pot_ratio_per_hand_std_std": 0.0}

    arr = np.asarray(all_ratios)
    phs = np.asarray(per_hand_ratio_stds)
    return {
        "pot_ratio_mean": float(arr.mean()),
        "pot_ratio_std": float(arr.std()),
        "pot_ratio_per_hand_std_mean": float(phs.mean()),
        "pot_ratio_per_hand_std_std": float(phs.std()),
    }


def _decision_consistency(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    """Do players make the SAME decision in SIMILAR situations across the chunk?

    Bots: deterministic → facing same situation = same action (high consistency).
    Humans: stochastic → facing same situation = different action (low consistency).
    """
    # Group actions by (street, actor_seat, pot_bucket) → track action_type distribution
    situation_actions: Dict[tuple, List[str]] = {}
    for h in hands:
        for a in h.get("actions") or []:
            pot = float(a.get("pot_after", 0))
            pot_bucket = int(pot * 100) // 10  # bucket by pot/10
            key = (a.get("street", ""), a.get("actor_seat", 0), pot_bucket)
            situation_actions.setdefault(key, []).append(a.get("action_type", ""))

    # For situations seen 3+ times, measure entropy of action distribution
    entropies = []
    for key, action_list in situation_actions.items():
        if len(action_list) < 3:
            continue
        counts = Counter(action_list)
        total = len(action_list)
        probs = np.asarray(list(counts.values()), dtype=np.float64) / total
        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        entropies.append(entropy)

    if not entropies:
        return {"decision_entropy_mean": 0.0, "decision_entropy_std": 0.0,
                "decision_n_repeated_situations": 0.0}

    arr = np.asarray(entropies)
    return {
        "decision_entropy_mean": float(arr.mean()),
        "decision_entropy_std": float(arr.std()),
        "decision_n_repeated_situations": float(len(entropies)),
    }


def _action_pattern_entropy(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    """Per-hand action sequence as a "fingerprint" → entropy across chunk.

    Bot chunk: all hands from same TableSession → similar fingerprints → low entropy.
    Human chunk: diverse hands → diverse fingerprints → high entropy.
    """
    fingerprints = []
    for h in hands:
        actions = h.get("actions") or []
        # Fingerprint: tuple of (action_type, amount_bucket)
        fp = tuple(
            (a.get("action_type", ""), int(float(a.get("normalized_amount_bb", 0)) * 10))
            for a in actions[:8]  # first 8 actions only (avoid padding artifacts)
        )
        fingerprints.append(fp)

    if not fingerprints:
        return {"fingerprint_unique_ratio": 0.0, "fingerprint_entropy": 0.0,
                "fingerprint_top1_frequency": 0.0}

    counts = Counter(fingerprints)
    n = len(fingerprints)
    unique_ratio = len(counts) / n
    probs = np.asarray(list(counts.values()), dtype=np.float64) / n
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    top1_freq = max(counts.values()) / n

    return {
        "fingerprint_unique_ratio": unique_ratio,
        "fingerprint_entropy": entropy,
        "fingerprint_top1_frequency": top1_freq,
    }


def extract_chunk_features(hands: List[Dict[str, Any]]) -> Dict[str, float]:
    """Main entry: sanitized chunk → flat feature dict."""
    if not hands:
        return {}

    per_hand = [_hand_level_signals(h) for h in hands]
    feats: Dict[str, float] = {}

    feats.update(_aggregate_stats(per_hand))
    feats.update(_street_distribution(hands))
    feats.update(_player_count_distribution(hands))
    feats.update(_action_ngrams(hands))
    feats.update(_amount_bucket_distribution(hands))
    feats["amount_second_decimal_entropy"] = _amount_second_decimal_entropy(hands)
    feats["chunk_size"] = float(len(hands))

    # === V2 SEQUENTIAL FEATURES ===
    feats.update(_per_hand_bot_scores(hands))
    feats.update(_pot_progression(hands))
    feats.update(_decision_consistency(hands))
    feats.update(_action_pattern_entropy(hands))

    # Cross-chunk homogeneity: unique (n_actions, n_streets) pairs
    pairs = {(int(h["n_actions"]), int(h["n_streets"])) for h in per_hand}
    feats["unique_shape_pairs"] = float(len(pairs))
    feats["unique_shape_ratio"] = len(pairs) / max(len(hands), 1)

    # Unique starting stack values across chunk (bots: ~fixed per TableSession)
    stack_values = set()
    for h in hands:
        for p in h.get("players") or []:
            stack_values.add(round(float(p.get("starting_stack", 0.0)), 4))
    feats["unique_stack_values"] = float(len(stack_values))
    feats["unique_stack_ratio"] = len(stack_values) / max(sum(len(h.get("players") or []) for h in hands), 1)

    return feats


def features_to_matrix(chunks: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (X, y, feature_names) for an iterable of {"hands": ..., "is_bot": bool}."""
    rows: List[Dict[str, float]] = []
    labels: List[int] = []
    for c in chunks:
        feats = extract_chunk_features(c["hands"])
        rows.append(feats)
        labels.append(1 if c["is_bot"] else 0)

    if not rows:
        return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.int64), []

    all_names = sorted(rows[0].keys())
    X = np.zeros((len(rows), len(all_names)), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, name in enumerate(all_names):
            X[i, j] = float(r.get(name, 0.0))
    y = np.asarray(labels, dtype=np.int64)
    return X, y, all_names


if __name__ == "__main__":
    import pickle
    import sys

    if len(sys.argv) < 2:
        print("usage: python features.py <chunks.pkl>")
        sys.exit(1)
    with open(sys.argv[1], "rb") as f:
        chunks = pickle.load(f)
    X, y, names = features_to_matrix(chunks[:50])
    print(f"X shape={X.shape}, y positives={int(y.sum())}, features={len(names)}")
    print("sample feature names:", names[:12])
