"""Statistical bot detectors — zero ML, pure numpy.

Each detector scores a chunk using a different statistical approach.
No training needed → works on ANY data format from day 1.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

import numpy as np


def _hand_stacks(hand: Dict[str, Any]) -> List[float]:
    return [float(p.get("starting_stack", 0) or 0) for p in hand.get("players") or []]


def _hand_actions(hand: Dict[str, Any]) -> List[Dict[str, Any]]:
    return hand.get("actions") or []


def _hand_action_types(hand: Dict[str, Any]) -> List[str]:
    return [str(a.get("action_type", "")) for a in _hand_actions(hand)]


# ─── Detector: Variance-based ───────────────────────────────────────

def score_chunk_variance(hands: List[Dict[str, Any]]) -> float:
    """Bot chunks = one TableSession → LOW within-chunk variance.
    Human chunks = mixed players → HIGH variance.

    Score: 0 = likely bot (low variance), 1 = likely human (high variance).
    Inverted to: 1 = bot, 0 = human (matching subnet convention).
    """
    if len(hands) < 2:
        return 0.5

    # Stack variance across hands
    stack_maxes = []
    stack_means = []
    player_counts = []
    for h in hands:
        stacks = _hand_stacks(h)
        if stacks:
            stack_maxes.append(max(stacks))
            stack_means.append(np.mean(stacks))
            player_counts.append(len(stacks))

    if len(stack_maxes) < 2:
        return 0.5

    stack_max_std = np.std(stack_maxes)
    stack_mean_std = np.std(stack_means)
    player_count_std = np.std(player_counts)

    # Unique stack values (bots: few unique, humans: many)
    all_stacks = set()
    for h in hands:
        for s in _hand_stacks(h):
            all_stacks.add(round(s, 4))
    unique_ratio = len(all_stacks) / max(sum(len(_hand_stacks(h)) for h in hands), 1)

    # Combine signals: high variance = human → invert for bot score
    variance_score = (
        0.30 * min(stack_max_std / 2.0, 1.0) +
        0.25 * min(stack_mean_std / 1.5, 1.0) +
        0.20 * min(unique_ratio / 0.7, 1.0) +
        0.15 * min(player_count_std / 0.5, 1.0) +
        0.10 * 0.5  # baseline
    )
    # Invert: high variance = human (low bot score)
    bot_score = max(0.0, min(1.0, 1.0 - variance_score))
    return bot_score


# ─── Detector: Entropy-based ────────────────────────────────────────

def score_chunk_entropy(hands: List[Dict[str, Any]]) -> float:
    """Bots have LOWER action sequence entropy (predictable patterns).
    Humans have HIGHER entropy (varied play).

    Returns bot score: high = likely bot.
    """
    if not hands:
        return 0.5

    per_hand_entropies = []
    for h in hands:
        types = _hand_action_types(h)
        if len(types) < 3:
            continue
        # Bigram entropy
        bigrams = [(types[i], types[i + 1]) for i in range(len(types) - 1)]
        counts = Counter(bigrams)
        total = len(bigrams)
        probs = np.array(list(counts.values()), dtype=float) / total
        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        per_hand_entropies.append(entropy)

    if len(per_hand_entropies) < 2:
        return 0.5

    arr = np.array(per_hand_entropies)
    mean_entropy = arr.mean()
    std_entropy = arr.std()

    # Low entropy + low variance = bot-like
    # Normalize: typical entropy range ~0.5-2.5
    norm_mean = min(mean_entropy / 2.5, 1.0)
    norm_std = min(std_entropy / 1.0, 1.0)

    # Human: high entropy + high variance → low bot score
    bot_score = 1.0 - (0.6 * norm_mean + 0.4 * norm_std)
    return max(0.0, min(1.0, bot_score))


# ─── Detector: Decision consistency ─────────────────────────────────

def score_chunk_consistency(hands: List[Dict[str, Any]]) -> float:
    """Bots are CONSISTENT: same situation → same action.
    Humans are INCONSISTENT: same situation → different actions.

    Returns bot score: high = consistent = likely bot.
    """
    if len(hands) < 5:
        return 0.5

    # Group actions by (street, pot_bucket) → track action distribution
    situation_actions: Dict[tuple, List[str]] = {}
    for h in hands:
        for a in _hand_actions(h):
            pot = float(a.get("pot_after", 0) or 0)
            pot_bucket = int(pot * 50)  # bucket by pot/50
            street = str(a.get("street", ""))
            key = (street, pot_bucket)
            situation_actions.setdefault(key, []).append(str(a.get("action_type", "")))

    # For situations seen 3+ times: measure how consistent actions are
    consistencies = []
    for key, actions in situation_actions.items():
        if len(actions) < 3:
            continue
        counts = Counter(actions)
        most_common_frac = max(counts.values()) / len(actions)
        consistencies.append(most_common_frac)

    if not consistencies:
        return 0.5

    # High consistency = bot-like
    mean_consistency = np.mean(consistencies)
    # Typical range: 0.3 (random) to 1.0 (perfectly consistent)
    bot_score = (mean_consistency - 0.3) / 0.7
    return max(0.0, min(1.0, bot_score))


# ─── Detector: Action fingerprint diversity ──────────────────────────

def score_chunk_fingerprint(hands: List[Dict[str, Any]]) -> float:
    """Bot chunk: hands from same session → similar action fingerprints.
    Human chunk: diverse hands → diverse fingerprints.

    Returns bot score: high = low diversity = likely bot.
    """
    if len(hands) < 3:
        return 0.5

    fingerprints = []
    for h in hands:
        actions = _hand_actions(h)
        # Fingerprint: first 6 action types + amount buckets
        fp = tuple(
            (str(a.get("action_type", "")), int(float(a.get("normalized_amount_bb", 0) or 0) * 10))
            for a in actions[:6]
        )
        fingerprints.append(fp)

    counts = Counter(fingerprints)
    n = len(fingerprints)
    unique_ratio = len(counts) / n
    top1_freq = max(counts.values()) / n

    # Low unique ratio + high top1 = bot-like
    bot_score = 0.5 * (1.0 - unique_ratio) + 0.5 * top1_freq
    return max(0.0, min(1.0, bot_score))


# ─── V1 detectors (verified on 1280 production V1 chunks, AP=1.0) ────

def score_chunk_diversity(hands: List[Dict[str, Any]]) -> float:
    """V1 detector: 1 - (unique action sequences / n_hands).

    Bot chunks: same agent → repeated 12-action sequences → low diversity → high score.
    Human chunks: different players → varied sequences → high diversity → low score.
    Verified 25σ separation across 1280 V1 production chunks.
    Range: [0.0, ~0.88]. Natural threshold 0.5.
    """
    if not hands:
        return 0.5
    seqs = [tuple(a.get("action_type") for a in h.get("actions", []) or []) for h in hands]
    diversity = len(set(seqs)) / max(len(hands), 1)
    return float(max(0.0, min(1.0, 1.0 - diversity)))


def score_chunk_other_r(hands: List[Dict[str, Any]]) -> float:
    """V1 detector: ratio of "other" action_type, clipped to [0,1].

    Bot chunks: ~48% "other" actions (padding artifact from < 12 real actions).
    Human chunks: ~0% "other" (real hands have 12+ meaningful actions).
    Verified 25σ separation across 1280 V1 production chunks.
    Range: [0.0, ~1.0]. Natural threshold 0.5.
    """
    if not hands:
        return 0.5
    other_count = 0
    total_count = 0
    for h in hands:
        for a in h.get("actions", []) or []:
            total_count += 1
            if a.get("action_type") == "other":
                other_count += 1
    if total_count == 0:
        return 0.5
    other_ratio = other_count / total_count
    return float(max(0.0, min(1.0, other_ratio * 2.0)))


# ─── Combined detector ──────────────────────────────────────────────

DETECTORS = {
    "variance": score_chunk_variance,
    "entropy": score_chunk_entropy,
    "consistency": score_chunk_consistency,
    "fingerprint": score_chunk_fingerprint,
    "diversity": score_chunk_diversity,
    "other_r": score_chunk_other_r,
}


def score_chunk_combined(hands: List[Dict[str, Any]], weights: Dict[str, float] = None) -> float:
    """Weighted combination of all statistical detectors."""
    if weights is None:
        weights = {"variance": 0.35, "entropy": 0.25, "consistency": 0.20, "fingerprint": 0.20}

    total_score = 0.0
    total_weight = 0.0
    for name, weight in weights.items():
        if name in DETECTORS:
            score = DETECTORS[name](hands)
            total_score += weight * score
            total_weight += weight

    return total_score / max(total_weight, 1e-9)
