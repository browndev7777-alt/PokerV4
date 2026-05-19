"""Per-hand scoring with different aggregation strategies.

Scores each hand independently, then combines into chunk score
using various aggregation methods. Independent of chunk size.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from poker44.models.hand_encoder import encode_hand
from poker44.models.cnn_detector import ChunkDetector


def _load_cnn_safe(model_path: str) -> ChunkDetector:
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        detector = ChunkDetector(hidden_dim=ckpt.get("hidden_dim", 64))
        detector.load_state_dict(ckpt["model_state"])
        detector.eval()
        return detector
    except Exception:
        return None


def get_per_hand_scores(hands: List[Dict[str, Any]], cnn: ChunkDetector) -> np.ndarray:
    """Score each hand independently using CNN. Returns [N] array."""
    if not hands or cnn is None:
        return np.array([0.5] * max(len(hands), 1))

    vecs = torch.tensor(
        np.stack([encode_hand(h) for h in hands]),
        dtype=torch.float32,
    )
    with torch.no_grad():
        scores = torch.sigmoid(cnn.hand_cnn(vecs)).squeeze(-1).numpy()
    return scores


# ─── Aggregation: Mean ───────────────────────────────────────────────

def aggregate_mean(hand_scores: np.ndarray) -> float:
    """Simple mean — balanced, robust."""
    return float(hand_scores.mean()) if len(hand_scores) > 0 else 0.5


# ─── Aggregation: Majority Vote ─────────────────────────────────────

def aggregate_vote(hand_scores: np.ndarray) -> float:
    """Each hand votes bot (>0.5) or human (<0.5). Majority wins.
    Returns fraction of bot votes as chunk score."""
    if len(hand_scores) == 0:
        return 0.5
    bot_votes = (hand_scores >= 0.5).sum()
    return float(bot_votes / len(hand_scores))


# ─── Aggregation: Max Confidence ─────────────────────────────────────

def aggregate_maxconf(hand_scores: np.ndarray) -> float:
    """The most confident hand decides.
    Hand furthest from 0.5 (either direction) determines chunk label."""
    if len(hand_scores) == 0:
        return 0.5
    distances = np.abs(hand_scores - 0.5)
    most_confident_idx = np.argmax(distances)
    return float(hand_scores[most_confident_idx])


# ─── Aggregation: Confidence-Weighted Mean ───────────────────────────

def aggregate_weighted(hand_scores: np.ndarray) -> float:
    """Hands with strong opinions (close to 0 or 1) weigh more.
    Uncertain hands (close to 0.5) weigh less."""
    if len(hand_scores) == 0:
        return 0.5
    confidences = np.abs(hand_scores - 0.5) * 2  # 0=uncertain, 1=confident
    weights = confidences + 0.1  # small floor to avoid zero weights
    return float(np.average(hand_scores, weights=weights))


AGGREGATORS = {
    "mean": aggregate_mean,
    "vote": aggregate_vote,
    "maxconf": aggregate_maxconf,
    "weighted": aggregate_weighted,
}


def score_chunk_perhand(
    hands: List[Dict[str, Any]],
    cnn: ChunkDetector,
    method: str = "mean",
) -> float:
    """Full pipeline: per-hand CNN → aggregate by method."""
    hand_scores = get_per_hand_scores(hands, cnn)
    aggregator = AGGREGATORS.get(method, aggregate_mean)
    return aggregator(hand_scores)
