"""Encode raw sanitized poker hands into tensors for neural net inference.

Converts a hand dict (as received from validator) into a fixed-size tensor:
  [12 actions × 8 features] + [5 hand-level features] = flat vector

Handles variable action counts (pad/truncate to 12).
Works with both V0 sanitized hands and V1 eval hands.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

# Action type encoding (one-hot index)
ACTION_MAP = {
    "fold": 0, "check": 1, "call": 2,
    "bet": 3, "raise": 4, "all_in": 5,
    "small_blind": 2, "big_blind": 2,  # treat as call-like
    "ante": 2, "other": 1,
}
N_ACTION_TYPES = 6
MAX_ACTIONS = 12
FEATURES_PER_ACTION = N_ACTION_TYPES + 2  # one-hot + amount_bb + pot_ratio = 8
HAND_LEVEL_FEATURES = 5  # hero_seat, n_players, n_streets, stack_mean, stack_std
HAND_VECTOR_DIM = MAX_ACTIONS * FEATURES_PER_ACTION + HAND_LEVEL_FEATURES  # 12*8 + 5 = 101


def encode_action(action: Dict[str, Any], prev_pot: float) -> np.ndarray:
    """Encode single action into [8] vector: one-hot(6) + amount_bb(1) + pot_ratio(1)."""
    vec = np.zeros(FEATURES_PER_ACTION, dtype=np.float32)

    # One-hot action type
    atype = str(action.get("action_type", "other")).strip().lower()
    idx = ACTION_MAP.get(atype, 1)
    vec[idx] = 1.0

    # Normalized amount
    amount_bb = float(action.get("normalized_amount_bb", 0.0) or 0.0)
    vec[N_ACTION_TYPES] = min(amount_bb / 50.0, 1.0)  # normalize to ~[0,1]

    # Pot progression ratio
    pot_after = float(action.get("pot_after", 0.0) or 0.0)
    if prev_pot > 0:
        vec[N_ACTION_TYPES + 1] = min(pot_after / prev_pot, 5.0) / 5.0
    else:
        vec[N_ACTION_TYPES + 1] = 0.0

    return vec


def encode_hand(hand: Dict[str, Any]) -> np.ndarray:
    """Encode one sanitized hand into flat [101] vector."""
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []

    # Action sequence: pad/truncate to MAX_ACTIONS
    action_matrix = np.zeros((MAX_ACTIONS, FEATURES_PER_ACTION), dtype=np.float32)
    prev_pot = 0.0
    for i in range(min(len(actions), MAX_ACTIONS)):
        action_matrix[i] = encode_action(actions[i], prev_pot)
        prev_pot = float(actions[i].get("pot_after", 0.0) or 0.0)

    # Hand-level features
    stacks = [float(p.get("starting_stack", 0.0) or 0.0) for p in players]
    hand_feats = np.zeros(HAND_LEVEL_FEATURES, dtype=np.float32)
    hand_feats[0] = float(hand.get("metadata", {}).get("hero_seat", 0) or 0) / 6.0
    hand_feats[1] = len(players) / 6.0
    hand_feats[2] = len(streets) / 4.0
    hand_feats[3] = (np.mean(stacks) / 5.0 if stacks else 0.0)
    hand_feats[4] = (np.std(stacks) / 3.0 if len(stacks) > 1 else 0.0)

    # Flatten: [12*8 + 5] = [101]
    return np.concatenate([action_matrix.flatten(), hand_feats])


def encode_chunk(hands: List[Dict[str, Any]]) -> torch.Tensor:
    """Encode chunk of N hands into [N, 101] tensor."""
    if not hands:
        return torch.zeros(1, HAND_VECTOR_DIM)
    vectors = [encode_hand(h) for h in hands]
    return torch.tensor(np.stack(vectors), dtype=torch.float32)


def encode_chunk_batch(chunks: List[List[Dict[str, Any]]], max_hands: int = 70) -> torch.Tensor:
    """Encode batch of chunks into [B, max_hands, 101] padded tensor + mask."""
    B = len(chunks)
    batch = torch.zeros(B, max_hands, HAND_VECTOR_DIM)
    mask = torch.zeros(B, max_hands, dtype=torch.bool)

    for i, hands in enumerate(chunks):
        n = min(len(hands), max_hands)
        for j in range(n):
            batch[i, j] = torch.tensor(encode_hand(hands[j]))
            mask[i, j] = True

    return batch, mask
