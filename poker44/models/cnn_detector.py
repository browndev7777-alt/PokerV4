"""1D-CNN bot detector for Poker44.

Architecture:
  Per-hand: [12 actions × 8 features] → Conv1D → Pool → FC → hand_score
  Per-chunk: mean(hand_scores) → chunk_score

Designed for variable chunk sizes (1-70 hands).
Small model (~50k params) for fast CPU inference.
"""

from __future__ import annotations

from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from poker44.models.hand_encoder import (
    HAND_VECTOR_DIM, MAX_ACTIONS, FEATURES_PER_ACTION, HAND_LEVEL_FEATURES,
    encode_hand,
)


class HandCNN(nn.Module):
    """Per-hand 1D-CNN on action sequence + hand-level features."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        # Conv on [12 actions × 8 features] → treat as 1D sequence, 8 channels
        self.conv1 = nn.Conv1d(FEATURES_PER_ACTION, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Combine conv output + hand-level features
        self.fc1 = nn.Linear(hidden_dim + HAND_LEVEL_FEATURES, 32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.1)

    def forward_hand(self, hand_vec: torch.Tensor) -> torch.Tensor:
        """Score one hand. Input: [101] flat vector. Output: [1] score."""
        # Split into action sequence and hand features
        action_flat = hand_vec[:MAX_ACTIONS * FEATURES_PER_ACTION]
        hand_feats = hand_vec[MAX_ACTIONS * FEATURES_PER_ACTION:]

        # Reshape actions: [12, 8] → Conv1D expects [channels, length] = [8, 12]
        actions = action_flat.view(MAX_ACTIONS, FEATURES_PER_ACTION).T  # [8, 12]

        # Conv layers
        x = F.relu(self.conv1(actions.unsqueeze(0)))   # [1, 32, 12]
        x = F.relu(self.conv2(x))                       # [1, 64, 12]
        x = self.pool(x).squeeze(-1).squeeze(0)         # [64]

        # Combine with hand features
        combined = torch.cat([x, hand_feats])            # [64 + 5]
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        return self.fc2(x)                               # [1]

    def forward(self, hand_vecs: torch.Tensor) -> torch.Tensor:
        """Score batch of hands. Input: [N, 101]. Output: [N, 1]."""
        scores = []
        for i in range(hand_vecs.shape[0]):
            scores.append(self.forward_hand(hand_vecs[i]))
        return torch.stack(scores)


class ChunkDetector(nn.Module):
    """Chunk-level bot detector: per-hand CNN → aggregate → chunk score."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.hand_cnn = HandCNN(hidden_dim=hidden_dim)
        # Aggregate hand scores into chunk score
        self.chunk_fc = nn.Linear(3, 1)  # mean, std, max of hand scores

    def forward_chunk(self, hand_vecs: torch.Tensor) -> torch.Tensor:
        """Score one chunk. Input: [N, 101]. Output: scalar score."""
        hand_scores = torch.sigmoid(self.hand_cnn(hand_vecs)).squeeze(-1)  # [N]
        # Simple mean — proven to give natural gap (human ~0.0, bot ~1.0)
        return hand_scores.mean().unsqueeze(0)

    def score_chunk(self, hands: List[Dict[str, Any]]) -> float:
        """Inference on raw sanitized hands. Returns float in [0, 1]."""
        if not hands:
            return 0.5
        vecs = torch.tensor(
            np.stack([encode_hand(h) for h in hands]),
            dtype=torch.float32,
        )
        with torch.no_grad():
            score = self.forward_chunk(vecs)
        return float(score.item())

    def score_chunks(self, chunks: List[List[Dict[str, Any]]]) -> List[float]:
        """Score multiple chunks."""
        return [self.score_chunk(hands) for hands in chunks]
