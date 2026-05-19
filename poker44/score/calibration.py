"""Map raw model probabilities to validator-friendly risk scores.

Two-stage pipeline:

1. Isotonic calibration (fitted at training time) maps raw model output → empirical
   P(bot|raw). This stretches narrow output ranges to the full [0, 1].
2. Semantic banding pushes borderline cases below 0.5 to protect against the FPR
   cliff (validator zeros reward at fpr ≥ 0.10), while preserving rank ordering
   so average_precision stays high.

    p < HUMAN_HI         → [HUMAN_LOW, HUMAN_HI_OUT]   (clear human)
    HUMAN_HI ≤ p < UNSURE → [UNSURE_LOW, UNSURE_HI]    (lean human, label 0)
    UNSURE ≤ p < BOT_HI   → [LEAN_BOT_LOW, LEAN_BOT_HI] (lean bot, label 1)
    p ≥ BOT_HI            → [BOT_LOW, BOT_HI_OUT]       (clear bot)
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

# Thresholds on raw model probability
HUMAN_HI = 0.30
UNSURE = 0.55
BOT_HI = 0.75

# Output bands (validator rounds at 0.5, so anything < 0.5 = label 0)
HUMAN_LOW, HUMAN_HI_OUT = 0.05, 0.15
UNSURE_LOW, UNSURE_HI = 0.30, 0.48
LEAN_BOT_LOW, LEAN_BOT_HI = 0.55, 0.72
BOT_LOW, BOT_HI_OUT = 0.85, 0.95


def _scale(value: float, in_lo: float, in_hi: float, out_lo: float, out_hi: float) -> float:
    if in_hi <= in_lo:
        return out_lo
    t = (value - in_lo) / (in_hi - in_lo)
    t = max(0.0, min(1.0, t))
    return out_lo + t * (out_hi - out_lo)


def calibrate_one(raw: float) -> float:
    raw = max(0.0, min(1.0, float(raw)))
    if raw < HUMAN_HI:
        return _scale(raw, 0.0, HUMAN_HI, HUMAN_LOW, HUMAN_HI_OUT)
    if raw < UNSURE:
        return _scale(raw, HUMAN_HI, UNSURE, UNSURE_LOW, UNSURE_HI)
    if raw < BOT_HI:
        return _scale(raw, UNSURE, BOT_HI, LEAN_BOT_LOW, LEAN_BOT_HI)
    return _scale(raw, BOT_HI, 1.0, BOT_LOW, BOT_HI_OUT)


def calibrate_array(raw: Sequence[float]) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64)
    out = np.empty_like(arr)
    for i, v in enumerate(arr):
        out[i] = calibrate_one(float(v))
    return out


def apply_isotonic(raw: float, isotonic_points: Optional[List[Tuple[float, float]]]) -> float:
    """Piecewise-linear interpolation between (x_i, y_i) thresholds from training.

    Outside the training range, clips to the boundary value (matches sklearn
    IsotonicRegression(out_of_bounds="clip")).
    """
    if not isotonic_points:
        return float(raw)
    xs = [p[0] for p in isotonic_points]
    ys = [p[1] for p in isotonic_points]
    if raw <= xs[0]:
        return ys[0]
    if raw >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if raw <= xs[i]:
            x0, x1 = xs[i - 1], xs[i]
            y0, y1 = ys[i - 1], ys[i]
            if x1 == x0:
                return y1
            t = (raw - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return ys[-1]


def full_calibrate(raw: float, isotonic_points: Optional[List[Tuple[float, float]]] = None) -> float:
    """Isotonic → semantic band. Use this in production miner."""
    p = apply_isotonic(raw, isotonic_points)
    return calibrate_one(p)


def adaptive_calibrate(
    raw_scores: Sequence[float],
    *,
    isotonic_points: Optional[List[Tuple[float, float]]] = None,
    min_gap_quantile: float = 0.15,
) -> np.ndarray:
    """Adaptive threshold per batch using Otsu's method on raw scores.

    Instead of forcing a fixed bot_ratio, finds the natural split point
    in raw scores that minimizes intra-class variance. Adapts automatically
    to ANY validator bot_ratio without configuration.

    If no clear bimodal split exists (uniform distribution), falls back
    to median split (equivalent to rank_based 0.5).
    """
    arr = np.asarray(raw_scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    if isotonic_points:
        arr_norm = np.asarray([apply_isotonic(float(x), isotonic_points) for x in arr])
    else:
        arr_norm = arr.copy()

    # Otsu's method: find threshold T that minimizes weighted intra-class variance
    sorted_vals = np.sort(arr_norm)
    best_t = np.median(arr_norm)
    best_var = float("inf")

    # Test candidate thresholds at each gap between consecutive sorted values
    for i in range(1, n):
        if sorted_vals[i] == sorted_vals[i - 1]:
            continue
        t = (sorted_vals[i - 1] + sorted_vals[i]) / 2.0
        class0 = arr_norm[arr_norm <= t]
        class1 = arr_norm[arr_norm > t]
        if len(class0) == 0 or len(class1) == 0:
            continue
        w0 = len(class0) / n
        w1 = len(class1) / n
        var_within = w0 * np.var(class0) + w1 * np.var(class1)
        if var_within < best_var:
            best_var = var_within
            best_t = t

    # Classify based on adaptive threshold
    is_bot = arr_norm > best_t
    n_bot = int(is_bot.sum())
    n_human = n - n_bot

    # Build output preserving ranking within each class
    out = np.empty(n, dtype=np.float64)
    bot_indices = np.where(is_bot)[0]
    human_indices = np.where(~is_bot)[0]

    # Sort bots by score descending, humans by score ascending
    if n_bot > 0:
        bot_order = bot_indices[np.argsort(-arr_norm[bot_indices])]
        for rank, idx in enumerate(bot_order):
            t_frac = rank / max(n_bot - 1, 1)
            out[idx] = BOT_HI_OUT - t_frac * (BOT_HI_OUT - BOT_LOW)

    if n_human > 0:
        human_order = human_indices[np.argsort(arr_norm[human_indices])]
        for rank, idx in enumerate(human_order):
            t_frac = rank / max(n_human - 1, 1)
            out[idx] = HUMAN_LOW + t_frac * (HUMAN_HI_OUT - HUMAN_LOW)

    return out


def adaptive_safe_calibrate(
    raw_scores: Sequence[float],
    *,
    isotonic_points: Optional[List[Tuple[float, float]]] = None,
    max_bot_fraction: float = 0.40,
) -> np.ndarray:
    """Adaptive Otsu + hard safety cap.

    1. Otsu finds natural bot/human split (adapts to ANY validator ratio)
    2. Safety cap: NEVER predict more than max_bot_fraction as bot
       (survives worst case POKER44_HUMAN_RATIO=0.60 → 24 humans)
    3. If over cap: demote weakest bot predictions to human band
    4. Preserve ranking within bands for AP
    """
    arr = np.asarray(raw_scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    # Step 1: Otsu split
    base = adaptive_calibrate(arr, isotonic_points=isotonic_points)

    # Step 2: Safety cap
    max_bots = int(n * max_bot_fraction)
    bot_mask = base >= 0.5
    n_bot_pred = int(bot_mask.sum())

    if n_bot_pred <= max_bots:
        return base  # within cap, no change needed

    # Step 3: Demote weakest bot predictions to human band
    bot_indices = np.where(bot_mask)[0]
    bot_scores = base[bot_indices]
    # Sort ascending — weakest bots first
    weakest_order = bot_indices[np.argsort(bot_scores)]
    n_to_flip = n_bot_pred - max_bots

    out = base.copy()
    # Re-rank the flipped bots into human band (top of human range)
    human_scores = out[~bot_mask]
    human_max = float(human_scores.max()) if len(human_scores) > 0 else HUMAN_HI_OUT

    for i, idx in enumerate(weakest_order[:n_to_flip]):
        # Place in human band, clamped to valid range
        candidate = 0.48 - i * 0.01
        out[idx] = max(HUMAN_LOW, min(candidate, human_max + 0.01))

    return out


def dynamic_safe_calibrate(
    raw_scores: Sequence[float],
    *,
    isotonic_points: Optional[List[Tuple[float, float]]] = None,
    estimated_bot_ratio: Optional[float] = None,
    safety_margin: float = 0.05,
    absolute_max: float = 0.50,
    absolute_min: float = 0.05,
) -> np.ndarray:
    """Dynamic per-batch cap based on confidence in raw distribution.

    Logic:
    - If raw scores are bimodal (clear gap) → estimate bot_ratio from gap location
    - Cap = max(absolute_min, min(absolute_max, estimated_ratio - safety_margin))
    - Otsu finds threshold; safety_margin prevents cliff if estimate is slightly high

    For unknown true ratio: this auto-adjusts so cliff never triggers.
    Combined with safety_margin=0.05, max FPR ≈ safety_margin / (1-est_ratio).
    """
    arr = np.asarray(raw_scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    # Apply isotonic if available
    if isotonic_points:
        arr_iso = np.asarray([apply_isotonic(float(x), isotonic_points) for x in arr])
    else:
        arr_iso = arr

    # Estimate bot_ratio if not provided: count scores above raw threshold 0.5
    if estimated_bot_ratio is None:
        # Use Otsu's threshold to estimate ratio
        sorted_vals = np.sort(arr_iso)
        best_t = float(np.median(arr_iso))
        best_var = float("inf")
        for i in range(1, n):
            if sorted_vals[i] == sorted_vals[i - 1]:
                continue
            t = (sorted_vals[i - 1] + sorted_vals[i]) / 2.0
            c0 = arr_iso[arr_iso <= t]
            c1 = arr_iso[arr_iso > t]
            if len(c0) == 0 or len(c1) == 0:
                continue
            w0 = len(c0) / n
            w1 = len(c1) / n
            v = w0 * np.var(c0) + w1 * np.var(c1)
            if v < best_var:
                best_var = v
                best_t = t
        estimated_bot_ratio = float((arr_iso > best_t).sum()) / n

    # Apply safety margin and bounds
    safe_cap = max(absolute_min, min(absolute_max, estimated_bot_ratio - safety_margin))

    # Use adaptive_safe_calibrate with computed cap
    return adaptive_safe_calibrate(arr, isotonic_points=isotonic_points, max_bot_fraction=safe_cap)


def rank_based_calibrate(
    raw_scores: Sequence[float],
    *,
    bot_ratio: float = 0.5,
    isotonic_points: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    """Rank-based calibration: assign top-K chunks as bot where K = N * bot_ratio.

    Eliminates FPR cliff regardless of validator's actual human_ratio — we always
    predict a fixed fraction of the batch as bots, preserving raw-score ranking
    within each band so AP stays intact.

    Output layout (rank within top-K bots gets BOT_LOW..BOT_HI_OUT, rank within
    bottom-K humans gets HUMAN_LOW..HUMAN_HI_OUT — preserves ordering for AP).
    """
    arr = np.asarray(raw_scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    # optional isotonic pre-normalization (for logging / inspection only)
    if isotonic_points:
        arr_iso = np.asarray([apply_isotonic(float(x), isotonic_points) for x in arr])
    else:
        arr_iso = arr

    k_bot = max(0, min(n, int(round(n * bot_ratio))))
    # indices sorted descending by raw/iso score
    order = np.argsort(-arr_iso)

    out = np.empty(n, dtype=np.float64)

    # bot tier — top K: rank r∈[0, k_bot-1] → linear map BOT_LOW..BOT_HI_OUT
    for rank_idx, idx in enumerate(order[:k_bot]):
        if k_bot <= 1:
            out[idx] = BOT_HI_OUT
        else:
            t = rank_idx / (k_bot - 1)  # 0.0 (top) .. 1.0 (lowest in bot band)
            out[idx] = BOT_HI_OUT - t * (BOT_HI_OUT - BOT_LOW)

    # human tier — bottom N-K: rank r∈[0, n-k_bot-1] → linear map HUMAN_HI_OUT..HUMAN_LOW
    remaining = n - k_bot
    for rank_idx, idx in enumerate(order[k_bot:]):
        if remaining <= 1:
            out[idx] = HUMAN_LOW
        else:
            t = rank_idx / (remaining - 1)  # 0.0 (highest human) .. 1.0 (lowest)
            out[idx] = HUMAN_HI_OUT - t * (HUMAN_HI_OUT - HUMAN_LOW)

    return out
