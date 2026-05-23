#!/usr/bin/env python3
"""Train a Poker44 LightGBM model on the benchmark API data using Travis861's
rich chunk_features() extractor.

Loads benchmark JSON files from /root/PokerNew2/data/benchmark/ (downloaded
earlier by download_benchmark.py), extracts ~300 features per chunk via
poker44_ml.features.chunk_features, trains a LightGBM model with a
conformal score_shift tuned to keep validation FPR below 4%, and saves a
joblib artifact compatible with poker44_ml.inference.Poker44Model.

Usage (run from /root/PokerNew3 with miner_env active):
    python training/train_benchmark.py
    python training/train_benchmark.py --target-fpr 0.03
    python training/train_benchmark.py --output models/poker44_v1_custom.joblib
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = Path("/root/PokerNew2/data/benchmark")

sys.path.insert(0, str(REPO))

warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import lightgbm as lgb

from poker44_ml.features import chunk_features
from poker44_ml.stacked import StackedEnsemble


class LGBProbaAdapter:
    """Picklable adapter exposing LightGBM Booster as predict_proba."""

    def __init__(self, booster):
        self.booster = booster

    def predict_proba(self, X):
        p1 = self.booster.predict(X)
        return np.stack([1 - p1, p1], axis=1)


def git_head(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root, check=True, capture_output=True, text=True,
        ).stdout.strip()
    except Exception:
        return ""


def load_labeled() -> list[tuple[list[dict], int]]:
    files = sorted(BENCHMARK_DIR.glob("*.json"))
    if not files:
        sys.exit(f"No benchmark files in {BENCHMARK_DIR}. Run download_benchmark.py first.")

    rows: list[tuple[list[dict], int]] = []
    for f in files:
        outer = json.loads(f.read_text(encoding="utf-8"))
        chunks = outer.get("chunks") or []
        labels = outer.get("groundTruth") or []
        if len(chunks) != len(labels):
            continue
        for chunk, label in zip(chunks, labels):
            rows.append((chunk, int(label)))
    return rows


def extract_matrix(rows: list[tuple[list[dict], int]]):
    feats_list = []
    labels = []
    for chunk, label in rows:
        feats = chunk_features(chunk)
        feats["hand_count"] = float(len(chunk))
        feats_list.append(feats)
        labels.append(label)
    feature_names = sorted(set().union(*[d.keys() for d in feats_list]))
    X = np.zeros((len(feats_list), len(feature_names)), dtype=np.float64)
    for i, d in enumerate(feats_list):
        for j, name in enumerate(feature_names):
            X[i, j] = float(d.get(name, 0.0))
    y = np.asarray(labels, dtype=np.int64)
    return X, y, feature_names


def fit_score_shift(raw_val: np.ndarray, y_val: np.ndarray, target_fpr: float) -> float:
    """Find a logit shift that pushes FPR below target_fpr on the validation set.

    StackedEnsemble._logit_shift adds `shift` to logit(score) and re-sigmoids.
    Negative shift -> probabilities decrease -> fewer bot predictions -> lower FPR.
    """
    for shift in np.arange(0.0, -3.0, -0.1):
        clipped = np.clip(raw_val, 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped)) + float(shift)
        shifted = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        preds = (shifted >= 0.5).astype(int)
        cm = confusion_matrix(y_val, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / max(tn + fp, 1)
        if fpr <= target_fpr:
            return float(shift)
    return -3.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-fpr", type=float, default=0.04,
                   help="Maximum tolerated FPR on validation set (default 0.04)")
    p.add_argument("--output", type=str,
                   default=str(REPO / "models" / "poker44_v1_custom.joblib"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-boost-round", type=int, default=600)
    p.add_argument("--early-stopping", type=int, default=50)
    args = p.parse_args()

    print(f"Loading benchmark data from {BENCHMARK_DIR} ...")
    rows = load_labeled()
    n_bot = sum(1 for _, y in rows if y == 1)
    n_human = len(rows) - n_bot
    print(f"  {len(rows)} inner chunks  ({n_bot} bot, {n_human} human)")

    print("Extracting Travis861 chunk_features (this can take a minute) ...")
    X, y, feature_names = extract_matrix(rows)
    print(f"  Feature matrix: {X.shape[0]} rows x {X.shape[1]} features")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y,
    )
    print(f"  Split: train={len(y_train)}  val={len(y_val)}")

    print(f"Training LightGBM (num_boost_round={args.num_boost_round}, early_stopping={args.early_stopping}) ...")
    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "max_depth": -1,
        "learning_rate": 0.05,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "min_child_samples": 10,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "seed": args.seed,
    }
    booster = lgb.train(
        params, train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(args.early_stopping, verbose=False),
            lgb.log_evaluation(50),
        ],
    )

    raw_val = booster.predict(X_val)
    auc = roc_auc_score(y_val, raw_val)
    ap = average_precision_score(y_val, raw_val)
    preds = (raw_val >= 0.5).astype(int)
    cm = confusion_matrix(y_val, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / max(tn + fp, 1)
    recall = tp / max(tp + fn, 1)
    print(f"\nValidation (uncalibrated):  AUC={auc:.4f}  AP={ap:.4f}  FPR={fpr:.4f}  Recall={recall:.4f}")

    print(f"\nFitting conformal score_shift for target_fpr={args.target_fpr} ...")
    shift = fit_score_shift(raw_val, y_val, args.target_fpr)
    print(f"  score_shift = {shift:.3f}")

    # Apply shift, re-eval
    clipped = np.clip(raw_val, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)) + shift
    shifted = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
    preds = (shifted >= 0.5).astype(int)
    cm = confusion_matrix(y_val, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr_shifted = fp / max(tn + fp, 1)
    recall_shifted = tp / max(tp + fn, 1)
    ap_shifted = average_precision_score(y_val, shifted)
    print(f"  After shift:  AP={ap_shifted:.4f}  FPR={fpr_shifted:.4f}  Recall={recall_shifted:.4f}")

    from sklearn.linear_model import LogisticRegression
    # Trivial meta: identity logistic on the single base column
    meta_X = np.stack([1 - raw_val, raw_val], axis=1)[:, [1]]
    meta = LogisticRegression(C=1e6, solver="lbfgs").fit(meta_X, y_val)

    stacked = StackedEnsemble(
        base_models=[LGBProbaAdapter(booster)],
        meta_model=meta,
        calibrator=None,
        feature_indices=None,
        score_shift=shift,
    )

    artifact = {
        "models": [stacked],
        "feature_names": feature_names,
        "model_weights": [1.0],
        "metadata": {
            "model_name": "poker44-v1-custom-lgb",
            "model_version": "1",
            "framework": "lightgbm+score_shift",
            "benchmark_rows": int(len(rows)),
            "benchmark_positive_rows": int(n_bot),
            "benchmark_negative_rows": int(n_human),
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_val)),
            "val_auc": float(auc),
            "val_ap": float(ap),
            "val_ap_shifted": float(ap_shifted),
            "val_fpr_shifted": float(fpr_shifted),
            "val_recall_shifted": float(recall_shifted),
            "score_shift": float(shift),
            "target_fpr": float(args.target_fpr),
            "repo_url": "https://github.com/browndev7777-alt/PokerV4",
            "repo_commit": git_head(REPO),
            "feature_schema": "poker44_ml.features.chunk_features",
        },
        "calibrator": None,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out, compress=3)
    print(f"\nSaved artifact -> {out}  ({out.stat().st_size/1024/1024:.2f} MB)")
    print(f"To use: POKER44_MODEL_PATH={out.resolve()}  (or symlink to models/poker44_benchmark_supervised.joblib)")


if __name__ == "__main__":
    main()
