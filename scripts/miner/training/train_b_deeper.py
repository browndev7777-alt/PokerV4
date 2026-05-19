#!/usr/bin/env python3
"""Train LightGBM B_deeper artifacts for PokverV3 miner_v1 (v1_b_deeper_adaptive)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parents[3]
TRAIN_DIR = REPO / "data" / "miner_training"
TAG = "B_deeper"


def _fit_isotonic(raw: np.ndarray, y: np.ndarray) -> list[list[float]]:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw, y)
    xs = np.linspace(float(raw.min()), float(raw.max()), 20)
    ys = iso.predict(xs)
    return [[float(x), float(y)] for x, y in zip(xs, ys)]


def main() -> None:
    sys.path.insert(0, str(REPO))
    from poker44.score.features import features_to_matrix
    from poker44.training.synthetic import generate_labeled_chunks

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=5000, help="Balanced synthetic chunks")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default=TAG)
    args = p.parse_args()

    labeled = generate_labeled_chunks(args.samples, seed=args.seed)
    chunks = [{"hands": h, "is_bot": bool(y)} for h, y in labeled]
    X, y, feature_names = features_to_matrix(chunks)
    if X.shape[0] < 100:
        raise SystemExit("Not enough training rows")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
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
        "min_child_samples": 20,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "seed": args.seed,
    }

    model = lgb.train(
        params,
        train_set,
        num_boost_round=400,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )

    raw_val = model.predict(X_val)
    auc = roc_auc_score(y_val, raw_val)
    ap = average_precision_score(y_val, raw_val)
    iso_points = _fit_isotonic(raw_val, y_val)

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    model_path = TRAIN_DIR / f"bot_detector_lgbm_{args.tag}.txt"
    meta_path = TRAIN_DIR / f"bot_detector_meta_{args.tag}.json"
    model.save_model(str(model_path))

    meta = {
        "tag": args.tag,
        "feature_names": feature_names,
        "isotonic_points": iso_points,
        "v1_optimized": False,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "val_auc": float(auc),
        "val_ap": float(ap),
        "notes": "PokverV3 synthetic B_deeper; replace with production-labeled data for competitive scores",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {model_path}")
    print(f"Wrote {meta_path}")
    print(f"features={len(feature_names)} val_auc={auc:.4f} val_ap={ap:.4f}")


if __name__ == "__main__":
    main()
