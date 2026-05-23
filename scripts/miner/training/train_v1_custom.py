#!/usr/bin/env python3
"""Train a V1-optimized LightGBM bot detector on Poker44 benchmark data.

Reads labeled chunks from data/benchmark/ (downloaded by download_benchmark.py),
extracts V1 features, trains LightGBM, fits isotonic calibration, and saves
model artifacts to data/miner_training/ as tag 'v1_custom'.

Usage (run from repo root with miner_env active):
    python scripts/miner/training/train_v1_custom.py
    python scripts/miner/training/train_v1_custom.py --samples-per-class 0  # use all data
    python scripts/miner/training/train_v1_custom.py --tag my_tag           # custom tag
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

from poker44.score.features_v1 import chunks_to_matrix

BENCHMARK_DIR = REPO / "data" / "benchmark"
TRAIN_DIR = REPO / "data" / "miner_training"
DEFAULT_TAG = "v1_custom"


def load_labeled_chunks(benchmark_dir: Path) -> list[dict]:
    """Load all outer-chunk JSON files and flatten to inner scoring units."""
    labeled = []
    files = sorted(benchmark_dir.glob("*.json"))
    if not files:
        sys.exit(
            f"No benchmark files found in {benchmark_dir}.\n"
            "Run download_benchmark.py first."
        )

    for f in files:
        outer = json.loads(f.read_text(encoding="utf-8"))
        inner_chunks = outer.get("chunks", [])
        ground_truth = outer.get("groundTruth", [])
        if len(inner_chunks) != len(ground_truth):
            print(f"  WARNING: skipping {f.name} — chunk/label count mismatch")
            continue
        for chunk, label in zip(inner_chunks, ground_truth):
            labeled.append({"hands": chunk, "is_bot": bool(label)})

    return labeled


def fit_isotonic(raw_val: np.ndarray, y_val: np.ndarray) -> list[list[float]]:
    """Fit isotonic regression and return piecewise-linear lookup table."""
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_val, y_val)
    xs = np.linspace(float(raw_val.min()), float(raw_val.max()), 30)
    ys = iso.predict(xs)
    return [[float(x), float(y)] for x, y in zip(xs, ys)]


def print_confusion(y_true: np.ndarray, y_pred_raw: np.ndarray, threshold: float = 0.5) -> None:
    preds = (y_pred_raw >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    n_neg = max(tn + fp, 1)
    n_pos = max(tp + fn, 1)
    fpr = fp / n_neg
    recall = tp / n_pos
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  FPR={fpr:.4f}  Recall={recall:.4f}")
    if fpr >= 0.10:
        print("  WARNING: FPR >= 10% — reward will be 0 on live data!")
    else:
        print(f"  FPR OK ({fpr:.2%} < 10%)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default=DEFAULT_TAG, help="Model tag (default: v1_custom)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--samples-per-class",
        type=int,
        default=0,
        help="Cap samples per class for balanced training (0 = use all data)",
    )
    p.add_argument("--num-boost-round", type=int, default=600)
    p.add_argument("--early-stopping", type=int, default=50)
    args = p.parse_args()

    print(f"Loading benchmark data from {BENCHMARK_DIR} ...")
    labeled = load_labeled_chunks(BENCHMARK_DIR)
    n_bot = sum(1 for d in labeled if d["is_bot"])
    n_human = sum(1 for d in labeled if not d["is_bot"])
    print(f"  Loaded {len(labeled)} inner chunks: {n_bot} bot, {n_human} human")

    if len(labeled) < 50:
        sys.exit("Too few samples to train. Download more benchmark data first.")

    # Optional per-class cap to balance classes
    if args.samples_per_class > 0:
        rng = np.random.default_rng(args.seed)
        bots = [d for d in labeled if d["is_bot"]]
        humans = [d for d in labeled if not d["is_bot"]]
        cap = args.samples_per_class
        bots = [bots[i] for i in rng.choice(len(bots), min(cap, len(bots)), replace=False)]
        humans = [humans[i] for i in rng.choice(len(humans), min(cap, len(humans)), replace=False)]
        labeled = bots + humans
        rng.shuffle(labeled)
        print(f"  After cap: {len(labeled)} chunks ({len(bots)} bot, {len(humans)} human)")

    print("Extracting V1 features ...")
    X, y, feature_names = chunks_to_matrix(labeled)
    print(f"  Feature matrix: {X.shape[0]} rows x {X.shape[1]} features")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    print(f"  Train={len(y_train)} Val={len(y_val)}")

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

    print(f"Training LightGBM (num_boost_round={args.num_boost_round}, early_stopping={args.early_stopping}) ...")
    callbacks = [
        lgb.early_stopping(args.early_stopping, verbose=False),
        lgb.log_evaluation(50),
    ]
    model = lgb.train(
        params,
        train_set,
        num_boost_round=args.num_boost_round,
        valid_sets=[val_set],
        callbacks=callbacks,
    )

    raw_val = model.predict(X_val)
    auc = roc_auc_score(y_val, raw_val)
    ap = average_precision_score(y_val, raw_val)

    print(f"\nValidation results:")
    print(f"  AUC={auc:.4f}  AP={ap:.4f}")
    print_confusion(y_val, raw_val)

    print("\nFitting isotonic calibration ...")
    iso_points = fit_isotonic(raw_val, y_val.astype(float))

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    model_path = TRAIN_DIR / f"bot_detector_lgbm_{args.tag}.txt"
    meta_path = TRAIN_DIR / f"bot_detector_meta_{args.tag}.json"

    model.save_model(str(model_path))

    meta = {
        "tag": args.tag,
        "feature_names": feature_names,
        "isotonic_points": iso_points,
        "v1_optimized": True,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "val_auc": float(auc),
        "val_ap": float(ap),
        "n_bot_train": int((y_train == 1).sum()),
        "n_human_train": int((y_train == 0).sum()),
        "notes": "V1-optimized model trained on Poker44 benchmark API data",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\nSaved:")
    print(f"  {model_path}")
    print(f"  {meta_path}")
    print(f"\nSummary: tag={args.tag} features={len(feature_names)} AUC={auc:.4f} AP={ap:.4f}")
    print(f"Use with: POKER44_V1_VARIANT=v1_custom")


if __name__ == "__main__":
    main()
