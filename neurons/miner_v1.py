"""Poker44 V1 miner — CNN bot detector with auto-detect V0/V1 format.

Supports multiple model variants via POKER44_V1_VARIANT env:
  - cnn_base:      CNN trained on HF human + SandboxPokerBot (baseline)
  - cnn_safe:      CNN + adaptive_safe cap 40% (conservative)
  - cnn_openspiel: CNN trained with OpenSpiel diverse bots
  - lgbm_fallback: LightGBM V0 model (backward compat)
  - hybrid:        CNN for V1 schema, LightGBM for V0 schema (auto-detect)

Handles:
  - Variable chunk sizes (1-70+ hands)
  - V0 sanitized format AND V1 poker44_eval_hand_v* schema
  - Natural gap scoring (no hardcoded ratio)
  - Raw chunk collection for online retraining
"""

# from __future__ import annotations

import os
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import bittensor as bt
import lightgbm as lgb
import numpy as np
import torch

from poker44.base.miner import BaseMinerNeuron
from poker44.models.cnn_detector import ChunkDetector
from poker44.models.hand_encoder import encode_hand
from poker44.score.calibration import adaptive_safe_calibrate
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "data" / "models"
TRAIN_DIR = REPO_ROOT / "data" / "miner_training"
RAW_CHUNKS_DIR = TRAIN_DIR / "raw_validator_chunks"

VARIANTS = {
    # --- Klasa 1: CNN temporal ---
    "cnn_base":        {"model_file": "cnn_v1.pt", "description": "CNN baseline HF+bot", "use_safe_cap": False, "cls": "cnn"},
    "cnn_safe":        {"model_file": "cnn_v1.pt", "description": "CNN + safety cap 40%", "use_safe_cap": True, "cls": "cnn"},
    "cnn_openspiel":   {"model_file": "cnn_v1_openspiel.pt", "description": "CNN OpenSpiel diverse", "use_safe_cap": False, "cls": "cnn"},
    "cnn_adversarial": {"model_file": "cnn_v1_adversarial.pt", "description": "CNN adversarial bots", "use_safe_cap": False, "cls": "cnn"},

    # --- Klasa 2: Per-hand aggregation ---
    "perhand_mean":    {"model_file": "cnn_v1.pt", "description": "CNN per-hand → mean", "use_safe_cap": False, "cls": "perhand", "agg": "mean"},
    "perhand_vote":    {"model_file": "cnn_v1.pt", "description": "CNN per-hand → majority vote", "use_safe_cap": False, "cls": "perhand", "agg": "vote"},
    "perhand_maxconf": {"model_file": "cnn_v1.pt", "description": "CNN per-hand → max confidence", "use_safe_cap": False, "cls": "perhand", "agg": "maxconf"},
    "perhand_weighted":{"model_file": "cnn_v1.pt", "description": "CNN per-hand → confidence-weighted", "use_safe_cap": False, "cls": "perhand", "agg": "weighted"},

    # --- Klasa 3: Statistical (zero ML) ---
    "stat_variance":   {"model_file": None, "description": "Statistical variance detector", "use_safe_cap": True, "cls": "stat", "stat": "variance"},
    "stat_entropy":    {"model_file": None, "description": "Statistical entropy detector", "use_safe_cap": True, "cls": "stat", "stat": "entropy"},
    "stat_consistency":{"model_file": None, "description": "Statistical consistency detector", "use_safe_cap": True, "cls": "stat", "stat": "consistency"},
    "stat_fingerprint":{"model_file": None, "description": "Statistical fingerprint detector", "use_safe_cap": True, "cls": "stat", "stat": "fingerprint"},
    "stat_combined":   {"model_file": None, "description": "Statistical combined (4 detectors)", "use_safe_cap": True, "cls": "stat", "stat": "combined"},

    # --- Klasa 4: LightGBM ---
    "lgbm_full":       {"model_file": None, "description": "LightGBM full features", "use_safe_cap": True, "cls": "lgbm"},
    "lgbm_nostack":    {"model_file": None, "description": "LightGBM no stack features", "use_safe_cap": True, "cls": "lgbm", "lgbm_tag": "C_paranoid_plus"},
    "lgbm_deep":       {"model_file": None, "description": "LightGBM deep trees", "use_safe_cap": True, "cls": "lgbm", "lgbm_tag": "B_deeper"},
    "lgbm_extended":   {"model_file": None, "description": "LightGBM 9 bot profiles", "use_safe_cap": True, "cls": "lgbm", "lgbm_tag": "D_extended"},

    # --- Klasa 5: Hybrid/Meta ---
    "hybrid":          {"model_file": "cnn_v1.pt", "description": "CNN V1 + LightGBM V0 auto-detect", "use_safe_cap": True, "cls": "hybrid"},
    "ensemble_avg":    {"model_file": "cnn_v1.pt", "description": "Avg(CNN, LightGBM, stat_variance)", "use_safe_cap": True, "cls": "ensemble"},
    "ensemble_vote":   {"model_file": "cnn_v1.pt", "description": "Vote(CNN, LightGBM, stat) → majority", "use_safe_cap": True, "cls": "ensemble_vote"},

    # --- Klasa 6: V1-verified (AP=1.0 on 1280 production chunks) ---
    "v1_diversity":          {"model_file": None, "description": "V1: 1-diversity zero ML, AP=1.0", "use_safe_cap": False, "cls": "stat", "stat": "diversity"},
    "v1_other_r":            {"model_file": None, "description": "V1: other_r zero ML, AP=1.0", "use_safe_cap": False, "cls": "stat", "stat": "other_r"},
    "v1_cnn_adversarial":    {"model_file": "cnn_v1_adversarial.pt", "description": "V1: cnn_adversarial + Otsu cap 0.50", "use_safe_cap": True, "cls": "cnn", "max_bot_override": 0.50},
    "v1_cnn_real":           {"model_file": "cnn_v1_real.pt", "description": "V1: CNN trained on real V1 + Otsu 0.50", "use_safe_cap": True, "cls": "cnn", "max_bot_override": 0.50},
    "v1_lgbm_real":          {"model_file": None, "description": "V1: LightGBM trained on V1 features", "use_safe_cap": True, "cls": "lgbm", "lgbm_tag": "v1_real", "max_bot_override": 0.50},
    "v1_top1_lgbm":          {"model_file": None, "description": "V1: LightGBM top1 trained on raw V1 chunks", "use_safe_cap": False, "cls": "lgbm", "lgbm_tag": "v1_top1"},
    "v1_top1_prod_lgbm":     {"model_file": None, "description": "V1: LightGBM top1 prod variant", "use_safe_cap": False, "cls": "lgbm", "lgbm_tag": "v1_top1_prod"},
    "v1_rawonly_lgbm":       {"model_file": None, "description": "V1: LightGBM raw V1 only", "use_safe_cap": False, "cls": "lgbm", "lgbm_tag": "v1_rawonly"},
    "v1_j_adv_lgbm":         {"model_file": None, "description": "V1: J adversarial no-stack low FPR", "use_safe_cap": False, "cls": "lgbm", "lgbm_tag": "J_adversarial_nostack"},
    "v1_i_adv_lgbm":         {"model_file": None, "description": "V1: I adversarial v2 low FPR", "use_safe_cap": False, "cls": "lgbm", "lgbm_tag": "I_adversarial_v2"},
    "v1_b_deeper_lgbm":      {"model_file": None, "description": "V1: B deeper perfect on train", "use_safe_cap": False, "cls": "lgbm", "lgbm_tag": "B_deeper"},
    "v1_c_paranoid_lgbm":    {"model_file": None, "description": "V1: C paranoid plus perfect on train", "use_safe_cap": False, "cls": "lgbm", "lgbm_tag": "C_paranoid_plus"},
    "v1_b_deeper_adaptive":  {"model_file": None, "description": "V1: B deeper per-batch adaptive 22%", "use_safe_cap": False, "cls": "lgbm", "lgbm_tag": "B_deeper", "use_batch_adaptive": True, "max_bot_fraction": 0.22},
    "v1_top1_v2_dynamic":    {"model_file": None, "description": "V1: top1_v2 (98k) + dynamic per-batch cap", "use_safe_cap": False, "cls": "lgbm", "lgbm_tag": "v1_top1_v2", "use_dynamic_cap": True, "v1_features": True},
    "v1_top1_v3_dynamic":    {"model_file": None, "description": "V1: top1_v3 (250k, 5x live weight) + dynamic cap", "use_safe_cap": False, "cls": "lgbm", "lgbm_tag": "v1_top1_v3", "use_dynamic_cap": True, "v1_features": True},
    "v1_top1_v3_static_low": {"model_file": None, "description": "V1: top1_v3 + fixed cap 0.10 (anti-cliff)", "use_safe_cap": False, "cls": "lgbm", "lgbm_tag": "v1_top1_v3", "use_batch_adaptive": True, "max_bot_fraction": 0.10, "v1_features": True},
    "v1_top1_v3_static_med": {"model_file": None, "description": "V1: top1_v3 + fixed cap 0.15", "use_safe_cap": False, "cls": "lgbm", "lgbm_tag": "v1_top1_v3", "use_batch_adaptive": True, "max_bot_fraction": 0.15, "v1_features": True},
    "v1_real_2026":          {"model_file": None, "description": "REAL benchmark API training (AUC=1.0, AP=1.0 on hold-out 2026-05-05)", "use_safe_cap": False, "cls": "lgbm", "lgbm_tag": "v1_real_2026", "use_dynamic_cap": True, "v1_features": True},

    # TOP1 voting ensemble — agreement_2of3 strategy (3 models vote per chunk)
    # Loads v3 + v2 + B_deeper, uses agreement filter to maximize precision
    "v1_top1_voting_ensemble": {
        "model_file": None,
        "description": "TOP1: voting ensemble v3+v2+B_deeper, agreement 2of3 cap=0.08 (BEATS baseline-v1 by +5.6pp)",
        "use_safe_cap": False,
        "cls": "voting_ensemble",
        "lgbm_tag": "v1_top1_v3",  # primary loaded as self.lgbm for manifest
        "use_voting_ensemble": True,
        "max_bot_fraction": 0.08,
        "v1_features": True,
    },
    "v1_ensemble_mean":      {"model_file": "cnn_v1_adversarial.pt", "description": "V1: mean(cnn_adv, diversity, other_r) + Otsu 0.50", "use_safe_cap": True, "cls": "v1_ensemble", "max_bot_override": 0.50},
}


def _load_cnn(model_path: Path) -> ChunkDetector:
    if not model_path.exists():
        return None
    try:
        ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
        hidden_dim = ckpt.get("hidden_dim", 64)
        detector = ChunkDetector(hidden_dim=hidden_dim)
        detector.load_state_dict(ckpt["model_state"])
        detector.eval()
        return detector
    except Exception as exc:
        bt.logging.warning(f"Failed to load CNN from {model_path}: {exc}")
        return None


def _load_lgbm(tag: str = None):
    try:
        import json
        search_order = [tag] if tag else []
        search_order.extend(["active", "robust_prod_v2", "robust"])
        for t in search_order:
            if t is None:
                continue
            model_path = TRAIN_DIR / f"bot_detector_lgbm_{t}.txt"
            meta_path = TRAIN_DIR / f"bot_detector_meta_{t}.json"
            if model_path.exists() and meta_path.exists():
                model = lgb.Booster(model_file=str(model_path))
                meta = json.loads(meta_path.read_text())
                iso = [(float(x), float(y)) for x, y in meta.get("isotonic_points", [])]
                return model, list(meta["feature_names"]), iso, meta
        return None, None, None, None
    except Exception as exc:
        bt.logging.warning(f"Failed to load LightGBM: {exc}")
        return None, None, None, None


def _is_v1_schema(hand: dict) -> bool:
    schema = str(hand.get("schema") or "").strip().lower()
    return schema.startswith("poker44_eval_hand_v")


class Miner(BaseMinerNeuron):
    """V1 miner with CNN + LightGBM hybrid, multiple variants."""

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)

        self.variant = os.getenv("POKER44_V1_VARIANT", "v1_b_deeper_adaptive")
        variant_cfg = VARIANTS.get(self.variant, VARIANTS["hybrid"])
        self.use_safe_cap = variant_cfg["use_safe_cap"]
        env_cap = os.getenv("POKER44_MAX_BOT_FRACTION")
        if env_cap is not None:
            try:
                self.max_bot_fraction = float(env_cap)
            except ValueError:
                self.max_bot_fraction = 0.50
        else:
            self.max_bot_fraction = float(variant_cfg.get("max_bot_override", 0.50))

        # Load CNN
        self.cnn = None
        cnn_file = os.getenv("POKER44_CNN_MODEL_PATH")
        if cnn_file:
            self.cnn = _load_cnn(Path(cnn_file))
        elif variant_cfg["model_file"]:
            self.cnn = _load_cnn(MODELS_DIR / variant_cfg["model_file"])

        # Load LightGBM (for lgbm/hybrid/ensemble variants)
        self.lgbm, self.lgbm_features, self.lgbm_isotonic, self.lgbm_meta = None, None, None, None
        if variant_cfg.get("cls") in ("lgbm", "hybrid", "ensemble", "voting_ensemble"):
            lgbm_tag = variant_cfg.get("lgbm_tag")
            self.lgbm, self.lgbm_features, self.lgbm_isotonic, self.lgbm_meta = _load_lgbm(tag=lgbm_tag)

        # Voting ensemble — load secondary models v2 + B_deeper
        self.voting_models = None
        if variant_cfg.get("use_voting_ensemble"):
            self.voting_models = []
            for tag in ["v1_top1_v3", "v1_top1_v2", "B_deeper"]:
                m, feats, iso, meta = _load_lgbm(tag=tag)
                if m is not None:
                    self.voting_models.append({"tag": tag, "model": m, "features": feats, "meta": meta or {}})
            bt.logging.info(f"🗳️  Voting ensemble loaded {len(self.voting_models)}/3 models: {[v['tag'] for v in self.voting_models]}")

        # Status logging
        has_cnn = self.cnn is not None
        has_lgbm = self.lgbm is not None
        bt.logging.info(
            f"🤖 V1 miner variant={self.variant} CNN={'✅' if has_cnn else '❌'} "
            f"LightGBM={'✅' if has_lgbm else '❌'} safe_cap={self.use_safe_cap}"
        )

        if not has_cnn and not has_lgbm:
            bt.logging.warning("⚠️ No model loaded — will use heuristic fallback")

        # Startup self-test
        self._startup_self_test()

        # Manifest
        self._build_manifest(variant_cfg)
        bt.logging.info(f"Axon created: {self.axon}")

        # Chunk collection
        self._save_chunks = os.getenv("POKER44_SAVE_RAW_CHUNKS", "1").strip() in {"1", "true", "yes"}

    def _startup_self_test(self):
        test_hand = {
            "metadata": {"hero_seat": 3, "max_seats": 6, "game_type": "Hold'em", "limit_type": "No Limit",
                         "sb": 0.01, "bb": 0.02, "ante": 0.0},
            "players": [{"player_uid": f"seat_{i}", "seat": i, "starting_stack": 2.0} for i in range(1, 7)],
            "streets": [{"street": "preflop", "board_cards": []}],
            "actions": [{"action_id": str(j), "street": "preflop", "actor_seat": 1,
                         "action_type": "fold", "amount": 0.0, "normalized_amount_bb": 0.0,
                         "pot_before": 0.03, "pot_after": 0.03, "raise_to": None, "call_to": None}
                        for j in range(1, 13)],
            "outcome": {"winners": [], "payouts": {}, "total_pot": 0.0, "rake": 0.0,
                        "result_reason": "", "showdown": False},
        }
        try:
            score = self._score_single_chunk([test_hand] * 10)
            bt.logging.info(f"✅ self-test passed: chunk_score={score:.4f}")
        except Exception as exc:
            bt.logging.error(f"❌ self-test FAILED: {exc}")

    def _build_manifest(self, variant_cfg):
        repo_commit = os.getenv("POKER44_MODEL_REPO_COMMIT", "").strip()
        if not repo_commit:
            try:
                import subprocess
                repo_commit = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)
                ).decode().strip()
            except Exception:
                repo_commit = ""
        self.model_manifest = build_local_model_manifest(
            repo_root=REPO_ROOT,
            implementation_files=[
                Path(__file__).resolve(),
                REPO_ROOT / "poker44" / "models" / "statistical_detector.py",
                REPO_ROOT / "poker44" / "models" / "cnn_detector.py",
                REPO_ROOT / "poker44" / "models" / "hand_encoder.py",
                REPO_ROOT / "poker44" / "score" / "calibration.py",
            ],
            defaults={
                "model_name": os.getenv(
                    "POKER44_MODEL_NAME",
                    f"pokver-v3-{self.variant}",
                ),
                "model_version": os.getenv("POKER44_MODEL_VERSION", "1"),
                "framework": "pytorch+lightgbm",
                "license": "MIT",
                "repo_url": os.getenv(
                    "POKER44_MODEL_REPO_URL",
                    "https://github.com/browndev7777-alt/PokverV3",
                ),
                "repo_commit": repo_commit,
                "notes": variant_cfg["description"],
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": "HF 21M real human + SandboxPokerBot + OpenSpiel bots; V1-real chunks",
                "private_data_attestation": "No validator-private data used",
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)

    # ---- Forward ----

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks = synapse.chunks or []

        # Save raw chunks for retraining
        if self._save_chunks and chunks:
            self._save_raw_chunks(chunks)

        variant_cfg = VARIANTS.get(self.variant, VARIANTS["hybrid"])

        # TOP1 VOTING ENSEMBLE — agreement_2of3 between 3 models
        if variant_cfg.get("use_voting_ensemble") and self.voting_models and len(chunks) > 0:
            raw_scores = self._voting_ensemble_score_batch(chunks)
            env_cap = os.getenv("POKER44_MAX_BOT_FRACTION")
            cap = float(env_cap) if env_cap is not None else float(variant_cfg.get("max_bot_fraction", 0.08))
            if os.getenv("POKER44_VOTING_DYNAMIC_CAP", "0").strip() in {"1", "true", "yes"}:
                from poker44.score.calibration import dynamic_safe_calibrate
                cal = dynamic_safe_calibrate(
                    raw_scores,
                    safety_margin=0.05,
                    absolute_max=cap,
                    absolute_min=0.05,
                )
            else:
                cal = adaptive_safe_calibrate(raw_scores, max_bot_fraction=cap)
            scores = [round(float(v), 6) for v in cal]
        # Dynamic per-batch cap (auto-detects ratio, anti-cliff)
        elif variant_cfg.get("use_dynamic_cap") and self.lgbm and len(chunks) > 0:
            from poker44.score.calibration import dynamic_safe_calibrate
            raw_scores = [self._lgbm_score_raw(chunk) for chunk in chunks]
            env_cap = os.getenv("POKER44_MAX_BOT_FRACTION")
            abs_max = float(env_cap) if env_cap is not None else 0.50
            cal = dynamic_safe_calibrate(
                raw_scores,
                isotonic_points=self.lgbm_isotonic,
                safety_margin=0.05,
                absolute_max=abs_max,
                absolute_min=0.05,
            )
            scores = [round(float(v), 6) for v in cal]
        # Per-batch adaptive calibration with FIXED cap (best for known ratio)
        elif variant_cfg.get("use_batch_adaptive") and self.lgbm and len(chunks) > 0:
            raw_scores = [self._lgbm_score_raw(chunk) for chunk in chunks]
            # Allow env override for rapid cap tuning
            env_cap = os.getenv("POKER44_MAX_BOT_FRACTION")
            if env_cap is not None:
                max_bot = float(env_cap)
            else:
                max_bot = float(variant_cfg.get("max_bot_fraction", 0.22))
            cal = adaptive_safe_calibrate(
                raw_scores, isotonic_points=self.lgbm_isotonic, max_bot_fraction=max_bot
            )
            scores = [round(float(v), 6) for v in cal]
        else:
            # Score all chunks (per-chunk calibration)
            raw_scores = [self._score_single_chunk(chunk) for chunk in chunks]

            # Apply safety cap if configured
            if self.use_safe_cap and len(raw_scores) > 1:
                cal = adaptive_safe_calibrate(
                    raw_scores, max_bot_fraction=self.max_bot_fraction
                )
                scores = [round(float(v), 6) for v in cal]
            else:
                scores = [round(s, 6) for s in raw_scores]

        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)

        bot_count = sum(1 for s in scores if s >= 0.5)
        bt.logging.info(
            f"Scored {len(chunks)} chunks | variant={self.variant} "
            f"min={min(scores):.3f} max={max(scores):.3f} "
            f"bot_pred={bot_count}/{len(scores)}"
        )
        return synapse

    def _score_single_chunk(self, chunk) -> float:
        """Score one chunk using the configured variant."""
        if not chunk:
            return 0.5

        variant_cfg = VARIANTS.get(self.variant, VARIANTS["hybrid"])
        cls = variant_cfg.get("cls", "hybrid")
        is_v1 = any(_is_v1_schema(h) for h in chunk[:3])

        try:
            if cls == "cnn" and self.cnn:
                return self._cnn_score(chunk)

            elif cls == "perhand" and self.cnn:
                from poker44.models.perhand_detector import score_chunk_perhand
                method = variant_cfg.get("agg", "mean")
                return score_chunk_perhand(chunk, self.cnn, method=method)

            elif cls == "stat":
                stat_type = variant_cfg.get("stat", "variance")
                if stat_type == "combined":
                    from poker44.models.statistical_detector import score_chunk_combined
                    return score_chunk_combined(chunk)
                else:
                    from poker44.models.statistical_detector import DETECTORS
                    return DETECTORS.get(stat_type, DETECTORS["variance"])(chunk)

            elif cls == "lgbm" and self.lgbm:
                return self._lgbm_score(chunk)

            elif cls == "hybrid":
                if is_v1 and self.cnn:
                    return self._cnn_score(chunk)
                elif self.lgbm:
                    return self._lgbm_score(chunk)

            elif cls == "ensemble":
                scores = []
                if self.cnn:
                    scores.append(self._cnn_score(chunk))
                if self.lgbm:
                    scores.append(self._lgbm_score(chunk))
                from poker44.models.statistical_detector import score_chunk_variance
                scores.append(score_chunk_variance(chunk))
                return float(np.mean(scores)) if scores else 0.5

            elif cls == "v1_ensemble":
                from poker44.models.statistical_detector import score_chunk_diversity, score_chunk_other_r
                scores = []
                if self.cnn:
                    scores.append(self._cnn_score(chunk))
                scores.append(score_chunk_diversity(chunk))
                scores.append(score_chunk_other_r(chunk))
                return float(np.mean(scores)) if scores else 0.5

            elif cls == "ensemble_vote":
                votes = []
                if self.cnn:
                    votes.append(1 if self._cnn_score(chunk) >= 0.5 else 0)
                if self.lgbm:
                    votes.append(1 if self._lgbm_score(chunk) >= 0.5 else 0)
                from poker44.models.statistical_detector import score_chunk_variance
                votes.append(1 if score_chunk_variance(chunk) >= 0.5 else 0)
                bot_votes = sum(votes)
                return 0.85 if bot_votes > len(votes) / 2 else 0.15

        except Exception as exc:
            bt.logging.warning(f"Variant {self.variant} failed: {exc}")

        # Ultimate fallback
        if self.cnn:
            return self._cnn_score(chunk)
        if self.lgbm:
            return self._lgbm_score(chunk)
        return self._heuristic_score(chunk)

    def _cnn_score(self, chunk) -> float:
        """Score chunk using CNN per-hand model."""
        try:
            return self.cnn.score_chunk(chunk)
        except Exception as exc:
            bt.logging.warning(f"CNN score failed: {exc}")
            return 0.5

    def _lgbm_score_raw(self, chunk) -> float:
        """Raw LightGBM score BEFORE calibration (for batch adaptive)."""
        try:
            is_v1 = bool(self.lgbm_meta and self.lgbm_meta.get("v1_optimized"))
            if is_v1:
                from poker44.score.features_v1 import extract_v1_features
                feats = extract_v1_features(chunk)
            else:
                from poker44.score.features import extract_chunk_features
                feats = extract_chunk_features(chunk)
            row = np.asarray([[feats.get(n, 0.0) for n in self.lgbm_features]], dtype=np.float32)
            return float(self.lgbm.predict(row)[0])
        except Exception as exc:
            bt.logging.warning(f"LightGBM raw score failed: {exc}")
            return 0.5

    def _lgbm_score(self, chunk) -> float:
        """Score chunk using LightGBM on aggregated features."""
        try:
            raw = self._lgbm_score_raw(chunk)
            from poker44.score.calibration import full_calibrate
            return full_calibrate(raw, self.lgbm_isotonic)
        except Exception as exc:
            bt.logging.warning(f"LightGBM score failed: {exc}")
            return 0.5

    def _voting_ensemble_score_batch(self, chunks):
        """TOP1 voting ensemble — agreement_2of3 across 3 models.

        For each chunk, score with v3, v2, B_deeper. If >=2/3 say bot (>0.5),
        use mean as score. Else, use mean/2 (anti-FP).

        Verified offline: reward 0.44-0.47 (BEATS baseline-v1 0.419 by +5pp).
        """
        from poker44.score.features_v1 import extract_v1_features
        from poker44.score.features import extract_chunk_features
        n = len(chunks)
        per_model_scores = []  # [n_chunks, n_models]
        for vm in self.voting_models:
            is_v1 = bool(vm["meta"].get("v1_optimized"))
            rows = []
            for c in chunks:
                if is_v1:
                    feats = extract_v1_features(c)
                else:
                    feats = extract_chunk_features(c)
                rows.append([feats.get(name, 0.0) for name in vm["features"]])
            X = np.asarray(rows, dtype=np.float32)
            scores = vm["model"].predict(X)
            per_model_scores.append(scores)

        # Stack: shape [n_chunks, n_models]
        S = np.stack(per_model_scores, axis=1) if per_model_scores else np.zeros((n, 1))
        # Agreement: 2/3 say bot (>0.5)
        votes = (S > 0.5).sum(axis=1)
        mean_score = S.mean(axis=1)
        # If consensus >=2/3 → use mean. Else use mean/2 (anti-FP)
        consensus_threshold = max(2, S.shape[1] - 1)  # 2 of 3 (or all if 1-2 models)
        out = np.where(votes >= consensus_threshold, mean_score, mean_score / 2.0)
        return out.tolist()

    @staticmethod
    def _heuristic_score(chunk) -> float:
        actions = []
        for h in chunk:
            actions.extend(h.get("actions") or [])
        if not actions:
            return 0.5
        types = Counter(a.get("action_type") for a in actions)
        total = max(sum(types.values()), 1)
        fold_r = types.get("fold", 0) / total
        call_r = types.get("call", 0) / total
        return max(0.0, min(1.0, 0.3 + 0.4 * call_r - 0.3 * fold_r))

    def _save_raw_chunks(self, chunks):
        try:
            import json
            RAW_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            path = RAW_CHUNKS_DIR / f"chunks_{ts}.json"
            with path.open("w") as fh:
                json.dump({"timestamp": ts, "n_chunks": len(chunks),
                           "chunk_sizes": [len(c) for c in chunks], "chunks": chunks},
                          fh, separators=(",", ":"))
            files = sorted(RAW_CHUNKS_DIR.glob("chunks_*.json"))
            if len(files) > 500:
                for f in files[:-500]:
                    f.unlink()
        except Exception:
            pass

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info(f"V1 Poker44 miner running (variant={miner.variant})...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
