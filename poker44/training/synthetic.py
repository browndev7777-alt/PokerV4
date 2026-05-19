"""Synthetic canonical Poker44 hands for supervised training (development only)."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Literal, Tuple

import numpy as np

from poker44.core.hand_json import V0_JSON_HAND

Label = Literal["human", "bot"]


def _action(
    action_id: int,
    street: str,
    actor_seat: int,
    action_type: str,
    bb: float,
    pot_before: float,
    amount: float,
    raise_to: float | None = None,
    call_to: float | None = None,
) -> Dict[str, Any]:
    pot_after = pot_before + (0.0 if action_type in {"fold", "check"} else amount)
    return {
        "action_id": str(action_id),
        "street": street,
        "actor_seat": actor_seat,
        "action_type": action_type,
        "amount": round(amount, 4),
        "raise_to": round(raise_to, 4) if raise_to is not None else None,
        "call_to": round(call_to, 4) if call_to is not None else None,
        "normalized_amount_bb": round(amount / bb, 4) if bb else 0.0,
        "pot_before": round(pot_before, 4),
        "pot_after": round(pot_after, 4),
    }


def _players(n: int, bb: float, rng: np.random.Generator) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for seat in range(1, n + 1):
        stack = float(rng.uniform(40.0, 120.0) * bb)
        out.append(
            {
                "player_uid": f"p{seat}",
                "seat": seat,
                "starting_stack": round(stack, 2),
                "hole_cards": None,
                "showed_hand": False,
            }
        )
    return out


def _streets(depth: int) -> List[Dict[str, Any]]:
    boards = [
        [],
        ["7s", "Jd", "Ad"],
        ["7s", "Jd", "Ad", "2c"],
        ["7s", "Jd", "Ad", "2c", "Kh"],
    ]
    names = ["flop", "turn", "river"]
    if depth <= 0:
        return []
    if depth == 1:
        return [{"street": "flop", "board_cards": boards[1]}]
    if depth == 2:
        return [
            {"street": "flop", "board_cards": boards[1]},
            {"street": "turn", "board_cards": boards[2]},
        ]
    return [
        {"street": "flop", "board_cards": boards[1]},
        {"street": "turn", "board_cards": boards[2]},
        {"street": "river", "board_cards": boards[3]},
    ]


def _regime_biases(
    rng: np.random.Generator, label: Label
) -> tuple[float, float, float, float, float]:
    """Return (raise, bet, fold, check, call) probability masses before normalization."""
    regime = int(rng.integers(0, 6))
    if label == "human":
        # Nit, LP, TAG, whale, random, standard
        tbl = [
            (0.06, 0.06, 0.38, 0.28, 0.22),
            (0.08, 0.09, 0.22, 0.18, 0.43),
            (0.22, 0.12, 0.18, 0.16, 0.32),
            (0.14, 0.18, 0.15, 0.12, 0.41),
            (0.12, 0.11, 0.25, 0.21, 0.31),
            (0.12, 0.1, 0.28, 0.22, 0.28),
        ]
    else:
        # Maniac, rock-bot, GTO-ish, overbet, limp-fold bot, blend
        tbl = [
            (0.55, 0.22, 0.08, 0.04, 0.11),
            (0.08, 0.05, 0.15, 0.12, 0.60),
            (0.28, 0.18, 0.12, 0.10, 0.32),
            (0.38, 0.25, 0.10, 0.06, 0.21),
            (0.18, 0.08, 0.35, 0.08, 0.31),
            (0.42, 0.28, 0.12, 0.06, 0.12),
        ]
    raw = np.array(tbl[regime], dtype=np.float64)
    raw /= raw.sum()
    raise_bias, bet_bias, fold_bias, check_bias, call_bias = raw.tolist()
    return raise_bias, bet_bias, fold_bias, check_bias, call_bias


def generate_hand(rng: np.random.Generator, label: Label) -> Dict[str, Any]:
    """Return a deep-copied canonical hand dict (including ``label``)."""
    bb = float(rng.choice([0.01, 0.02, 0.03, 0.05, 0.1, 0.25]))
    sb = bb / 2.0
    n_players = int(rng.integers(2, 7))
    button = int(rng.integers(1, n_players + 1))

    hand = copy.deepcopy(V0_JSON_HAND)
    hand["label"] = label
    hand["metadata"] = {
        "game_type": "Hold'em",
        "limit_type": "No Limit",
        "max_seats": 6,
        "hero_seat": int(rng.integers(1, n_players + 1)),
        "hand_ended_on_street": "preflop",
        "button_seat": button,
        "sb": sb,
        "bb": bb,
        "ante": 0.0,
        "rng_seed_commitment": None,
    }
    hand["players"] = _players(n_players, bb, rng)

    sb_seat = (button % n_players) + 1
    bb_seat = (sb_seat % n_players) + 1

    actions: List[Dict[str, Any]] = []
    pot = 0.0
    aid = 1

    def add(act: Dict[str, Any]) -> None:
        nonlocal aid, pot
        actions.append(act)
        pot = act["pot_after"]
        aid += 1

    add(_action(aid, "preflop", sb_seat, "small_blind", bb, pot, sb))
    add(_action(aid, "preflop", bb_seat, "big_blind", bb, pot, bb))

    street = "preflop"
    depth_target = int(rng.integers(0, 4))  # postflop depth
    raise_bias, bet_bias, fold_bias, check_bias, call_bias = _regime_biases(rng, label)

    acting = (bb_seat % n_players) + 1
    steps = int(rng.integers(6, 36))
    current_bet = bb

    p_fold = fold_bias
    p_check = p_fold + check_bias
    p_call = p_check + call_bias
    p_raise = p_call + raise_bias
    p_bet = p_raise + bet_bias

    for _ in range(steps):
        r = rng.random()
        if street != "preflop" and r < p_fold:
            add(_action(aid, street, acting, "fold", bb, pot, 0.0))
        elif street != "preflop" and r < p_check:
            add(_action(aid, street, acting, "check", bb, pot, 0.0))
        elif street == "preflop" and r < p_fold and len(actions) > 3:
            add(_action(aid, street, acting, "fold", bb, pot, 0.0))
        elif r < p_call:
            to_call = max(0.0, current_bet)
            amt = min(to_call + float(rng.uniform(0, bb * 0.2)), bb * 8) if to_call > 0 else bb
            add(_action(aid, street, acting, "call", bb, pot, round(amt, 4)))
        elif r < p_raise:
            bump = float(rng.uniform(bb, bb * 5))
            add(
                _action(
                    aid,
                    street,
                    acting,
                    "raise",
                    bb,
                    pot,
                    bump,
                    raise_to=round(current_bet + bump, 4),
                )
            )
            current_bet += bump
        elif r < p_bet:
            add(
                _action(
                    aid,
                    street,
                    acting,
                    "bet",
                    bb,
                    pot,
                    round(float(rng.uniform(bb, bb * 4)), 4),
                )
            )
        else:
            add(
                _action(
                    aid,
                    street,
                    acting,
                    "call",
                    bb,
                    pot,
                    round(float(rng.uniform(0, bb * 2)), 4),
                )
            )

        acting = (acting % n_players) + 1
        if rng.random() < 0.08 and street == "preflop" and depth_target > 0:
            street = "flop"
            current_bet = 0.0
        elif rng.random() < 0.06 and street == "flop":
            street = "turn"
            current_bet = 0.0
        elif rng.random() < 0.05 and street == "turn":
            street = "river"
            current_bet = 0.0

    street_depth_idx = min(3, max(0, depth_target))
    hand["streets"] = _streets(street_depth_idx)
    meta_street = ("preflop", "flop", "turn", "river")[min(3, street_depth_idx)]
    hand["metadata"]["hand_ended_on_street"] = meta_street
    hand["actions"] = actions
    showdown = bool(rng.random() < (0.15 if label == "human" else 0.35))
    hand["outcome"] = {
        "winners": [hand["players"][0]["player_uid"]],
        "payouts": {hand["players"][0]["player_uid"]: round(pot * 0.95, 2)},
        "total_pot": round(pot, 2),
        "rake": round(pot * 0.05, 2),
        "result_reason": "showdown" if showdown else "fold",
        "showdown": showdown,
    }
    return hand


def generate_chunk(
    rng: np.random.Generator,
    label: Label,
    size: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Return (canonical_hands, chunk_label) with chunk_label 0 human 1 bot."""
    hands = [generate_hand(rng, label) for _ in range(max(1, size))]
    y = 0 if label == "human" else 1
    return hands, y


def generate_labeled_chunks(
    n_chunks: int,
    *,
    seed: int = 42,
    chunk_size_lo: int = 1,
    chunk_size_hi: int = 4,
) -> List[Tuple[List[Dict[str, Any]], int]]:
    rng = np.random.default_rng(seed)
    out: List[Tuple[List[Dict[str, Any]], int]] = []
    for i in range(n_chunks):
        label: Label = "human" if i % 2 == 0 else "bot"
        sz = int(rng.integers(chunk_size_lo, chunk_size_hi + 1))
        out.append(generate_chunk(rng, label, sz))
    return out
