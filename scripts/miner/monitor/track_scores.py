#!/usr/bin/env python3
"""Periodically poll the Bittensor metagraph for subnet 126 and record miner scores.

Stores results in a SQLite database at data/scores.db.
Tracked miners are read from TRACKED_HOTKEYS (wallet name + hotkey pairs).

Usage (run from repo root with miner_env active):
    python scripts/miner/monitor/track_scores.py
    python scripts/miner/monitor/track_scores.py --interval 300   # poll every 5 min
    python scripts/miner/monitor/track_scores.py --show           # print recent history
    python scripts/miner/monitor/track_scores.py --show --rows 50 # last 50 records
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

DB_PATH = REPO / "data" / "scores.db"

WALLET_NAME = "Juker126"
TRACKED_HOTKEYS = ["miner1", "miner2", "miner3"]
NETUID = 126
NETWORK = "finney"
DEFAULT_INTERVAL_SEC = 600  # 10 minutes (~every 50 blocks)


# ── Database setup ──────────────────────────────────────────────────────────────

def get_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.execute("""
        CREATE TABLE IF NOT EXISTS scores (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            recorded_at TEXT    NOT NULL,
            block       INTEGER NOT NULL,
            uid         INTEGER NOT NULL,
            hotkey_name TEXT    NOT NULL,
            hotkey_ss58 TEXT    NOT NULL,
            incentive   REAL    NOT NULL,
            consensus   REAL    NOT NULL,
            stake       REAL    NOT NULL,
            emission    REAL    NOT NULL,
            active      INTEGER NOT NULL,
            last_update INTEGER NOT NULL
        )
    """)
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_scores_uid_block
        ON scores (uid, block)
    """)
    con.commit()
    return con


def insert_row(con: sqlite3.Connection, row: dict) -> None:
    con.execute("""
        INSERT INTO scores
            (recorded_at, block, uid, hotkey_name, hotkey_ss58,
             incentive, consensus, stake, emission, active, last_update)
        VALUES
            (:recorded_at, :block, :uid, :hotkey_name, :hotkey_ss58,
             :incentive, :consensus, :stake, :emission, :active, :last_update)
    """, row)
    con.commit()


# ── Metagraph polling ───────────────────────────────────────────────────────────

def poll_once(con: sqlite3.Connection) -> list[dict]:
    import bittensor as bt

    sub = bt.Subtensor(network=NETWORK)
    mg = sub.metagraph(NETUID)
    block = int(mg.block.item())
    now = datetime.now(timezone.utc).isoformat()

    rows = []
    for hotkey_name in TRACKED_HOTKEYS:
        try:
            wallet = bt.Wallet(name=WALLET_NAME, hotkey=hotkey_name)
            ss58 = wallet.hotkey.ss58_address
            uid = mg.hotkeys.index(ss58)
        except Exception as e:
            print(f"  [{hotkey_name}] not found in metagraph: {e}")
            continue

        row = {
            "recorded_at": now,
            "block":        block,
            "uid":          uid,
            "hotkey_name":  hotkey_name,
            "hotkey_ss58":  ss58,
            "incentive":    float(mg.I[uid].item()),
            "consensus":    float(mg.C[uid].item()),
            "stake":        float(mg.S[uid].item()),
            "emission":     float(mg.E[uid].item()),
            "active":       int(mg.active[uid].item()),
            "last_update":  int(mg.last_update[uid].item()),
        }
        insert_row(con, row)
        rows.append(row)

    return rows


# ── Display ─────────────────────────────────────────────────────────────────────

def show_recent(con: sqlite3.Connection, n_rows: int = 30) -> None:
    cur = con.execute("""
        SELECT recorded_at, block, hotkey_name, uid,
               incentive, consensus, emission, active, last_update
        FROM scores
        ORDER BY id DESC
        LIMIT ?
    """, (n_rows,))
    rows = cur.fetchall()
    if not rows:
        print("No records yet.")
        return

    print(f"\n{'Time (UTC)':<22} {'Block':>8}  {'Miner':<8} {'UID':>4}  "
          f"{'Incentive':>10}  {'Consensus':>10}  {'Emission':>10}  {'Active':>6}  {'LastUpd':>9}")
    print("-" * 105)
    for r in reversed(rows):
        ts = r["recorded_at"][:19].replace("T", " ")
        blocks_ago = ""
        print(
            f"{ts:<22} {r['block']:>8}  {r['hotkey_name']:<8} {r['uid']:>4}  "
            f"{r['incentive']:>10.6f}  {r['consensus']:>10.6f}  {r['emission']:>10.6f}  "
            f"{'yes' if r['active'] else 'no':>6}  {r['last_update']:>9}"
        )

    # Summary: latest incentive per miner
    print("\nLatest incentive per miner:")
    cur2 = con.execute("""
        SELECT hotkey_name, uid, incentive, block, recorded_at
        FROM scores
        WHERE id IN (
            SELECT MAX(id) FROM scores GROUP BY hotkey_name
        )
        ORDER BY hotkey_name
    """)
    for r in cur2.fetchall():
        ts = r["recorded_at"][:19].replace("T", " ")
        print(f"  {r['hotkey_name']:<8} uid={r['uid']:>3}  incentive={r['incentive']:.6f}"
              f"  block={r['block']}  at={ts}")


# ── Main loop ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SEC,
                   help="Seconds between polls (default: 600)")
    p.add_argument("--show", action="store_true",
                   help="Print recent records from DB and exit")
    p.add_argument("--rows", type=int, default=30,
                   help="Number of rows to show with --show (default: 30)")
    p.add_argument("--db", type=str, default=str(DB_PATH),
                   help="Path to SQLite database file")
    p.add_argument("--once", action="store_true",
                   help="Poll once, print results, and exit")
    args = p.parse_args()

    db_path = Path(args.db)
    con = get_db(db_path)

    if args.show:
        show_recent(con, args.rows)
        return

    if args.once:
        print(f"Polling metagraph (netuid={NETUID}, network={NETWORK}) ...")
        rows = poll_once(con)
        for r in rows:
            print(f"  {r['hotkey_name']:<8} uid={r['uid']:>3}  "
                  f"incentive={r['incentive']:.6f}  block={r['block']}")
        print(f"Saved to {db_path}")
        return

    print(f"Score tracker started. Polling every {args.interval}s. DB: {db_path}")
    print(f"Tracking: {TRACKED_HOTKEYS} on netuid={NETUID}")
    print("Use --show to view history. Ctrl+C to stop.\n")

    while True:
        try:
            print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC] Polling ...")
            rows = poll_once(con)
            for r in rows:
                print(f"  {r['hotkey_name']:<8} uid={r['uid']:>3}  "
                      f"incentive={r['incentive']:.6f}  consensus={r['consensus']:.4f}  "
                      f"emission={r['emission']:.6f}  active={'yes' if r['active'] else 'no'}")
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as exc:
            print(f"  ERROR: {exc}")

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
