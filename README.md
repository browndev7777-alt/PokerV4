# PokverV3 — Poker44 SN126 Bot Detection Miner

Honest fork of [aceguard-engine](https://github.com/Krzysiek99999/aceguard-engine) (MIT) for [Poker44 subnet 126](https://poker44.net) on Bittensor.

## Architecture

LightGBM bot detection with per-batch adaptive calibration:

- **Default variant**: `v1_b_deeper_adaptive` (`POKER44_V1_VARIANT`)
- **Model tag**: LightGBM `B_deeper` under `data/miner_training/`
- **Calibration**: `adaptive_safe_calibrate` (Otsu threshold + safety cap)
- **Default cap**: `POKER44_MAX_BOT_FRACTION=0.22` (override via env)

## Setup

```bash
./scripts/miner/setup.sh
source miner_env/bin/activate
python scripts/miner/training/train_b_deeper.py --samples 5000
```

## Run (PM2)

```bash
export POKER44_MODEL_REPO_URL=https://github.com/browndev7777-alt/PokverV3
./scripts/miner/run/run_miner.sh
```

## License

MIT — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
