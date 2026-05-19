#!/bin/bash
# PokverV3 miner startup (PM2 + venv Python)

NETUID="${NETUID:-126}"
WALLET_NAME="${WALLET_NAME:-Juker126}"
HOTKEY="${HOTKEY:-miner1}"
NETWORK="${NETWORK:-finney}"
MINER_SCRIPT="${MINER_SCRIPT:-./neurons/miner_v1.py}"
PM2_NAME="${PM2_NAME:-poker44_miner1}"
AXON_PORT="${AXON_PORT:-8091}"
ALLOWED_VALIDATOR_HOTKEYS="${ALLOWED_VALIDATOR_HOTKEYS:-}"

export POKER44_V1_VARIANT="${POKER44_V1_VARIANT:-v1_b_deeper_adaptive}"
export POKER44_MAX_BOT_FRACTION="${POKER44_MAX_BOT_FRACTION:-0.22}"
export POKER44_SAVE_RAW_CHUNKS="${POKER44_SAVE_RAW_CHUNKS:-0}"
export POKER44_MODEL_NAME="${POKER44_MODEL_NAME:-pokver-v3-adaptive}"
export POKER44_MODEL_VERSION="${POKER44_MODEL_VERSION:-1}"
export POKER44_MODEL_REPO_URL="${POKER44_MODEL_REPO_URL:-https://github.com/browndev7777-alt/PokverV3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT" || exit 1

if [ -x "$REPO_ROOT/miner_env/bin/python" ]; then
  PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/miner_env/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if [ ! -f "$MINER_SCRIPT" ]; then
    echo "Error: Miner script not found at $MINER_SCRIPT"
    exit 1
fi

if ! command -v pm2 &> /dev/null; then
    echo "Error: PM2 is not installed (npm install -g pm2)"
    exit 1
fi

pm2 delete "$PM2_NAME" 2>/dev/null || true

export PYTHONPATH="$(pwd)"

if [[ -z "${POKER44_MODEL_REPO_COMMIT:-}" ]] && [[ -d "$REPO_ROOT/.git" ]]; then
  if FULL_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null)"; then
    export POKER44_MODEL_REPO_COMMIT="$FULL_SHA"
  fi
fi

MINER_ARGS=(
  "$MINER_SCRIPT"
  --netuid "$NETUID"
  --wallet.name "$WALLET_NAME"
  --wallet.hotkey "$HOTKEY"
  --subtensor.network "$NETWORK"
  --axon.port "$AXON_PORT"
  --logging.debug
)

if [ -n "$ALLOWED_VALIDATOR_HOTKEYS" ]; then
  read -r -a VALIDATOR_HOTKEY_ARRAY <<< "$ALLOWED_VALIDATOR_HOTKEYS"
  MINER_ARGS+=(--blacklist.allowed_validator_hotkeys "${VALIDATOR_HOTKEY_ARRAY[@]}")
else
  MINER_ARGS+=(--blacklist.force_validator_permit)
fi

pm2 start "$PYTHON_BIN" \
  --name "$PM2_NAME" \
  --cwd "$REPO_ROOT" \
  --interpreter none \
  -- "${MINER_ARGS[@]}"

pm2 save

echo "Miner started: $PM2_NAME"
echo "Python: $PYTHON_BIN"
echo "Variant: $POKER44_V1_VARIANT"
echo "Manifest repo_url: ${POKER44_MODEL_REPO_URL:-}"
echo "Manifest repo_commit: ${POKER44_MODEL_REPO_COMMIT:-}"
echo "View logs: pm2 logs $PM2_NAME"
