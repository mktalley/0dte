#!/usr/bin/env bash
set -euo pipefail

# Load environment variables from .env if present
if [ -f .env ]; then
  set -o allexport
  source .env
  set +o allexport
fi

# Ensure Alpaca API credentials are set
default_empty=""
if [[ "${ALPACA_API_KEY:-$default_empty}" == "$default_empty" || "${ALPACA_SECRET_KEY:-$default_empty}" == "$default_empty" ]]; then
  echo "Error: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment or .env file."
  exit 1
fi

# Compute date range: past 365 days
START_DATE=$(date -d "365 days ago" '+%Y-%m-%d')
END_DATE=$(date '+%Y-%m-%d')
OUTPUT_DIR="backtests/SPY_${START_DATE}_to_${END_DATE}"
mkdir -p "$OUTPUT_DIR"

echo "[1/1] Running minute-level SPY backtest from $START_DATE to $END_DATE..."
python3 scripts/backtest_spy_alpaca.py \
  --start "$START_DATE" \
  --end   "$END_DATE" \
  --output "$OUTPUT_DIR/trades.csv"

echo "Backtest complete. Results saved to $OUTPUT_DIR/trades.csv"