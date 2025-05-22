#!/usr/bin/env bash
set -euo pipefail

# Load environment variables from .env if present
if [ -f .env ]; then
  set -o allexport
  source .env
  set +o allexport
fi

# Wrapper to run full pipeline: backtest -> charts -> parameter optimize
# Usage:
#   export ALPACA_API_KEY=...
#   export ALPACA_SECRET_KEY=...
#   bash scripts/run_full_pipeline.sh

if [[ -z "${ALPACA_API_KEY:-}" || -z "${ALPACA_SECRET_KEY:-}" ]]; then
  echo "Error: Please export ALPACA_API_KEY and ALPACA_SECRET_KEY"
  exit 1
fi

START_DATE=$(date -d "365 days ago" '+%Y-%m-%d')
END_DATE=$(date '+%Y-%m-%d')
OUTPUT_DIR="backtests/$(date '+%Y-%m-%d')_SPY_FULL"
mkdir -p "$OUTPUT_DIR"

# Step 1: run minute-level backtest
echo "[1/4] Running minute-level SPY backtest from $START_DATE to $END_DATE..."
python3 scripts/backtest_spy_alpaca.py \
  --start "$START_DATE" \
  --end   "$END_DATE" \
  --output "$OUTPUT_DIR/trades.csv"

# Step 2: generate charts
echo "[2/4] Generating charts..."
python3 scripts/generate_charts_fixed.py \
  --input "$OUTPUT_DIR/trades.csv" \
  --outdir "$OUTPUT_DIR/charts"

# Step 3: refine costs (already baked in compute_pnl)
echo "[3/4] Costs (commission & slippage) applied in compute_pnl"

# Step 4: parameter optimization
echo "[4/4] Running parameter optimization for SPY over same period..."
python3 scripts/optimize_parameters.py SPY \
  --start "$START_DATE" \
  --end   "$END_DATE" \
  --output "$OUTPUT_DIR/optimize_results.csv"

echo "Full pipeline complete. Results in $OUTPUT_DIR"