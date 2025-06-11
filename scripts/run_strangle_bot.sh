#!/usr/bin/env bash
# Auto-restart wrapper for the 0DTE Strangle Bot

# Move to script's directory (repository root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export TZ="America/Los_Angeles"
cd "$ROOT_DIR"

# Auto-install Python dependencies
mkdir -p logs
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔄 Installing Python dependencies" >> logs/setup.log
python3 -m pip install -r requirements.txt --quiet

LOG_DIR="logs"

while true; do
  # Build dated log file name for strangle bot
  current_date=$(date '+%Y-%m-%d')
  LOG_FILE="$LOG_DIR/strangle_bot_${current_date}.log"
  mkdir -p "$LOG_DIR"

  # Log start time
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🚀 Starting Strangle Bot" >> "$LOG_FILE"

  # Calculate seconds until next midnight PST for log rotation
  now=$(date '+%s')
  tomorrow=$(date -d 'tomorrow 00:00' '+%s')
  duration=$((tomorrow - now))

  # Run the strangle bot until midnight PST (or until it exits)
  timeout "${duration}s" python3 -u "$ROOT_DIR/scripts/trading_bot.py" >> "$LOG_FILE" 2>&1
  EXIT_CODE=$?

  if [ $EXIT_CODE -eq 124 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔄 Midnight PST reached; rotating log file" >> "$LOG_FILE"
  fi

  # Log exit and restart info
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ Strangle Bot exited with code $EXIT_CODE; restarting in 5s" >> "$LOG_FILE"
  sleep 5

done
