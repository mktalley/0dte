#!/usr/bin/env bash
# Auto-restart wrapper for the 0DTE SPY bot

# Move to script's directory (repository root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export TZ="America/Los_Angeles"  # Ensure timestamps use Pacific Time
cd "$ROOT_DIR"

LOG_DIR="logs"

while true; do
  # Build dated log file name
  current_date=$(date '+%Y-%m-%d')
  LOG_FILE="$LOG_DIR/paper_bot_${current_date}.log"
  mkdir -p "$LOG_DIR"

  # Log start time
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🚀 Starting bot" >> "$LOG_FILE"
  # Calculate seconds until next midnight PST for log rotation
  now=$(date '+%s')
  tomorrow=$(date -d 'tomorrow 00:00' '+%s')
  duration=$((tomorrow - now))
  # Run the bot until midnight PST (or until it exits)
  timeout "${duration}s" python3 -u "$ROOT_DIR/src/main.py" >> "$LOG_FILE" 2>&1
  EXIT_CODE=$?
  if [ $EXIT_CODE -eq 124 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🔄 Midnight PST reached; rotating log file" >> "$LOG_FILE"
  fi
  # Log exit and restart info
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ Bot exited with code $EXIT_CODE; restarting in 5s" >> "$LOG_FILE"
  # Wait before restarting
  sleep 5
done
