#!/usr/bin/env bash
# Auto-restart wrapper for the 0DTE SPY bot

# Move to script's directory (repository root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_FILE="logs/paper_bot.log"

while true; do
  # Log start time
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 🚀 Starting bot" >> "$LOG_FILE"
  # Run the bot
  python3 main.py >> "$LOG_FILE" 2>&1
  EXIT_CODE=$?
  # Log exit and restart info
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ Bot exited with code $EXIT_CODE; restarting in 5s" >> "$LOG_FILE"
  # Wait before restarting
  sleep 5
done
