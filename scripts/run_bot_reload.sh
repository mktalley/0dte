#!/usr/bin/env bash
# Hot-reload wrapper for the 0DTE bot using watchdog's watchmedo

# Move to script's directory (repository root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Ensure watchdog's watchmedo is installed
if ! command -v watchmedo >/dev/null 2>&1; then
  echo "Error: watchmedo not found. Please install dependencies with 'pip install -r requirements.txt'."
  exit 1
fi

# Launch the bot with auto-reload on Python file changes
watchmedo auto-restart \
  --pattern="*.py" \
  --recursive \
  -- \
  python3 -u src/main.py
