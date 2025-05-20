#!/usr/bin/env python3
"""
run_backtest_spy_last_year.py

Run a backtest for SPY over the past year using the 0DTE put credit spread strategy.
Usage:
    python run_backtest_spy_last_year.py
"""
import os
import sys
# Add project root to PYTHONPATH so scripts package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import date, timedelta
from pathlib import Path
from scripts.backtest_spy import backtest_symbol

def main():
    end_date = date.today()
    start_date = end_date - timedelta(days=365)
    output_dir = Path("backtests") / f"SPY_{start_date.isoformat()}_to_{end_date.isoformat()}"
    backtest_symbol("SPY", start_date, end_date, output_dir)

if __name__ == "__main__":
    main()
