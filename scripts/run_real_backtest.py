#!/usr/bin/env python3
"""
run_real_backtest.py

Run real-data backtest for 0DTE SPY put credit spread strategy using pre-downloaded minute bars.

Usage:
    python run_real_backtest.py SPY --start YYYY-MM-DD --end YYYY-MM-DD --data-dir backtest_data --output-dir backtests/real
"""
import argparse
from datetime import date
from backtest_spy import real_backtest_symbol


def parse_args():
    parser = argparse.ArgumentParser(description="Run real-data backtest using pre-downloaded data")
    parser.add_argument(
        "symbol", help="Ticker symbol to backtest"
    )
    parser.add_argument(
        "--start", required=True, help="Start date YYYY-MM-DD"
    )
    parser.add_argument(
        "--end", required=True, help="End date YYYY-MM-DD"
    )
    parser.add_argument(
        "--data-dir", required=True, help="Directory with downloaded backtest data"
    )
    parser.add_argument(
        "--output-dir", default="backtests/real", help="Directory to store real backtest results"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    real_backtest_symbol(
        args.symbol, start_date, end_date, args.output_dir, args.data_dir
    )


if __name__ == "__main__":
    main()
