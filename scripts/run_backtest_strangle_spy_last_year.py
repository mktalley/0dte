#!/usr/bin/env python3
"""
run_backtest_strangle_spy_last_year.py

Run a daily 0DTE SPY short strangle backtest over the past year.
"""
import sys, os
# Add project root to PYTHONPATH to import scripts package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import date, timedelta
from argparse import ArgumentParser
import math
from pathlib import Path
import pandas as pd
import yfinance as yf
from scripts.optimize_strangle_spy import backtest_strangle, backtest_dynamic_strangle

# Dynamic contract sizing based on risk configuration (align with trading_bot)
ACCOUNT_CAPITAL = float(os.getenv('ACCOUNT_CAPITAL', '38000'))  # total cash available
DAILY_RISK_PCT  = float(os.getenv('DAILY_RISK_PCT', '0.02'))   # percent of capital to risk per day
RISK_PER_CONTRACT = float(os.getenv('RISK_PER_CONTRACT', '100')) # worst-case loss per contract
DAILY_RISK = ACCOUNT_CAPITAL * DAILY_RISK_PCT
CONTRACTS_PER_DAY = max(1, int(DAILY_RISK / RISK_PER_CONTRACT))  # number of strangle contracts per day based on risk budget

# Parse command-line arguments
parser = ArgumentParser(description="Run daily 0DTE SPY short strangle backtest")
parser.add_argument('-d', '--dynamic', action='store_true', help='Enable dynamic intraday leg adjustments')
parser.add_argument('--drop-pct', type=float, default=0.002, help='Threshold for drop from open to remove call/add put (default: 0.2%)')
parser.add_argument('--rise-pct', type=float, default=0.001, help='Threshold for rise from open to remove put/add call (default: 0.1%)')
parser.add_argument('-n', '--contracts', type=int, default=CONTRACTS_PER_DAY, help='Number of strangle contracts per day')
parser.add_argument('-t', '--target', type=float, default=None, help='Target PnL for suggestion')
parser.add_argument('--start', type=lambda s: date.fromisoformat(s), default=None, help='Start date (YYYY-MM-DD) of backtest')
parser.add_argument('--end', type=lambda s: date.fromisoformat(s), default=None, help='End date (YYYY-MM-DD) of backtest')
args = parser.parse_args()

# Past-year window
def main():
    # Determine backtest window using start/end overrides or past year by default
    end_date = args.end if args.end is not None else date.today()
    start_date = args.start if args.start is not None else end_date - timedelta(days=365)

    # Download SPY and VIX data
    df_spy = yf.download('SPY', start=start_date, end=end_date + timedelta(days=1), progress=False)
    if isinstance(df_spy.columns, pd.MultiIndex):
        df_spy.columns = df_spy.columns.droplevel(level=1)
    df_vix = yf.download('^VIX', start=start_date, end=end_date + timedelta(days=1), progress=False)
    if isinstance(df_vix.columns, pd.MultiIndex):
        df_vix.columns = df_vix.columns.droplevel(level=1)

    # Use best params from optimization
    sp_sd, sp_ld, sc_sd, sc_ld = 0.45, 0.1, 0.35, 0.15

    # Choose backtest method
    if args.dynamic:
        total_pnl, count, win_rate = backtest_dynamic_strangle(
            sp_sd, sp_ld, sc_sd, sc_ld, df_spy, df_vix,
            drop_pct=args.drop_pct, rise_pct=args.rise_pct
        )
        strategy_desc = f"Dynamic Intraday Strangle (drop={args.drop_pct}, rise={args.rise_pct})"
    else:
        total_pnl, count, win_rate = backtest_strangle(
            sp_sd, sp_ld, sc_sd, sc_ld, df_spy, df_vix, daily=True
        )
        strategy_desc = "Static Daily Strangle"



    print(f"===== {strategy_desc} Backtest =====")
    print(f"Period: {start_date} to {end_date}")
    print(f"Params: Short Put Delta -{sp_sd}, Long Put Delta -{sp_ld}, Short Call Delta +{sc_sd}, Long Call Delta +{sc_ld}")
    print(f"Trades: {count}, Win Rate: {win_rate:.1f}%, Total P&L: ${total_pnl:.2f}")

    # Scaled P&L calculation for given contracts per day
    scaled_pnl = total_pnl * args.contracts
    print(f"Contracts per day: {args.contracts}")
    print(f"Scaled Total P&L: ${scaled_pnl:.2f}")
    if args.target is not None:
        if total_pnl > 0:
            req_contracts = math.ceil(args.target / total_pnl)
            print(f"To reach target P&L of ${args.target:.2f}, trade {req_contracts} contracts per day")
        else:
            print("Cannot suggest contracts; per-contract P&L is non-positive.")
    # Save summary.csv
    out_dir = Path("backtests") / f"SPY_{start_date}_strangle"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "strangle_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"===== {strategy_desc} Backtest =====\n")
        f.write(f"Period: {start_date} to {end_date}\n")
        f.write(f"Params: -{sp_sd}/-{sp_ld}/+{sc_sd}/+{sc_ld}\n")
        f.write(f"Trades: {count}, Win Rate: {win_rate:.1f}%, Total PnL: ${total_pnl:.2f}\n")
    print(f"Summary written to {summary_path}")

if __name__ == '__main__':
    main()