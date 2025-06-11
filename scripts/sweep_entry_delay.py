#!/usr/bin/env python3
"""
sweep_entry_delay.py

Run static 0DTE SPY short strangle backtest over the past year
for different entry delays (minutes after market open) to evaluate PnL.
"""
import sys, os
from datetime import date, datetime, timedelta, time
from argparse import ArgumentParser
import pandas as pd
import yfinance as yf

# add project root to PYTHONPATH for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.optimize_strangle_spy import backtest_strangle


def main():
    parser = ArgumentParser(
        description="Sweep SPY 0DTE short strangle backtest over entry delays"
    )
    parser.add_argument(
        '--offsets', '-m', type=int, nargs='+',
        default=[0,5,10,15,20,30],
        help='Minutes after market open for entry (default: 0 5 10 15 20 30)'
    )
    parser.add_argument(
        '--start', type=lambda s: date.fromisoformat(s),
        default=None,
        help='Start date YYYY-MM-DD (default: one year ago)'
    )
    parser.add_argument(
        '--end', type=lambda s: date.fromisoformat(s),
        default=None,
        help='End date YYYY-MM-DD (default: today)'
    )
    parser.add_argument(
        '--contracts', '-n', type=int,
        default=None,
        help='Contracts per day (scales PnL, default uses per-contract basis)'
    )
    args = parser.parse_args()

    today = date.today()
    end_date = args.end or today
    start_date = args.start or (end_date - timedelta(days=365))

    # Download daily SPY & VIX
    print(f"Loading daily SPY and VIX data from {start_date} to {end_date}")
    df_spy = yf.download(
        'SPY', start=start_date, end=end_date + timedelta(days=1), progress=False
    )
    if isinstance(df_spy.columns, pd.MultiIndex):
        df_spy.columns = df_spy.columns.droplevel(1)
    df_vix = yf.download(
        '^VIX', start=start_date, end=end_date + timedelta(days=1), progress=False
    )
    if isinstance(df_vix.columns, pd.MultiIndex):
        df_vix.columns = df_vix.columns.droplevel(1)

    # Download minute-level SPY data for entry prices
    print("Loading minute-level SPY data for entry offsets...")
    df_min = yf.download(
        'SPY', start=start_date, end=end_date + timedelta(days=1),
        interval='1m', progress=False
    )
    # Ensure tz-awareness in Eastern
    try:
        df_min.index = df_min.index.tz_convert('America/New_York')
    except Exception:
        df_min.index = df_min.index.tz_localize('UTC').tz_convert('America/New_York')

    results = []
    for m in args.offsets:
        print(f"Backtesting entry delay = {m} minute(s)")
        # override daily open with minute entry price
        df_spy2 = df_spy.copy()
        for dt in df_spy2.index:
            d = dt.date()
            entry_dt = datetime.combine(d, time(9, 30)) + timedelta(minutes=m)
            # localize entry datetime to match df_min index
            if entry_dt.tzinfo is None:
                entry_dt = entry_dt.replace(tzinfo=df_min.index.tz)
            try:
                price = df_min.at[entry_dt, 'Open']
            except KeyError:
                # skip days without matching minute data
                price = None
            if price is not None and not pd.isna(price):
                df_spy2.at[dt, 'Open'] = price
        # run static daily backtest
        total_pnl, count, win_rate = backtest_strangle(
            0.45, 0.10, 0.35, 0.15,
            df_spy2, df_vix, daily=True
        )
        # scale by contracts if provided
        scaled = total_pnl * args.contracts if args.contracts else total_pnl
        results.append({
            'offset_min': m,
            'trades': count,
            'win_rate_pct': round(win_rate, 2),
            'total_pnl_per_contract': round(total_pnl, 2),
            'scaled_pnl': round(scaled, 2)
        })

    df_res = pd.DataFrame(results)
    out_file = f"backtests/entry_delay_sweep_{start_date}_{end_date}.csv"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df_res.to_csv(out_file, index=False)
    print("Results:")
    print(df_res.to_string(index=False))
    print(f"Saved results to {out_file}")


if __name__ == '__main__':
    main()
