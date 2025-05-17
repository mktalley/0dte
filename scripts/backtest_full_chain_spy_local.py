#!/usr/bin/env python3
"""
backtest_full_chain_spy_local.py

Backtest full-chain 0DTE put credit spread strategy using pre-downloaded local data.
Requires download_backtest_data.py to have saved data in <data_dir>/<symbol>/YYYY-MM-DD with chain.csv and minute bar CSVs.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, date
from zoneinfo import ZoneInfo
from scipy.stats import norm


def put_delta(S, K, r, T, sigma):
    """Black-Scholes put delta."""
    if T <= 0:
        return 0 if S >= K else -1
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1) - 1


def main():
    parser = argparse.ArgumentParser(description="Backtest SPY 0DTE put credit spreads from local data")
    parser.add_argument("--symbol", default="SPY", help="Underlying symbol")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--data-dir", default="backtest_data", help="Base directory for downloaded data")
    parser.add_argument("--output-dir", default="backtests", help="Directory to save backtest results")
    parser.add_argument("--min-credit-pct", type=float, default=0.15,
                        help="Min credit as fraction of spread width")
    parser.add_argument("--strike-range", type=float, default=0.30,
                        help="Strike range fraction around spot (e.g. 0.3 for ±30%)")
    parser.add_argument("--short-delta", nargs=2, type=float, metavar=("LOW","HIGH"),
                        default=[-0.6, -0.4], help="Short put delta range")
    parser.add_argument("--long-delta", nargs=2, type=float, metavar=("LOW","HIGH"),
                        default=[-0.4, -0.2], help="Long put delta range")
    parser.add_argument("--risk-caps", nargs='+', type=float, default=[300.0],
                        help="List of max per-contract risk in dollars")
    args = parser.parse_args()

    sym = args.symbol.upper()
    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    base_dir = Path(args.data_dir)
    data_dir = base_dir / sym
    out_base = Path(args.output_dir)
    r = 0.01
    T = 1/252
    short_min, short_max = args.short_delta
    long_min, long_max = args.long_delta
    min_credit_pct = args.min_credit_pct
    strike_range = args.strike_range
    risk_caps = args.risk_caps

    # Load daily underlying data
    underlying_file = data_dir / 'underlying.csv'
    if not underlying_file.exists():
        raise FileNotFoundError(f"Underlying data not found: {underlying_file}")
    df_daily = pd.read_csv(underlying_file, index_col=0, parse_dates=True)
    # Expect columns: Open, Close, VIX

    for cap in risk_caps:
        records = []
        for idx, row in df_daily.iterrows():
            d = idx.date()
            if d < start_date or d > end_date:
                continue
            date_str = d.isoformat()
            day_dir = data_dir / date_str
            if not day_dir.exists():
                continue
            S_open = float(row['Open'])
            S_close = float(row['Close'])
            # VIX close for sigma
            sigma = float(row.get('VIX', np.nan)) / 100.0
            if np.isnan(sigma) or sigma <= 0:
                sigma = 0.2

            # Load full chain
            chain_file = day_dir / 'chain.csv'
            if not chain_file.exists():
                continue
            df_chain = pd.read_csv(chain_file)
            # Filter by strike range
            lo = S_open * (1 - strike_range)
            hi = S_open * (1 + strike_range)
            df_chain = df_chain[(df_chain['strike'] >= lo) & (df_chain['strike'] <= hi)].copy()
            if df_chain.empty:
                continue
            # Mid price
            df_chain['mid'] = (df_chain['bid'] + df_chain['ask']) / 2
            df_chain = df_chain.dropna(subset=['mid'])
            if df_chain.empty:
                continue
            # Compute delta
            df_chain['delta'] = df_chain['strike'].apply(lambda K: put_delta(S_open, K, r, T, sigma))
            # Select candidates
            shorts = df_chain[(df_chain['delta'] >= short_min) & (df_chain['delta'] <= short_max)]
            longs = df_chain[(df_chain['delta'] >= long_min) & (df_chain['delta'] <= long_max)]
            if shorts.empty or longs.empty:
                continue
            # Find best spread
            best = None
            for _, rs in shorts.iterrows():
                for _, rl in longs.iterrows():
                    K_s, mid_s = rs['strike'], rs['mid']
                    K_l, mid_l = rl['strike'], rl['mid']
                    if K_s <= K_l:
                        continue
                    width = K_s - K_l
                    credit = mid_s - mid_l
                    if credit < min_credit_pct * width:
                        continue
                    if width * 100 > cap:
                        continue
                    if best is None or credit > best['credit']:
                        best = dict(date=d, cap=cap,
                                    K_short=K_s, K_long=K_l,
                                    mid_short=mid_s, mid_long=mid_l,
                                    delta_short=rs['delta'], delta_long=rl['delta'],
                                    width=width, credit=credit,
                                    S_open=S_open, S_close=S_close)
            if best is None:
                continue
            # Compute payoff and PnL
            payoff = max(best['K_short'] - best['S_close'], 0) - max(best['K_long'] - best['S_close'], 0)
            pnl_share = best['credit'] - payoff
            pnl = pnl_share * 100
            best.update(payoff=payoff, pnl=pnl, win=pnl>0)
            records.append(best)
        # Summarize
        df = pd.DataFrame(records)
        total = len(df)
        wins = int(df['win'].sum()) if total>0 else 0
        losses = total - wins
        pnl_total = df['pnl'].sum() if total>0 else 0.0
        win_rate = wins/total*100 if total>0 else 0.0
        print(f"\n=== Risk cap ${cap}: Trades={total}, Wins={wins}, Losses={losses}, WinRate={win_rate:.1f}%, PnL=${pnl_total:.2f} ===")
        # Save CSV
        out_dir = out_base / f"{date.today().isoformat()}_{sym}_local"
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_dir / f"{sym}_local_cap{int(cap)}.csv", index=False)
    print("Done.")

if __name__ == '__main__':
    main()