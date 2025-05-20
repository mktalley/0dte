#!/usr/bin/env python3
"""
performance_metrics.py

Compute per-symbol performance metrics (win rate, drawdown, Sharpe) from exit logs.
Usage:
    python scripts/performance_metrics.py --exit-log logs/exit_log.csv --output metrics.csv
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def compute_metrics(df):
    metrics = []
    for sym, group in df.groupby('symbol'):
        group = group.sort_values('timestamp')
        pnls = group['pnl'].values
        total = len(pnls)
        wins = (pnls > 0).sum()
        win_rate = wins / total if total else np.nan
        cum_pnl = np.cumsum(pnls)
        peak = np.maximum.accumulate(cum_pnl)
        drawdowns = peak - cum_pnl
        max_dd = drawdowns.max() if len(drawdowns) else np.nan
        vol = np.std(pnls, ddof=1) if total > 1 else np.nan
        sharpe = (np.mean(pnls) / vol * np.sqrt(total)) if total > 1 and vol > 0 else np.nan
        metrics.append({
            'symbol': sym,
            'trades': total,
            'wins': wins,
            'win_rate': win_rate,
            'total_pnl': cum_pnl[-1] if total else 0,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
        })
    return pd.DataFrame(metrics)


def main():
    parser = argparse.ArgumentParser(description='Compute performance metrics from exit logs')
    parser.add_argument('--exit-log', default='logs/exit_log.csv', help='Path to exit log CSV')
    parser.add_argument('--output', default='performance_metrics.csv', help='Output CSV for metrics')
    args = parser.parse_args()

    path = Path(args.exit_log)
    if not path.exists():
        print(f"Exit log file not found at {path}")
        return

    # exit_log: timestamp,symbol,side,qty,exit_price,pnl,ratio,status
    df = pd.read_csv(path, header=None,
                     names=['timestamp','symbol','side','qty','exit_price','pnl','ratio','status'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0)
    metrics_df = compute_metrics(df)
    metrics_df.to_csv(args.output, index=False)
    print(metrics_df.to_string(index=False))


if __name__ == '__main__':
    main()
