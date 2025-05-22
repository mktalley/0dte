#!/usr/bin/env python3
"""
generate_charts_fixed.py

Generate charts from backtest results (minute-level or daily trades).
Requirements: pandas, matplotlib.
Usage:
    pip install pandas matplotlib
    python scripts/generate_charts_fixed.py \
      --input backtests/2025-05-11_SPY_FULL/trades.csv \
      --outdir backtests/2025-05-11_SPY_FULL/charts
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Generate charts from backtest results")
    parser.add_argument('--input', '-i', default='backtest_results.csv', help='Input CSV file for backtest results')
    parser.add_argument('--outdir', '-o', default='charts', help='Output directory for charts')
    return parser.parse_args()


def main(input_csv: str, output_dir: str):
    if not os.path.isfile(input_csv):
        print(f"Error: input file '{input_csv}' not found.")
        return
    df = pd.read_csv(input_csv, parse_dates=['date'])
    df.sort_values('date', inplace=True)

    # Compute capital if not present
    if 'capital' not in df.columns:
        df['capital'] = df['pnl'].cumsum()

    os.makedirs(output_dir, exist_ok=True)

    # Equity Curve
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['capital'], lw=2)
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Capital')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'equity_curve.png'))
    plt.close()

    # Drawdown Curve
    cum_max = df['capital'].cummax()
    drawdown = (df['capital'] - cum_max) / cum_max * 100
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], drawdown, lw=2, color='red')
    plt.title('Drawdown (%)')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'drawdown_curve.png'))
    plt.close()

    # PnL Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['pnl'], bins=50, color='skyblue', edgecolor='gray')
    plt.title('PnL per Trade Distribution')
    plt.xlabel('PnL')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pnl_distribution.png'))
    plt.close()

    print(f"Charts saved to '{output_dir}/'")


if __name__ == '__main__':
    args = parse_args()
    main(args.input, args.outdir)
