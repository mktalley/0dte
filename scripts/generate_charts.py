#!/usr/bin/env python3
"""
Generate charts from SPY backtest results.
Requirements: pandas, matplotlib.
Usage:
    pip install pandas matplotlib
    python generate_charts.py
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Generate charts from backtest results")
    parser.add_argument('--input', '-i', default='backtest_spy_results.csv', help='Input CSV file for backtest results')
    parser.add_argument('--outdir', '-o', default='charts', help='Output directory for charts')
    return parser.parse_args()


def main(input_csv: str, output_dir: str):
    # Load backtest results
    csv_file = input_csv
    if not os.path.isfile(csv_file):
        print(f"Error: {csv_file} not found.")
        return
    df = pd.read_csv(csv_file, parse_dates=["date"])
    df.sort_values("date", inplace=True)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)


        return
    df = pd.read_csv(csv_file, parse_dates=["date"])  
    df.sort_values("date", inplace=True)

    # Create output directory
    output_dir = "charts"
    os.makedirs(output_dir, exist_ok=True)

    # Equity Curve
    plt.figure(figsize=(10, 6))
    plt.plot(df["date"], df["capital"], lw=2)
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Capital ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "equity_curve.png"))
    plt.close()

    # Drawdown Curve
    cum_max = df["capital"].cummax()
    drawdown = (df["capital"] - cum_max) / cum_max * 100
    plt.figure(figsize=(10, 6))
    plt.plot(df["date"], drawdown, lw=2, color='red')
    plt.title("Drawdown (%)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "drawdown_curve.png"))
    plt.close()

    # PnL Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['pnl'], bins=50, color='skyblue', edgecolor='gray')
    plt.title("PnL per Trade Distribution")
    plt.xlabel("PnL ($)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pnl_distribution.png"))
    plt.close()

    print(f"Charts saved to '{output_dir}/'")

if __name__ == "__main__":
    main()
