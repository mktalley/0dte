#!/usr/bin/env python3
"""
Generate daily ticker list based on average volume liquidity filter.
Saves top N tickers into data/tickers_selected/tickers_selected_YYYY-MM-DD.txt
"""
import os
import datetime
import yfinance as yf
import numpy as np

# === CONFIGURATION ===
# Candidate universe (modify as needed)
CANDIDATE_TICKERS = [
    "SPY", "QQQ", "DIA", "IWM", "XLF", "XLE", "XBI",  # broad ETFs
    "AAPL", "MSFT", "NVDA", "GOOG", "META", "AMZN", "TSLA", # tech names
    "AMD", "BA", "TSM", "GDX", "ARKK"  # other high-volume plays
]
NUM_TO_SELECT = int(os.getenv("NUM_SELECT_TICKERS", "15"))
MIN_AVG_VOLUME = int(os.getenv("MIN_AVG_VOLUME", "1000000"))  # shares
VOLUME_PERIOD_DAYS = int(os.getenv("VOLUME_PERIOD_DAYS", "30"))
OUTPUT_DIR = os.path.join("data", "tickers_selected")


def fetch_liquid_tickers():
    """Fetch tickers whose average volume over the past period exceeds threshold."""
    volumes = {}
    for ticker in CANDIDATE_TICKERS:
        try:
            hist = yf.Ticker(ticker).history(period=f"{VOLUME_PERIOD_DAYS}d")
            if hist.empty or 'Volume' not in hist.columns:
                continue
            avg_vol = hist['Volume'].mean()
            if avg_vol >= MIN_AVG_VOLUME:
                volumes[ticker] = avg_vol
        except Exception as e:
            print(f"⚠️ Error fetching volume for {ticker}: {e}")
    # Sort tickers by descending average volume
    sorted_tickers = sorted(volumes.items(), key=lambda x: x[1], reverse=True)
    selected = [t[0] for t in sorted_tickers][:NUM_TO_SELECT]
    # If not enough, fill with highest-volume from candidates regardless of threshold
    if len(selected) < NUM_TO_SELECT:
        extras = [t for t, _ in sorted(volumes.items(), key=lambda x: x[1], reverse=True) if t not in selected]
        for t in extras:
            selected.append(t)
            if len(selected) >= NUM_TO_SELECT:
                break
    print(f"✅ Selected {len(selected)} tickers: {selected}")
    return selected


def save_tickers(tickers):
    """Save tickers list to OUTPUT_DIR with today's date."""
    today = datetime.date.today().isoformat()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.join(OUTPUT_DIR, f"tickers_selected_{today}.txt")
    with open(filename, "w") as f:
        for t in tickers:
            f.write(f"{t}\n")
    print(f"✅ Tickers saved to {filename}")


if __name__ == "__main__":
    tickers = fetch_liquid_tickers()
    save_tickers(tickers)
