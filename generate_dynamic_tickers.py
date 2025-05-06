import requests
import datetime
import os

# === CONFIG ===
API_URL = "https://api.ivolatility.com/ivrank"  # Placeholder, use actual endpoint if available
FALLBACK_TICKERS = [
    "SPY", "QQQ", "TSLA", "AAPL", "MSFT",
    "NVDA", "META", "AMZN", "AMD", "GOOG",
    "BA", "XLF", "XLK", "DIA", "IWM",
    "SPX", "VIX", "XLE", "XBI", "TSM", "GDX", "ARKK"
]
NUM_TO_SELECT = 15
OUTPUT_DIR = "tickers_selected"

# === MAIN ===
def fetch_high_iv_tickers():
    try:
        # Placeholder fetch; use real IV Rank source/API here
        response = requests.get("https://raw.githubusercontent.com/financeai/data/main/ivrank_demo.json")
        tickers_data = response.json()
        sorted_tickers = sorted(tickers_data, key=lambda x: x["iv_rank"], reverse=True)
        top_tickers = [x["ticker"] for x in sorted_tickers[:NUM_TO_SELECT]]
        print(f"✅ Top IV tickers selected: {top_tickers}")
        return top_tickers
    except Exception as e:
        print(f"⚠️ Error fetching IV ranks, using fallback list. Error: {e}")
        return FALLBACK_TICKERS[:NUM_TO_SELECT]

def save_tickers(tickers):
    today = datetime.date.today().isoformat()
    filename = f"{OUTPUT_DIR}/tickers_selected_{today}.txt"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(filename, "w") as f:
        for t in tickers:
            f.write(f"{t}\n")
    print(f"✅ Tickers saved to {filename}")

if __name__ == "__main__":
    tickers = fetch_high_iv_tickers()
    save_tickers(tickers)