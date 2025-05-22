import os
import csv
from dotenv import load_dotenv
from datetime import datetime
from alpaca.data.historical import OptionHistoricalDataClient
from alpaca.data.requests import OptionChainRequest

# Helper to parse strike price from OCC symbol
def _parse_strike(symbol: str) -> float:
     """Parse strike price from OCC symbol: last 8 digits are strike*1000."""
     return int(symbol[-8:]) / 1000.0

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

client = OptionHistoricalDataClient(API_KEY, API_SECRET)

symbol = "SPY"
expiration = "2024-04-12"  # Choose a Friday expiration

request = OptionChainRequest(
    underlying_symbol=symbol,
    expiration_date=expiration,
    option_type="put"
)

response = client.get_option_chain(request)

results = []
for option in response:
    if option.quote and option.greeks:
        results.append({
            "symbol": option.symbol,
            "strike": _parse_strike(option.symbol),
            "expiration": option.expiration_date,
            "delta": option.greeks.delta,
            "bid": option.quote.bid_price,
            "ask": option.quote.ask_price
        })
    else:
        print(f"⚠️ Incomplete: {option.symbol} | Quote: {option.quote is not None} | Greeks: {option.greeks is not None}")

if results:
    with open("spy_0419_options.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("✅ Saved to spy_0419_options.csv")
else:
    print("⚠️ No contracts with quote and delta data found.")