import requests
import csv
from datetime import datetime

# === CONFIG ===
TRADIER_TOKEN = "84jHmJl0dIGjRMtmQDpleA1PMDp3"
BASE_URL = "https://api.tradier.com/v1/markets"
HEADERS = {
    "Authorization": f"Bearer {TRADIER_TOKEN}",
    "Accept": "application/json"
}

symbol = "SPY"
expiration = "2024-04-19"  # <-- rock-solid SPY expiration date
output_file = "spy_tradier_0dte.csv"

# === GET OPTION CHAIN ===
params = {
    "symbol": symbol,
    "expiration": expiration,
    "greeks": "true",
    "includeAllRoots": "true",
    "format": "json"
}

response = requests.get(f"{BASE_URL}/options/chains", headers=HEADERS, params=params)
print("🔎 Raw response:\n", response.text)

try:
    data = response.json()
except requests.exceptions.JSONDecodeError:
    print("❌ Failed to decode JSON. Raw text was not valid JSON.")
    exit()

if not data or "options" not in data or not data["options"] or "option" not in data["options"]:
    print("❌ No options returned. Check expiration date, symbol, or token permissions.")
    exit()

chain = data["options"]["option"]
puts = [opt for opt in chain if opt["option_type"] == "put"]
print(f"✅ Found {len(puts)} SPY puts expiring {expiration}")

# === WRITE TO CSV ===
with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "symbol", "strike", "bid", "ask", "delta", "theta", "implied_volatility"
    ])
    writer.writeheader()
    for opt in puts:
        writer.writerow({
            "symbol": opt["symbol"],
            "strike": opt["strike"] or 0,
            "bid": opt["bid"] or 0,
            "ask": opt["ask"] or 0,
            "delta": opt.get("greeks", {}).get("delta"),
            "theta": opt.get("greeks", {}).get("theta"),
            "implied_volatility": opt.get("greeks", {}).get("iv")
        })

print(f"📁 Saved to {output_file}")