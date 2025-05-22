import pandas as pd
import datetime

# === CONFIGURATION ===
TARGET_PROFIT = 0.75  # 75% gain on credit  # 50% gain on credit
STOP_LOSS = 0.5      # 50% loss on credit
MIN_CREDIT = 0.25    # Minimum acceptable credit
SHORT_DELTA_RANGE = (-0.42, -0.38)
LONG_DELTA_RANGE = (-0.22, -0.18)

# === MOCK HISTORICAL DATA ===
data = [
    # timestamp, short_strike, short_bid, short_ask, short_delta, long_strike, long_bid, long_ask, long_delta
    ("2025-04-24 09:35", 440, 1.25, 1.35, -0.40, 435, 0.60, 0.70, -0.20),  # Entry
    ("2025-04-24 09:50", 440, 1.00, 1.10, -0.40, 435, 0.50, 0.60, -0.20),  # Still open
    ("2025-04-24 10:05", 440, 0.60, 0.70, -0.40, 435, 0.30, 0.40, -0.20),  # Hit target profit
    ("2025-04-24 10:35", 440, 2.10, 2.20, -0.40, 435, 1.10, 1.20, -0.20),  # Would've been a stop if still open

    ("2025-04-24 11:00", 445, 1.40, 1.50, -0.41, 440, 0.80, 0.90, -0.19),  # New trade entry
    ("2025-04-24 11:15", 445, 1.60, 1.70, -0.41, 440, 0.95, 1.05, -0.19),  # Rising...
    ("2025-04-24 11:25", 445, 1.90, 2.00, -0.41, 440, 1.10, 1.20, -0.19),  # Hit stop loss
]

# === CONVERT TO DF ===
df = pd.DataFrame(data, columns=[
    "timestamp", "short_strike", "short_bid", "short_ask", "short_delta",
    "long_strike", "long_bid", "long_ask", "long_delta"
])
df["timestamp"] = pd.to_datetime(df["timestamp"])

# === BACKTEST LOGIC ===
trades = []
i = 0
while i < len(df) - 1:
    entry = df.iloc[i]
    entry_credit = (entry.short_bid + entry.short_ask) / 2 - (entry.long_bid + entry.long_ask) / 2

    if (
        SHORT_DELTA_RANGE[0] <= entry.short_delta <= SHORT_DELTA_RANGE[1] and
        LONG_DELTA_RANGE[0] <= entry.long_delta <= LONG_DELTA_RANGE[1] and
        entry_credit >= MIN_CREDIT
    ):
        print(f"\nPlacing spread: Sell {entry.short_strike} / Buy {entry.long_strike} @ Credit: {entry_credit:.2f}")
        target_price = entry_credit * (1 - TARGET_PROFIT)
        stop_price = entry_credit * (1 + STOP_LOSS)

        for j in range(i + 1, len(df)):
            row = df.iloc[j]
            if row.short_strike != entry.short_strike or row.long_strike != entry.long_strike:
                break  # next trade setup found

            spread_price = (row.short_bid + row.short_ask)/2 - (row.long_bid + row.long_ask)/2
            if spread_price <= target_price:
                print(f"[EXIT] ✅ Target hit at {row.timestamp} — Spread: {spread_price:.2f} <= {target_price:.2f}")
                print(f"P&L: +{entry_credit - spread_price:.2f}")
                trades.append((entry.timestamp, row.timestamp, "target", entry_credit - spread_price))
                break
            elif spread_price >= stop_price:
                print(f"[EXIT] ❌ Stop hit at {row.timestamp} — Spread: {spread_price:.2f} >= {stop_price:.2f}")
                print(f"P&L: -{spread_price - entry_credit:.2f}")
                trades.append((entry.timestamp, row.timestamp, "stop", -(spread_price - entry_credit)))
                break

        i = j
    else:
        i += 1

# === SUMMARY ===
print("\n===== SUMMARY =====")
print(f"Total Trades: {len(trades)}")
wins = sum(1 for t in trades if t[2] == "target")
losses = len(trades) - wins
total_pnl = sum(t[3] for t in trades)
print(f"Wins: {wins}  Losses: {losses}  Win Rate: {wins / len(trades) * 100:.1f}%")
print(f"Total P&L: ${total_pnl:.2f}")
