#!/usr/bin/env python3
"""
get_daily_pnl.py

Fetches the realized P&L for the current trading day from Alpaca by comparing current equity
to the last close equity. Prints the dollar P&L.
"""
import os
import sys
from alpaca.trading.client import TradingClient


def main():
    # Read Alpaca credentials from environment
    API_KEY = os.getenv('APCA_API_KEY_ID') or os.getenv('ALPACA_API_KEY')
    API_SECRET = os.getenv('APCA_API_SECRET_KEY') or os.getenv('ALPACA_SECRET_KEY')
    BASE_URL = os.getenv('APCA_API_BASE_URL') or os.getenv('ALPACA_API_BASE_URL')

    if not API_KEY or not API_SECRET:
        print("Error: Alpaca API key/secret not set in environment.", file=sys.stderr)
        sys.exit(1)

    # Initialize Alpaca trading client
    client = TradingClient(
        API_KEY,
        API_SECRET,
        paper=True,
        url_override=BASE_URL
    )

    # Fetch account data
    account = client.get_account()

    # Compute P&L: current equity minus last close equity
    try:
        current_equity = float(account.equity)
        last_equity    = float(account.last_equity)
    except (AttributeError, ValueError) as e:
        print(f"Error reading equity from account: {e}", file=sys.stderr)
        sys.exit(1)

    daily_pnl = current_equity - last_equity
    print(f"Realized P&L for today: ${daily_pnl:.2f}")


if __name__ == '__main__':
    main()