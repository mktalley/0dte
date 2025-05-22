#!/usr/bin/env python3
"""
send_eod_report.py

Generates end-of-day summary report for a given date (default: today) and sends via email.
Metrics: daily P&L, win rate, total trades, average P&L per trade, largest gain, largest loss,
 total credit collected, max drawdown, average time in trade.
Usage: python scripts/send_eod_report.py [YYYY-MM-DD]
"""
import os
import sys
from datetime import datetime, date
import pandas as pd
import numpy as np
import yfinance as yf



from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Add project root to PYTHONPATH so src.main can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.main import send_email


from alpaca.trading.client import TradingClient
# Global for date used in functions

df_date = None

def compute_trade_pnl(df):
    """Compute PnL per trade by fetching closing price for each symbol."""
    # Ensure numeric types
    df[['short_strike', 'long_strike', 'credit']] = df[['short_strike', 'long_strike', 'credit']].astype(float)
    # Determine next day for closing price lookup
    next_day = (pd.to_datetime(df_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    # Fetch closing price per symbol once
    close_prices = {}
    for sym in df['symbol'].unique():
        try:
            close_prices[sym] = yf.download(sym, start=df_date, end=next_day, progress=False)['Close'].iloc[0]
        except Exception as e:
            print(f"Error fetching close for {sym}: {e}")
            close_prices[sym] = 0.0
    # Map close price and compute PnL per row
    df['close_price'] = df['symbol'].map(close_prices)
    # Ensure close_price is numeric
    df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce').fillna(0.0)
    # Compute intrinsic values for short and long strikes using numpy arrays to avoid ambiguous truth-value
    diff_s = df['short_strike'] - df['close_price']
    arr_s = diff_s.to_numpy()
    df['instr_s'] = np.where(arr_s > 0, arr_s, 0.0)
    diff_l = df['long_strike'] - df['close_price']
    arr_l = diff_l.to_numpy()
    df['instr_l'] = np.where(arr_l > 0, arr_l, 0.0)
    df['pnl'] = (df['credit'] - (df['instr_s'] - df['instr_l'])) * 100
    # Clean up intermediate columns
    df.drop(columns=['close_price', 'instr_s', 'instr_l'], inplace=True)
    return df


def compute_metrics(df):
    """Compute summary metrics from per-trade PnL and timestamps."""
    # Convert timestamps and compute time in trade
    df['dt'] = pd.to_datetime(df['timestamp'])
    close_dt = datetime.strptime(f"{df_date} 16:00:00", "%Y-%m-%d %H:%M:%S")
    df['time_in_trade_min'] = (close_dt - df['dt']).dt.total_seconds() / 60.0
    total_trades = len(df)
    wins = int((df['pnl'] > 0).sum())
    losses = int((df['pnl'] < 0).sum())
    win_rate = wins / total_trades * 100 if total_trades else 0
    total_pnl = float(df['pnl'].sum()) if total_trades else 0
    avg_pnl = float(df['pnl'].mean()) if total_trades else 0
    largest_gain = float(df['pnl'].max()) if total_trades else 0
    largest_loss = float(df['pnl'].min()) if total_trades else 0
    total_credit = float((df['credit'] * 100).sum())
    # max drawdown on cumulative PnL
    df_sorted = df.sort_values('dt')
    df_sorted['cum_pnl'] = df_sorted['pnl'].cumsum()
    rolling_max = df_sorted['cum_pnl'].cummax()
    drawdowns = rolling_max - df_sorted['cum_pnl']
    max_drawdown = float(drawdowns.max()) if total_trades else 0
    avg_time = float(df['time_in_trade_min'].mean()) if total_trades else 0
    return {
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'largest_gain': largest_gain,
        'largest_loss': largest_loss,
        'total_credit': total_credit,
        'max_drawdown': max_drawdown,
        'avg_time_in_trade': avg_time
    }
    """Compute summary metrics from per-trade PnL and timestamps."""



def main():
    global df_date
    # parse date arg and set df_date
    if len(sys.argv) > 1:
        df_date = sys.argv[1]
    else:
        df_date = date.today().strftime("%Y-%m-%d")

    # Fetch account balance and realized P&L from Alpaca
    try:
        API_KEY = os.getenv('APCA_API_KEY_ID') or os.getenv('ALPACA_API_KEY')
        API_SECRET = os.getenv('APCA_API_SECRET_KEY') or os.getenv('ALPACA_SECRET_KEY')
        BASE_URL = os.getenv('APCA_API_BASE_URL') or os.getenv('ALPACA_API_BASE_URL')
        client = TradingClient(API_KEY, API_SECRET, paper=True, url_override=BASE_URL)
        acct = client.get_account()
        balance = float(acct.equity)
        last_equity = float(acct.last_equity)
        realized_pnl = balance - last_equity
    except Exception as e:
        print(f"Warning: unable to fetch Alpaca account: {e}")
        balance = 0.0
        realized_pnl = 0.0

    # load and filter closed trades
    exit_cols = ['timestamp','symbol','side','qty','exit_price','pnl','ratio','status']
    df = pd.read_csv('logs/exit_log.csv', names=exit_cols, header=0)
    df = df[df['timestamp'].str.startswith(df_date)].copy()
    if df.empty:
        print(f"No closed trades found for {df_date}")
        return
    # ensure numeric pnl
    df['pnl'] = df['pnl'].astype(float)
    # compute metrics for closed trades
    total_trades = len(df)
    wins = int((df['pnl'] > 0).sum())
    losses = int((df['pnl'] < 0).sum())
    win_rate = wins / total_trades * 100 if total_trades else 0.0
    total_pnl = float(df['pnl'].sum())
    avg_pnl = float(df['pnl'].mean())
    largest_gain = float(df['pnl'].max())
    largest_loss = float(df['pnl'].min())
    m = {
        'total_trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'largest_gain': largest_gain,
        'largest_loss': largest_loss,
    }
    # build report lines for closed trades
    lines = [
        f"End-of-Day Report for {df_date}",
        f"Total Closed Trades: {m['total_trades']}",
        f"Closed Trades P&L: ${m['total_pnl']:.2f}",
        f"Win Rate: {m['win_rate']:.1f}% ({m['wins']}W/{m['losses']}L)",
        f"Average P&L per Trade: ${m['avg_pnl']:.2f}",
        f"Largest Gain: ${m['largest_gain']:.2f}",
        f"Largest Loss: ${m['largest_loss']:.2f}",
        f"Account Balance Change: ${realized_pnl:.2f}",
        f"Account Balance: ${balance:.2f}",
    ]
    body = "
".join(lines)
    subject = f"EOD 0DTE Report {df_date}: Closed P&L ${m['total_pnl']:.2f}, Bal Change ${realized_pnl:.2f}, Balance ${balance:.2f}, Win {m['win_rate']:.1f}%"    # Display report
    print("Subject:", subject)
    print(body)
    # Send the email
    send_email(subject, body)
    print(f"Test email sent for {df_date}")

if __name__ == '__main__':
    main()
