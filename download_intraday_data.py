"""
Download 5-minute intraday data for training
This is MUCH more data than daily (78 bars/day vs 1 bar/day)
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pickle

# Start with 10 stocks to test
TEST_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
    'META', 'TSLA', 'JPM', 'WMT', 'HD'
]

def download_intraday_data(tickers, period_months=3):
    """
    Download 5-minute bars
    Note: yfinance limits intraday data to last 60 days for free tier
    """

    print("="*80)
    print("DOWNLOADING 5-MINUTE INTRADAY DATA")
    print("="*80)
    print(f"Tickers: {len(tickers)}")
    print(f"Frequency: 5-minute bars")
    print(f"Market hours: 9:30 AM - 4:00 PM ET (6.5 hours)")
    print(f"Bars per day: 78 (6.5 hours * 12 bars/hour)")
    print("="*80)
    print()

    # yfinance allows max 60 days of 5min data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)

    print(f"📥 Downloading: {start_date.date()} to {end_date.date()}")
    print(f"   Note: Free tier limited to 60 days for 5-min data")
    print()

    all_data = []
    failed = []

    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"  [{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)

            # Download 5-minute data
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval='5m',
                progress=False
            )

            if len(data) > 0:
                # Flatten columns
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] if col[1] == '' else col[0] for col in data.columns]

                data['tic'] = ticker
                data = data.reset_index()

                # Ensure Close is Series
                if 'Close' in data.columns and isinstance(data['Close'], pd.DataFrame):
                    data['Close'] = data['Close'].iloc[:, 0]

                # Filter to market hours only (9:30 AM - 4:00 PM ET)
                data = data[data['Datetime'].dt.hour.between(9, 16)]

                all_data.append(data)
                print(f"✓ ({len(data)} bars)")
            else:
                failed.append(ticker)
                print("✗")

        except Exception as e:
            failed.append(ticker)
            print(f"✗ ({e})")

    if failed:
        print(f"\n⚠️  Failed: {', '.join(failed)}")

    df = pd.concat(all_data, ignore_index=True)

    # Stats
    total_bars = len(df)
    bars_per_ticker = total_bars // len([t for t in tickers if t not in failed])
    trading_days = bars_per_ticker // 78  # Assuming 78 bars/day

    print()
    print("="*80)
    print("DOWNLOAD SUMMARY")
    print("="*80)
    print(f"Total bars: {total_bars:,}")
    print(f"Avg bars per stock: {bars_per_ticker:,}")
    print(f"Estimated trading days: {trading_days}")
    print(f"Data size: {len(df):,} rows")
    print(f"Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
    print("="*80)

    return df, [t for t in tickers if t not in failed]


def save_data(df, filename='intraday_5min_data.pkl'):
    """Save to pickle for fast loading"""
    print(f"\n💾 Saving to {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump(df, f)
    print(f"✅ Saved ({len(df):,} rows)")


if __name__ == "__main__":
    # Download 5-minute data
    df, successful = download_intraday_data(TEST_TICKERS)

    # Save
    save_data(df, 'intraday_5min_10stocks.pkl')

    print("\n✅ Ready for training!")
    print("\nNext steps:")
    print("  1. Create intraday training environment")
    print("  2. Train model with 78 decisions/day")
    print("  3. Backtest on 5-minute data")
