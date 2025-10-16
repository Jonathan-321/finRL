#!/usr/bin/env python3
"""
Test data download with timeout to identify issues
"""

import sys
sys.path.append('/Users/jonathanmuhire/finRL/FinRL')

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

print("Testing data download functionality...")
print("=" * 60)

# Test 1: Quick yfinance download
print("\n1. Testing quick yfinance download (5 days)...")
try:
    ticker = yf.Ticker("AAPL")
    hist = ticker.history(period="5d")
    print(f"✓ Downloaded {len(hist)} days of AAPL data")
    print(f"  Columns: {list(hist.columns)}")
    print(f"  Latest close: ${hist['Close'].iloc[-1]:.2f}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Multiple tickers
print("\n2. Testing multiple tickers download...")
try:
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    data = yf.download(tickers, period="5d", progress=False, group_by='ticker')
    print(f"✓ Downloaded data for {len(tickers)} tickers")
    print(f"  Shape: {data.shape}")
    print(f"  Multi-index columns: {isinstance(data.columns, pd.MultiIndex)}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: FinRL YahooDownloader
print("\n3. Testing FinRL YahooDownloader...")
try:
    from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)

    downloader = YahooDownloader(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        ticker_list=['AAPL', 'MSFT']
    )
    df = downloader.fetch_data()

    print(f"✓ FinRL downloader successful!")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Tickers: {df['tic'].unique()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    # Save sample for inspection
    print(f"\n  Sample data:")
    print(df.head(3).to_string())

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test with stockstats
print("\n4. Testing stockstats integration...")
try:
    import stockstats

    # Create sample data
    sample_data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [99, 100, 101, 102, 103],
        'close': [102, 103, 104, 105, 106],
        'volume': [1000000] * 5
    })

    stock = stockstats.StockDataFrame.retype(sample_data)
    stock['macd']  # Trigger MACD calculation

    print(f"✓ Stockstats working!")
    print(f"  Available indicators: macd, rsi, sma, bollinger, etc.")
    print(f"  Columns after MACD: {list(stock.columns)}")

except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "=" * 60)
print("Data download test complete!")
print("\nIf all tests passed, environment is ready for:")
print("  ✓ Data collection")
print("  ✓ Technical indicator calculation")
print("  ✓ FinRL preprocessing")
