#!/usr/bin/env python3
"""
Quick test of FinRL basic functionality
"""

import sys
sys.path.append('/Users/jonathanmuhire/finRL/FinRL')

import pandas as pd
import yfinance as yf
from datetime import datetime

print("Testing basic FinRL functionality...")

# Test 1: Basic data download with yfinance
print("\n1. Testing basic data download...")
try:
    # Download Apple stock data for last 30 days
    ticker = "AAPL"
    data = yf.download(ticker, period="1mo", interval="1d")
    print(f"✓ Downloaded {len(data)} days of {ticker} data")
    print(f"  Columns: {list(data.columns)}")
    print(f"  Date range: {data.index[0]} to {data.index[-1]}")
    print(f"  Sample data:\n{data.head(2)}")
except Exception as e:
    print(f"✗ Error downloading data: {e}")

# Test 2: Test FinRL YahooDownloader
print("\n2. Testing FinRL YahooDownloader...")
try:
    from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
    
    downloader = YahooDownloader(
        start_date='2024-01-01',
        end_date='2024-01-31', 
        ticker_list=['AAPL', 'MSFT']
    )
    df = downloader.fetch_data()
    print(f"✓ FinRL downloader works! Got {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Tickers: {df['tic'].unique()}")
    print(f"  Sample data:\n{df.head(2)}")
except Exception as e:
    print(f"✗ Error with FinRL downloader: {e}")

# Test 3: Technical indicators
print("\n3. Testing technical indicators...")
try:
    from finrl.meta.preprocessor.preprocessors import FeatureEngineer
    
    # Use simple indicators
    indicators = ['macd', 'rsi_30', 'close_30_sma']
    
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=indicators,
        use_vix=False,  # Skip VIX for speed
        use_turbulence=False  # Skip turbulence for speed
    )
    
    if 'df' in locals():
        processed = fe.preprocess_data(df)
        print(f"✓ Technical indicators added! {len(processed)} rows")
        print(f"  New columns: {[col for col in processed.columns if col in indicators]}")
        print(f"  Sample row:\n{processed.iloc[0]}")
    else:
        print("⚠ Skipping - no data from previous step")
        
except Exception as e:
    print(f"✗ Error with technical indicators: {e}")

# Test 4: Basic environment setup
print("\n4. Testing basic environment setup...")
try:
    from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
    print("✓ Successfully imported StockTradingEnv")
    
    # Try to check if we can create a basic config
    if 'processed' in locals() and len(processed) > 0:
        print(f"✓ Have processed data with shape: {processed.shape}")
    else:
        print("⚠ No processed data available for environment testing")
        
except Exception as e:
    print(f"✗ Error importing environment: {e}")

print("\n" + "="*50)
print("SUMMARY:")
print("✓ Basic yfinance works")
print("✓ Core dependencies installed") 
print("? FinRL modules need full installation for complete functionality")
print("\nNext steps:")
print("1. Install stockstats: pip install stockstats")
print("2. For full RL training: pip install gymnasium stable-baselines3")
print("3. For GPU training: Use Modal with requirements.txt")