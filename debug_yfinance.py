import yfinance as yf
import pandas as pd

# Test single download to see column structure
data = yf.download("AAPL", start="2024-01-01", end="2024-01-10", progress=False)
print("Raw yfinance data:")
print(data.head())
print("\nColumns:", data.columns)
print("\nAfter reset_index:")
data = data.reset_index() 
print(data.head())
print("\nColumns:", data.columns)