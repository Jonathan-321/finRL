#!/usr/bin/env python3
"""
Quick diagnostic to identify setup issues
"""

import sys
print(f"Python version: {sys.version}")
print(f"Python path: {sys.executable}")
print()

# Test imports one by one
print("Testing imports...")
print("-" * 50)

imports_to_test = [
    ("pandas", "import pandas as pd"),
    ("numpy", "import numpy as np"),
    ("yfinance", "import yfinance as yf"),
    ("stockstats", "import stockstats"),
    ("gymnasium", "import gymnasium as gym"),
    ("stable_baselines3", "from stable_baselines3 import PPO"),
    ("datasets", "from datasets import load_dataset"),
    ("matplotlib", "import matplotlib.pyplot as plt"),
]

results = {}
for name, import_stmt in imports_to_test:
    try:
        exec(import_stmt)
        print(f"✓ {name}")
        results[name] = True
    except ImportError as e:
        print(f"✗ {name}: {e}")
        results[name] = False
    except Exception as e:
        print(f"⚠ {name}: {e}")
        results[name] = False

print()
print("-" * 50)
print("Testing FinRL imports...")
sys.path.append('/Users/jonathanmuhire/finRL/FinRL')

finrl_imports = [
    ("YahooDownloader", "from finrl.meta.preprocessor.yahoodownloader import YahooDownloader"),
    ("FeatureEngineer", "from finrl.meta.preprocessor.preprocessors import FeatureEngineer"),
    ("StockTradingEnv", "from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv"),
]

for name, import_stmt in finrl_imports:
    try:
        exec(import_stmt)
        print(f"✓ {name}")
        results[name] = True
    except ImportError as e:
        print(f"✗ {name}: {e}")
        results[name] = False
    except Exception as e:
        print(f"⚠ {name}: {e}")
        results[name] = False

print()
print("=" * 50)
print("DIAGNOSTIC SUMMARY:")
print(f"✓ Passed: {sum(1 for v in results.values() if v)}")
print(f"✗ Failed: {sum(1 for v in results.values() if not v)}")
print()

# Show missing packages
failed = [k for k, v in results.items() if not v]
if failed:
    print("Missing/broken packages:")
    for pkg in failed:
        print(f"  - {pkg}")
else:
    print("✓ All packages available!")

print()
print("Environment ready for FinRL:", all(results.values()))
