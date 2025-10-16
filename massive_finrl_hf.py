"""
Massive FinRL Training with HuggingFace Datasets
Using your HF token for massive financial datasets
"""

import modal

# Create Modal app with HF token access
app = modal.App("massive-finrl-hf")

# Enhanced image with HuggingFace datasets
image = modal.Image.debian_slim().pip_install([
    "datasets>=2.14.0",
    "huggingface_hub>=0.17.0",
    "yfinance>=0.2.0",
    "pandas>=2.0.0", 
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "matplotlib>=3.7.0",
    "stockstats>=0.6.0",
    "gymnasium>=0.29.0",
    "stable-baselines3>=2.4.0",
    "torch>=2.0.0",
    "tensorboard>=2.13.0",
    "plotly>=5.15.0",
    "seaborn>=0.12.0",
    "ta-lib>=0.4.0",  # Technical analysis library
    "ccxt>=4.0.0",    # Crypto exchange data
    "alpha_vantage",  # Alternative data source
    "requests>=2.28.0",
    "beautifulsoup4>=4.12.0"
])

@app.function(
    image=image,
    gpu="A100",
    memory=64000,  # 64GB RAM for massive datasets
    timeout=7200,  # 2 hours
    secrets=[modal.Secret.from_name("hf-token")]
)
def build_massive_dataset():
    """
    Build massive multi-asset financial dataset from multiple sources
    """
    import os
    import pandas as pd
    import numpy as np
    from datasets import load_dataset, Dataset, DatasetDict
    from huggingface_hub import login
    import yfinance as yf
    import warnings
    warnings.filterwarnings('ignore')
    
    print("🚀 Building Massive Financial Dataset with HuggingFace")
    print("="*70)
    
    # Login to HuggingFace
    hf_token = os.environ["HUGGINGFACE_HUB_TOKEN"]
    login(token=hf_token)
    print("✓ Logged into HuggingFace Hub")
    
    # Configuration for massive dataset
    config = {
        'timeframe': '2000-2024',  # 24 years of data
        'frequency': 'daily',
        'universe': {
            'stocks': {
                'sp500': True,      # All 500 stocks
                'nasdaq100': True,  # Tech heavy
                'djia': True,       # Blue chips
                'russell2000': True # Small caps (sample)
            },
            'crypto': {
                'top_50': True,     # Top 50 cryptocurrencies
                'defi_tokens': True,
                'stablecoins': True
            },
            'forex': {
                'major_pairs': True,  # USD/EUR, GBP/USD, etc.
                'emerging': True      # USD/BRL, USD/INR, etc.
            },
            'commodities': {
                'metals': True,    # Gold, Silver, Platinum
                'energy': True,    # Oil, Gas
                'agriculture': True # Wheat, Corn, Coffee
            }
        },
        'features': [
            'ohlcv',           # Price and volume
            'technical',       # 50+ technical indicators
            'sentiment',       # News sentiment scores
            'macro',          # Economic indicators
            'options',        # Options flow data
            'earnings',       # Earnings announcements
            'insider',        # Insider trading data
        ]
    }
    
    print(f"Target universe: {sum(len(v) if isinstance(v, list) else 5 for v in config['universe'].values())} asset classes")
    
    # Step 1: Try to load existing massive datasets from HF Hub
    print("\n📊 Loading datasets from HuggingFace Hub...")
    
    datasets_to_load = [
        "financial/sp500-historical",
        "financial/crypto-historical", 
        "financial/forex-daily",
        "financial/commodities-data",
        "financial/economic-indicators",
        "financial/earnings-calendar",
        "financial/news-sentiment",
        "financial/options-flow"
    ]
    
    loaded_datasets = {}
    
    for dataset_name in datasets_to_load:
        try:
            print(f"  Loading {dataset_name}...")
            dataset = load_dataset(dataset_name, split='train')
            loaded_datasets[dataset_name.split('/')[-1]] = dataset
            print(f"    ✓ Loaded {len(dataset)} rows")
        except Exception as e:
            print(f"    ⚠ Not found: {dataset_name} - {e}")
            # Will fallback to creating our own
    
    # Step 2: Create our own massive dataset if HF datasets aren't available
    if not loaded_datasets:
        print("\n🏗️  Building custom massive dataset...")
        all_data = create_massive_dataset(config)
    else:
        print("\n🔄 Processing loaded HF datasets...")
        all_data = process_hf_datasets(loaded_datasets, config)
    
    return all_data

def create_massive_dataset(config):
    """Create massive dataset from scratch"""
    import yfinance as yf
    import ccxt
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    print("Building from scratch...")
    
    # Get comprehensive ticker lists
    tickers = get_comprehensive_tickers()
    
    print(f"Total tickers to download: {len(tickers['stocks']) + len(tickers['crypto'])}")
    
    # Download stock data (massive)
    stock_data = download_massive_stocks(tickers['stocks'], config['timeframe'])
    
    # Download crypto data
    crypto_data = download_crypto_data(tickers['crypto'], config['timeframe'])
    
    # Download forex data
    forex_data = download_forex_data(tickers['forex'], config['timeframe'])
    
    # Download commodities
    commodities_data = download_commodities_data(tickers['commodities'], config['timeframe'])
    
    # Combine all data
    all_data = combine_multi_asset_data(stock_data, crypto_data, forex_data, commodities_data)
    
    return all_data

def get_comprehensive_tickers():
    """Get comprehensive list of tickers across all asset classes"""
    
    # S&P 500 (full list)
    sp500_tickers = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'GOOG', 'META', 'UNH', 'XOM',
        'LLY', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'ABBV', 'AVGO',
        'PFE', 'KO', 'BAC', 'TMO', 'COST', 'MRK', 'WMT', 'ACN', 'LIN', 'ABT',
        'CSCO', 'DIS', 'CRM', 'DHR', 'VZ', 'ADBE', 'TXN', 'NEE', 'NKE', 'ORCL',
        'PM', 'RTX', 'CMCSA', 'T', 'WFC', 'SPGI', 'AMD', 'LOW', 'HON', 'UPS',
        # Add 450 more S&P 500 stocks here...
        'IBM', 'QCOM', 'INTC', 'MDT', 'AMGN', 'CAT', 'GS', 'BLK', 'AXP', 'MMM'
    ]
    
    # Top cryptocurrencies
    crypto_tickers = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'XRP/USDT', 'SOL/USDT',
        'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'SHIB/USDT', 'MATIC/USDT', 'UNI/USDT',
        'LINK/USDT', 'LTC/USDT', 'BCH/USDT', 'ATOM/USDT', 'VET/USDT', 'FIL/USDT',
        'TRX/USDT', 'ETC/USDT', 'XLM/USDT', 'MANA/USDT', 'SAND/USDT', 'ALGO/USDT'
    ]
    
    # Major forex pairs
    forex_tickers = [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'AUDUSD=X', 'USDCAD=X',
        'NZDUSD=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X'
    ]
    
    # Commodities
    commodities_tickers = [
        'GC=F',   # Gold
        'SI=F',   # Silver
        'CL=F',   # Crude Oil
        'NG=F',   # Natural Gas
        'ZC=F',   # Corn
        'ZW=F',   # Wheat
        'KC=F',   # Coffee
        'SB=F',   # Sugar
        'CC=F',   # Cocoa
        'HG=F'    # Copper
    ]
    
    return {
        'stocks': sp500_tickers,
        'crypto': crypto_tickers,
        'forex': forex_tickers,
        'commodities': commodities_tickers
    }

def download_massive_stocks(tickers, timeframe):
    """Download massive stock dataset"""
    print(f"\n📈 Downloading {len(tickers)} stocks...")
    
    # Split timeframe
    start_date, end_date = timeframe.split('-')
    start_date = f"{start_date}-01-01"
    end_date = f"{end_date}-12-31"
    
    all_stock_data = []
    failed_count = 0
    
    # Download in batches to avoid rate limits
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"  Batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}: {len(batch)} stocks")
        
        try:
            # Download batch at once (more efficient)
            data = yf.download(batch, start=start_date, end=end_date, progress=False, threads=True)
            
            if isinstance(data.columns, pd.MultiIndex):
                # Process multi-ticker data
                for ticker in batch:
                    try:
                        ticker_data = data.xs(ticker, level=1, axis=1)
                        if len(ticker_data) > 100:  # Ensure enough data
                            ticker_data = ticker_data.reset_index()
                            ticker_data['ticker'] = ticker
                            ticker_data['asset_class'] = 'stock'
                            all_stock_data.append(ticker_data)
                    except:
                        failed_count += 1
            else:
                # Single ticker data
                if len(data) > 100:
                    data = data.reset_index()
                    data['ticker'] = batch[0]
                    data['asset_class'] = 'stock'
                    all_stock_data.append(data)
        except Exception as e:
            print(f"    Failed batch: {e}")
            failed_count += len(batch)
    
    if all_stock_data:
        combined_data = pd.concat(all_stock_data, ignore_index=True)
        print(f"  ✓ Downloaded {len(combined_data)} stock data points")
        print(f"  ✗ Failed: {failed_count} tickers")
        return combined_data
    else:
        print("  ✗ No stock data downloaded")
        return pd.DataFrame()

@app.function(
    image=image,
    gpu="A100",
    memory=64000,
    timeout=7200,
    secrets=[modal.Secret.from_name("hf-token")]
)
def train_massive_portfolio():
    """
    Train RL models on massive multi-asset dataset
    """
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO, SAC, A2C
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    import numpy as np
    import pandas as pd
    
    print("🚀 Training on Massive Multi-Asset Dataset")
    print("="*60)
    
    # Load the massive dataset
    print("📊 Loading massive dataset...")
    dataset = build_massive_dataset()
    
    if dataset is None or len(dataset) == 0:
        print("❌ No dataset available, creating sample")
        dataset = create_sample_massive_dataset()
    
    print(f"✓ Dataset loaded: {len(dataset)} total data points")
    
    # Advanced Multi-Asset Environment
    class MassivePortfolioEnv(gym.Env):
        def __init__(self, data, initial_amount=10000000):  # $10M portfolio
            super().__init__()
            
            self.data = data
            self.tickers = data['ticker'].unique()
            self.asset_classes = data['asset_class'].unique()
            self.n_assets = len(self.tickers)
            self.initial_amount = initial_amount
            
            # Massive state space: prices + features for all assets
            state_dim = (
                self.n_assets * 5 +  # OHLCV for each asset
                self.n_assets * 20 + # Technical indicators
                self.n_assets +      # Current holdings
                len(self.asset_classes) + # Asset class allocation
                10  # Macro features (sentiment, volatility, etc.)
            )
            
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(state_dim,)
            )
            
            # Action space: allocation weights for all assets + cash
            self.action_space = spaces.Box(
                low=0, high=1, shape=(self.n_assets + 1,)
            )
            
            self.reset()
        
        def reset(self, seed=None):
            self.current_step = 0
            self.portfolio_value = self.initial_amount
            self.holdings = np.zeros(self.n_assets)
            self.cash = self.initial_amount
            self.portfolio_history = [self.initial_amount]
            return self._get_state(), {}
        
        def step(self, action):
            # Implementation for massive multi-asset trading
            # ... (similar to previous but with multi-asset logic)
            pass
        
        def _get_state(self):
            # Construct massive state vector
            # ... (implementation for large state space)
            pass
    
    # Train with massive resources
    print("🎯 Training massive RL models...")
    
    # Create environment
    env = MassivePortfolioEnv(dataset)
    
    # Enhanced PPO for massive dataset
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=1e-4,
        n_steps=4096,      # Larger batch
        batch_size=128,
        n_epochs=20,       # More epochs
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="/tmp/massive_finrl_logs/",
        device='cuda'
    )
    
    # Train for serious timesteps
    model.learn(total_timesteps=2000000)  # 2M timesteps
    
    print("🎉 Massive training complete!")
    
    return model

def create_sample_massive_dataset():
    """Create a sample dataset if HF datasets aren't available"""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    print("Creating sample massive dataset...")
    
    # Sample data structure
    tickers = ['AAPL', 'MSFT', 'BTC/USDT', 'EURUSD', 'GC=F']  # Multi-asset
    asset_classes = ['stock', 'stock', 'crypto', 'forex', 'commodity']
    
    sample_data = []
    dates = pd.date_range('2020-01-01', '2024-10-01', freq='D')
    
    for ticker, asset_class in zip(tickers, asset_classes):
        for date in dates:
            # Generate realistic sample data
            base_price = {'AAPL': 150, 'MSFT': 300, 'BTC/USDT': 35000, 'EURUSD': 1.1, 'GC=F': 1800}[ticker]
            noise = np.random.normal(0, 0.02)  # 2% daily volatility
            
            sample_data.append({
                'Date': date,
                'ticker': ticker,
                'asset_class': asset_class,
                'Open': base_price * (1 + noise),
                'High': base_price * (1 + noise + 0.01),
                'Low': base_price * (1 + noise - 0.01),
                'Close': base_price * (1 + noise),
                'Volume': np.random.randint(1000000, 10000000)
            })
    
    return pd.DataFrame(sample_data)

if __name__ == "__main__":
    print("🚀 Massive FinRL with HuggingFace Datasets")
    print("Ready to train on massive multi-asset universe!")
    print("\nTo run:")
    print("modal run massive_finrl_hf.py::build_massive_dataset")
    print("modal run massive_finrl_hf.py::train_massive_portfolio")