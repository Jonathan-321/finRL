"""
Production-Grade FinRL Training on Modal
- 100 stocks (expanded from 50)
- 20+ technical indicators (expanded from 13)
- 500K timesteps on A100
- Full production setup
"""

import modal
import os

# Create Modal app
app = modal.App("finrl-production")

# Create volume for persistent storage
volume = modal.Volume.from_name("finrl-volume", create_if_missing=True)

# Enhanced image with all dependencies
# numpy 1.26.4 for compatibility with local environment
image = (
    modal.Image.debian_slim()
    .pip_install([
        "yfinance>=0.2.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0,<2.0.0",  # Match local environment (1.26.4)
        "gymnasium>=0.28.0",
        "stable-baselines3>=2.0.0",
        "stockstats>=0.5.0,<0.6.0",
        "torch>=2.0.0",
        "ta>=0.11.0",  # Additional technical analysis library
    ])
)

# 100 S&P 500 stocks for production diversification
PRODUCTION_TICKERS = [
    # Tech (20 stocks)
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO', 'ORCL',
    'ADBE', 'CRM', 'CSCO', 'ACN', 'AMD', 'INTC', 'TXN', 'QCOM', 'IBM', 'INTU',

    # Financials (15 stocks)
    'JPM', 'BAC', 'WFC', 'MS', 'GS', 'BLK', 'C', 'SCHW', 'USB', 'AXP',
    'PNC', 'TFC', 'COF', 'BK', 'STT',

    # Healthcare (15 stocks)
    'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK', 'DHR', 'BMY',
    'AMGN', 'GILD', 'CVS', 'CI', 'ELV',

    # Consumer (15 stocks)
    'WMT', 'HD', 'PG', 'COST', 'KO', 'PEP', 'NKE', 'MCD', 'DIS', 'SBUX',
    'LOW', 'TGT', 'TJX', 'BKNG', 'CMG',

    # Energy (10 stocks)
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'PSX', 'VLO', 'OXY',

    # Industrials (10 stocks)
    'BA', 'HON', 'UPS', 'RTX', 'LMT', 'CAT', 'DE', 'GE', 'MMM', 'UNP',

    # Utilities & Real Estate (8 stocks)
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'PLD', 'AMT', 'CCI',

    # Materials & Other (7 stocks)
    'LIN', 'APD', 'SHW', 'NEM', 'FCX', 'ECL', 'DD'
]


@app.function(
    image=image,
    gpu="A100",  # NVIDIA A100 40GB
    timeout=7200,  # 2 hours max
    volumes={"/data": volume},
)
def train_production_model():
    """Train production model with 100 stocks and expanded indicators"""
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from datetime import datetime, timedelta
    import stockstats

    print("=" * 80)
    print("PRODUCTION FINRL TRAINING - 100 STOCKS")
    print("=" * 80)
    print(f"Stocks: {len(PRODUCTION_TICKERS)}")
    print(f"Training steps: 500,000")
    print(f"GPU: A100 40GB")
    print("=" * 80)

    # Download 2 years of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    print(f"\n📥 Downloading data: {start_date.date()} to {end_date.date()}")

    all_data = []
    failed_tickers = []

    for i, ticker in enumerate(PRODUCTION_TICKERS, 1):
        try:
            print(f"  [{i}/{len(PRODUCTION_TICKERS)}] {ticker}...", end=" ")
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if len(data) > 0:
                # Flatten columns
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] if col[1] == '' else col[0] for col in data.columns]

                data['tic'] = ticker
                data = data.reset_index()

                # Ensure Close is Series
                if 'Close' in data.columns and isinstance(data['Close'], pd.DataFrame):
                    data['Close'] = data['Close'].iloc[:, 0]

                all_data.append(data)
                print("✓")
            else:
                print("✗ (no data)")
                failed_tickers.append(ticker)
        except Exception as e:
            print(f"✗ ({e})")
            failed_tickers.append(ticker)

    if failed_tickers:
        print(f"\n⚠️  Failed tickers ({len(failed_tickers)}): {', '.join(failed_tickers)}")

    df = pd.concat(all_data, ignore_index=True)
    successful_tickers = [t for t in PRODUCTION_TICKERS if t not in failed_tickers]
    n_stocks = len(successful_tickers)

    print(f"\n✅ Downloaded {len(df):,} rows for {n_stocks} stocks")

    # Calculate expanded technical indicators
    print("\n📊 Calculating technical indicators...")

    tech_dfs = []
    for ticker in successful_tickers:
        ticker_df = df[df['tic'] == ticker].copy()

        # Save Date column before processing
        date_col = ticker_df['Date'].copy()

        # Lowercase for stockstats (except Date and tic)
        ticker_df.columns = [c.lower() if c not in ['tic', 'Date'] else c for c in ticker_df.columns]
        stock = stockstats.StockDataFrame.retype(ticker_df)

        # Expanded indicator set (20 indicators)
        indicators = [
            # Momentum
            'macd', 'rsi_6', 'rsi_12', 'rsi_30', 'cci_14', 'cci_30', 'dx_14', 'dx_30',

            # Trend
            'close_5_sma', 'close_10_sma', 'close_20_sma', 'close_50_sma',
            'close_100_sma', 'close_200_sma',

            # Volatility
            'boll', 'boll_ub', 'boll_lb', 'atr',

            # Volume
            'wr_14', 'kdjk'
        ]

        for indicator in indicators:
            try:
                _ = stock[indicator]
            except:
                pass

        # Convert to DataFrame and restore Date column
        ticker_df = pd.DataFrame(stock)
        ticker_df['Date'] = date_col.values
        ticker_df.columns = [c.capitalize() if c not in ['tic', 'Date'] else c for c in ticker_df.columns]
        tech_dfs.append(ticker_df)

    df_with_tech = pd.concat(tech_dfs, ignore_index=True)

    # Drop NaN rows from indicator calculation
    df_with_tech = df_with_tech.dropna()

    print(f"✅ Technical indicators calculated ({len(df_with_tech):,} rows after dropna)")

    # Prepare price and tech arrays
    print("\n🔧 Preparing training data...")

    dates = sorted(df_with_tech['Date'].unique())
    price_array = np.zeros((len(dates), n_stocks))

    # 20 indicators per stock
    tech_cols = [
        'Macd', 'Rsi_6', 'Rsi_12', 'Rsi_30', 'Cci_14', 'Cci_30', 'Dx_14', 'Dx_30',
        'Close_5_sma', 'Close_10_sma', 'Close_20_sma', 'Close_50_sma',
        'Close_100_sma', 'Close_200_sma',
        'Boll', 'Boll_ub', 'Boll_lb', 'Atr',
        'Wr_14', 'Kdjk'
    ]

    tech_array = np.zeros((len(dates), n_stocks, len(tech_cols)))

    for t, date in enumerate(dates):
        day_data = df_with_tech[df_with_tech['Date'] == date]
        for i, ticker in enumerate(successful_tickers):
            ticker_data = day_data[day_data['tic'] == ticker]
            if len(ticker_data) > 0:
                # Price
                price_val = ticker_data['Close'].iloc[0]
                if isinstance(price_val, (list, tuple, np.ndarray)):
                    price_val = price_val[0]
                price_array[t, i] = float(price_val)

                # Tech indicators
                for j, col in enumerate(tech_cols):
                    if col in ticker_data.columns:
                        val = ticker_data[col].iloc[0]
                        tech_array[t, i, j] = float(val) if not pd.isna(val) else 0.0

    print(f"✅ Data prepared:")
    print(f"   Price array: {price_array.shape}")
    print(f"   Tech array: {tech_array.shape}")

    # Create production environment
    class ProductionEnv(gym.Env):
        """Production environment with 100 stocks and 20 indicators"""

        def __init__(self, price_data, tech_data, initial_amount=100000):
            super().__init__()

            self.price_data = price_data
            self.tech_data = tech_data
            self.n_stocks = price_data.shape[1]
            self.n_indicators = tech_data.shape[2]
            self.initial_amount = initial_amount
            self.current_step = 0
            self.max_steps = len(price_data) - 1

            # Observation: prices + tech + portfolio + holdings + weekday
            # 100 + (100*20) + 2 + 100 + 1 = 2203 features
            obs_size = self.n_stocks + (self.n_stocks * self.n_indicators) + 2 + self.n_stocks + 1

            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(obs_size,),
                dtype=np.float32
            )

            # Action: continuous portfolio weights [-1, +1] for each stock
            self.action_space = spaces.Box(
                low=-1, high=1,
                shape=(self.n_stocks,),
                dtype=np.float32
            )

            self.reset()

        def reset(self, seed=None):
            super().reset(seed=seed)
            self.current_step = 0
            self.cash = self.initial_amount
            self.holdings = np.zeros(self.n_stocks)
            self.portfolio_value = self.initial_amount
            return self._get_observation(), {}

        def _get_observation(self):
            """Build 2203-feature observation"""
            prices = self.price_data[self.current_step]
            tech = self.tech_data[self.current_step].flatten()

            cash_ratio = self.cash / self.portfolio_value if self.portfolio_value > 0 else 0
            weekday = self.current_step % 7  # Simplified weekday

            obs = np.concatenate([
                prices,                    # 100
                tech,                      # 2000 (100 * 20)
                [cash_ratio],              # 1
                [self.portfolio_value],    # 1
                self.holdings,             # 100
                [weekday]                  # 1
            ])

            return obs.astype(np.float32)

        def step(self, action):
            # Current portfolio value
            prices = self.price_data[self.current_step]
            current_value = self.cash + np.sum(self.holdings * prices)

            # Execute trades based on actions
            for i in range(self.n_stocks):
                if prices[i] <= 0:
                    continue

                # Convert action to target shares
                target_value = (action[i] + 1) / 2 * current_value  # Scale [-1,1] to [0, value]
                target_shares = target_value / prices[i]

                # Calculate trade
                shares_to_trade = target_shares - self.holdings[i]

                if abs(shares_to_trade) < 0.01:
                    continue

                # Transaction cost (0.1%)
                cost = abs(shares_to_trade * prices[i] * 0.001)

                if shares_to_trade > 0:  # Buy
                    max_shares = self.cash / (prices[i] * 1.001)
                    shares_to_trade = min(shares_to_trade, max_shares)
                    self.cash -= shares_to_trade * prices[i] * 1.001
                    self.holdings[i] += shares_to_trade
                else:  # Sell
                    shares_to_trade = max(shares_to_trade, -self.holdings[i])
                    self.cash += abs(shares_to_trade) * prices[i] * 0.999
                    self.holdings[i] += shares_to_trade

            # Move to next step
            self.current_step += 1

            # Calculate new portfolio value
            new_prices = self.price_data[self.current_step]
            new_value = self.cash + np.sum(self.holdings * new_prices)

            # Reward: portfolio value change
            reward = new_value - current_value
            self.portfolio_value = new_value

            # Done if reached end
            done = self.current_step >= self.max_steps - 1
            truncated = False

            return self._get_observation(), reward, done, truncated, {}

    # Split data (80% train, 20% test)
    split_idx = int(len(price_array) * 0.8)
    train_price = price_array[:split_idx]
    train_tech = tech_array[:split_idx]

    print(f"\n🏋️  Creating environment...")
    print(f"   Observation space: 2203 features")
    print(f"   Action space: {n_stocks} continuous")
    print(f"   Training days: {len(train_price)}")

    env = ProductionEnv(train_price, train_tech)

    # Train model
    print(f"\n🚀 Starting PPO training (500K timesteps)...")

    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=None,
        device='cuda'
    )

    model.learn(total_timesteps=500_000)

    # Save model
    model_path = "/data/finrl_production_100stocks_500k.zip"
    model.save(model_path)
    print(f"\n💾 Model saved: {model_path}")

    # Test model
    print(f"\n🧪 Testing on validation set...")
    test_price = price_array[split_idx:]
    test_tech = tech_array[split_idx:]
    test_env = ProductionEnv(test_price, test_tech)

    obs, _ = test_env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = test_env.step(action)

    final_value = test_env.portfolio_value
    returns = (final_value - 100000) / 100000 * 100

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Stocks: {n_stocks}")
    print(f"Features: 2203")
    print(f"Test Return: {returns:+.2f}%")
    print(f"Final Portfolio: ${final_value:,.2f}")
    print(f"Model: {model_path}")
    print(f"{'='*80}")

    return {
        'n_stocks': n_stocks,
        'features': 2203,
        'test_return': returns,
        'final_value': final_value,
        'model_path': model_path,
        'successful_tickers': successful_tickers
    }


@app.local_entrypoint()
def main():
    """Launch production training"""
    result = train_production_model.remote()
    print("\n✅ Production training complete!")
    print(f"   Stocks: {result['n_stocks']}")
    print(f"   Test return: {result['test_return']:+.2f}%")
    print(f"\nDownload model:")
    print(f"   modal volume get finrl-volume finrl_production_100stocks_500k.zip models/")
