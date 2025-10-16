"""
Train 5-Minute Intraday Model
- Updates every 5 minutes (78 decisions/day vs 1/day)
- Uses intraday features: VWAP, volume imbalance, momentum
- Faster adaptation to market conditions
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import stockstats

# 10 stocks for faster training/testing
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM', 'WMT', 'HD']


def download_intraday_data(tickers, days=30):
    """Download 5-minute intraday data"""
    print(f"📥 Downloading 5-min data for {len(tickers)} stocks ({days} days)...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    all_data = []

    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"  [{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)
            data = yf.download(ticker, start=start_date, end=end_date, interval='5m', progress=False)

            if len(data) > 0:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] for col in data.columns]

                data['tic'] = ticker
                data = data.reset_index()

                # Market hours only (9:30-16:00 ET)
                if 'Datetime' in data.columns:
                    data = data[data['Datetime'].dt.hour.between(9, 16)]

                all_data.append(data)
                print(f"✓ ({len(data)} bars)")
        except Exception as e:
            print(f"✗ ({e})")

    df = pd.concat(all_data, ignore_index=True)
    print(f"✅ Downloaded {len(df):,} 5-min bars\n")

    return df, tickers


def calculate_intraday_features(df):
    """Calculate intraday-specific features"""
    print("📊 Calculating intraday features...")

    result_dfs = []

    for ticker in df['tic'].unique():
        ticker_df = df[df['tic'] == ticker].copy()
        ticker_df = ticker_df.sort_values('Datetime')

        # VWAP (Volume Weighted Average Price)
        ticker_df['vwap'] = (ticker_df['Close'] * ticker_df['Volume']).cumsum() / ticker_df['Volume'].cumsum()

        # Intraday momentum
        ticker_df['returns_5min'] = ticker_df['Close'].pct_change()
        ticker_df['returns_15min'] = ticker_df['Close'].pct_change(periods=3)
        ticker_df['returns_30min'] = ticker_df['Close'].pct_change(periods=6)

        # Volume metrics
        ticker_df['volume_ratio'] = ticker_df['Volume'] / ticker_df['Volume'].rolling(20).mean()

        # Technical indicators (fast-moving for intraday)
        ticker_df.columns = [c.lower() if c != 'tic' else c for c in ticker_df.columns]
        stock = stockstats.StockDataFrame.retype(ticker_df)

        # Fast indicators for 5-min
        indicators = ['macd', 'rsi_14', 'cci_14', 'close_20_sma', 'boll', 'atr']
        for ind in indicators:
            try:
                _ = stock[ind]
            except:
                pass

        ticker_df = pd.DataFrame(stock)
        ticker_df.columns = [c.capitalize() if c not in ['tic', 'datetime'] else c for c in ticker_df.columns]

        result_dfs.append(ticker_df)

    df_with_features = pd.concat(result_dfs, ignore_index=True)
    df_with_features = df_with_features.dropna()

    print(f"✅ Features calculated ({len(df_with_features):,} rows)\n")
    return df_with_features


class IntradayTradingEnv(gym.Env):
    """5-Minute Intraday Trading Environment"""

    def __init__(self, price_data, feature_data, n_stocks, initial_amount=100000):
        super().__init__()

        self.price_data = price_data  # (timesteps, n_stocks)
        self.feature_data = feature_data  # (timesteps, n_stocks, n_features)
        self.n_stocks = n_stocks
        self.n_features = feature_data.shape[2]
        self.initial_amount = initial_amount
        self.current_step = 0
        self.max_steps = len(price_data) - 1

        # Observation: prices + features + portfolio + holdings
        obs_size = n_stocks + (n_stocks * self.n_features) + 2 + n_stocks

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1, high=1,
            shape=(n_stocks,),
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
        prices = self.price_data[self.current_step]
        features = self.feature_data[self.current_step].flatten()

        cash_ratio = self.cash / self.portfolio_value if self.portfolio_value > 0 else 0

        obs = np.concatenate([
            prices,
            features,
            [cash_ratio],
            [self.portfolio_value],
            self.holdings
        ])

        return obs.astype(np.float32)

    def step(self, action):
        prices = self.price_data[self.current_step]
        current_value = self.cash + np.sum(self.holdings * prices)

        # Execute trades
        for i in range(self.n_stocks):
            if prices[i] <= 0:
                continue

            target_value = (action[i] + 1) / 2 * current_value
            target_shares = target_value / prices[i]
            shares_to_trade = target_shares - self.holdings[i]

            if abs(shares_to_trade) < 0.01:
                continue

            if shares_to_trade > 0:  # Buy
                max_shares = self.cash / (prices[i] * 1.001)
                shares_to_trade = min(shares_to_trade, max_shares)
                self.cash -= shares_to_trade * prices[i] * 1.001
                self.holdings[i] += shares_to_trade
            else:  # Sell
                shares_to_trade = max(shares_to_trade, -self.holdings[i])
                self.cash += abs(shares_to_trade) * prices[i] * 0.999
                self.holdings[i] += shares_to_trade

        self.current_step += 1

        new_prices = self.price_data[self.current_step]
        new_value = self.cash + np.sum(self.holdings * new_prices)

        reward = new_value - current_value
        self.portfolio_value = new_value

        done = self.current_step >= self.max_steps - 1
        truncated = False

        return self._get_observation(), reward, done, truncated, {}


def prepare_training_data(df, tickers):
    """Prepare data for training"""
    print("🔧 Preparing training data...")

    # Get unique timestamps
    timestamps = sorted(df['Datetime'].unique())
    n_stocks = len(tickers)

    # Price array
    price_array = np.zeros((len(timestamps), n_stocks))

    # Feature columns
    feature_cols = ['Vwap', 'Returns_5min', 'Returns_15min', 'Returns_30min',
                    'Volume_ratio', 'Macd', 'Rsi_14', 'Cci_14', 'Close_20_sma', 'Boll', 'Atr']

    feature_array = np.zeros((len(timestamps), n_stocks, len(feature_cols)))

    for t, timestamp in enumerate(timestamps):
        day_data = df[df['Datetime'] == timestamp]

        for i, ticker in enumerate(tickers):
            ticker_data = day_data[day_data['tic'] == ticker]

            if len(ticker_data) > 0:
                # Price
                price_array[t, i] = float(ticker_data['Close'].iloc[0])

                # Features
                for j, col in enumerate(feature_cols):
                    if col in ticker_data.columns:
                        val = ticker_data[col].iloc[0]
                        feature_array[t, i, j] = float(val) if not pd.isna(val) else 0.0

    print(f"  Price array: {price_array.shape}")
    print(f"  Feature array: {feature_array.shape}")
    print(f"  Timesteps (5-min bars): {len(timestamps)}")
    print()

    return price_array, feature_array


def train_intraday_model(price_data, feature_data, n_stocks, timesteps=50000):
    """Train PPO model on 5-minute data"""

    print("🏋️  Creating intraday environment...")
    env = IntradayTradingEnv(price_data, feature_data, n_stocks)

    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    print()

    print(f"🚀 Training PPO model ({timesteps:,} timesteps)...")
    print("   This is LOCAL training (CPU) - will take ~10-15 minutes")
    print()

    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1
    )

    model.learn(total_timesteps=timesteps)

    # Save
    model_path = "models/intraday_5min_model.zip"
    model.save(model_path)
    print(f"\n💾 Model saved: {model_path}")

    return model


if __name__ == "__main__":
    print("="*80)
    print("5-MINUTE INTRADAY MODEL TRAINING")
    print("="*80)
    print(f"Stocks: {len(TICKERS)}")
    print(f"Frequency: Every 5 minutes")
    print(f"Decisions per day: 78 (vs 1 for daily)")
    print("="*80)
    print()

    # Download 30 days of 5-min data
    df, successful_tickers = download_intraday_data(TICKERS, days=30)

    # Calculate features
    df = calculate_intraday_features(df)

    # Prepare data
    price_data, feature_data = prepare_training_data(df, successful_tickers)

    # Train
    model = train_intraday_model(price_data, feature_data, len(successful_tickers), timesteps=50000)

    print()
    print("="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print("Next: Test tomorrow morning at market open")
    print("  python3 paper_trading_intraday.py")
