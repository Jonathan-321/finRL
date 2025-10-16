"""
Quick local model training for paper trading demo
Trains a simple PPO model on recent data (last 3 months)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import warnings
warnings.filterwarnings('ignore')

# Same 50 stocks as Modal training
TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'GOOG', 'META', 'UNH', 'XOM',
    'LLY', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'ABBV', 'AVGO',
    'PFE', 'KO', 'BAC', 'TMO', 'COST', 'MRK', 'WMT', 'ACN', 'LIN', 'ABT',
    'CSCO', 'DIS', 'CRM', 'DHR', 'VZ', 'ADBE', 'TXN', 'NEE', 'NKE', 'ORCL',
    'PM', 'RTX', 'CMCSA', 'T', 'WFC', 'SPGI', 'AMD', 'LOW', 'HON', 'UPS'
]

class SimplePortfolioEnv(gym.Env):
    """Simplified portfolio environment for quick training"""

    def __init__(self, price_data, initial_amount=100000):
        super().__init__()
        self.price_data = price_data
        self.n_stocks = price_data.shape[1]
        self.initial_amount = initial_amount

        # Action: continuous values for each stock
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_stocks,), dtype=np.float32
        )

        # State: prices + holdings + cash
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_stocks * 2 + 2,),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_amount
        self.holdings = np.zeros(self.n_stocks)
        self.portfolio_values = [self.initial_amount]
        return self._get_state(), {}

    def step(self, actions):
        if self.current_step >= len(self.price_data) - 1:
            return self._get_state(), 0, True, False, {}

        current_prices = self.price_data[self.current_step]

        # Execute trades based on actions
        for i, action in enumerate(actions):
            target_value = self.cash * (action + 1) / 2  # Scale to [0, cash]
            target_shares = int(target_value / current_prices[i]) if current_prices[i] > 0 else 0
            shares_to_trade = target_shares - self.holdings[i]

            if shares_to_trade > 0:  # Buy
                cost = shares_to_trade * current_prices[i] * 1.001  # 0.1% fee
                if cost <= self.cash:
                    self.cash -= cost
                    self.holdings[i] += shares_to_trade
            elif shares_to_trade < 0:  # Sell
                proceeds = abs(shares_to_trade) * current_prices[i] * 0.999  # 0.1% fee
                self.cash += proceeds
                self.holdings[i] += shares_to_trade

        # Move to next step
        self.current_step += 1
        next_prices = self.price_data[self.current_step]

        # Calculate portfolio value
        portfolio_value = self.cash + np.sum(self.holdings * next_prices)
        self.portfolio_values.append(portfolio_value)

        # Reward is change in portfolio value
        reward = portfolio_value - self.portfolio_values[-2]

        done = self.current_step >= len(self.price_data) - 1

        return self._get_state(), reward, done, False, {'portfolio_value': portfolio_value}

    def _get_state(self):
        current_prices = self.price_data[self.current_step]
        portfolio_value = self.cash + np.sum(self.holdings * current_prices)

        state = np.concatenate([
            current_prices / np.max(current_prices),  # Normalized prices
            self.holdings / (np.sum(self.holdings) + 1),  # Normalized holdings
            [self.cash / portfolio_value, portfolio_value / self.initial_amount]
        ])

        return state.astype(np.float32)


def download_data(tickers, period='1y'):
    """Download recent data for all tickers"""
    print(f"📊 Downloading {period} data for {len(tickers)} stocks...")

    # Download all tickers at once
    data = yf.download(tickers, period=period, progress=False, group_by='ticker')

    all_data = []
    successful_tickers = []

    for ticker in tickers:
        try:
            if len(tickers) == 1:
                ticker_data = data['Close']
            else:
                ticker_data = data[ticker]['Close']

            # Drop NaN and check length
            ticker_data = ticker_data.dropna()
            if len(ticker_data) > 50:  # Need at least 50 days
                all_data.append(ticker_data.values)
                successful_tickers.append(ticker)
        except Exception as e:
            print(f"  ⚠️  Skipped {ticker}: {e}")

    if len(all_data) == 0:
        raise ValueError("No data downloaded - market may be closed")

    # Align all series to same length
    min_len = min(len(d) for d in all_data)
    price_matrix = np.array([d[-min_len:] for d in all_data]).T

    print(f"  ✅ Got {price_matrix.shape[0]} days × {price_matrix.shape[1]} stocks")
    print(f"  Stocks: {', '.join(successful_tickers[:10])}...")

    return price_matrix


def train_model(save_path='models/paper_trading_model.zip'):
    """Train a quick model for paper trading"""

    print("🚀 Starting quick local training...")
    print("="*60)

    # Download data (1 year for more robust training)
    prices = download_data(TICKERS, period='1y')

    # Split train/test
    split_idx = int(len(prices) * 0.8)
    train_prices = prices[:split_idx]
    test_prices = prices[split_idx:]

    print(f"📈 Training data: {len(train_prices)} days")
    print(f"📊 Test data: {len(test_prices)} days")

    # Create environment
    env = SimplePortfolioEnv(train_prices)

    print(f"\n🎯 Training PPO model...")
    print(f"  Timesteps: 50,000 (quick demo)")
    print(f"  Device: CPU (local training)")

    # Train model (smaller for speed)
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1
    )

    model.learn(total_timesteps=50000)

    # Test model
    print(f"\n📈 Testing model...")
    test_env = SimplePortfolioEnv(test_prices)
    obs, _ = test_env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)

    final_value = test_env.portfolio_values[-1]
    returns = (final_value / 100000 - 1) * 100

    print(f"\n✅ Training complete!")
    print(f"  Final portfolio value: ${final_value:,.2f}")
    print(f"  Return: {returns:+.2f}%")

    # Save model
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)

    print(f"\n💾 Model saved to: {save_path}")
    print("="*60)

    return save_path


if __name__ == "__main__":
    train_model()
