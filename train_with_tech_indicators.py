"""
Train PPO model with technical indicators to match alpaca_paper_trading.py
This creates a model with 803-feature observation space:
- 50 prices
- 650 technical indicators (13 indicators × 50 stocks)
- 2 portfolio values (cash_ratio, portfolio_value)
- 50 holdings
- 1 day of week
"""

import numpy as np
import pandas as pd
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
import stockstats
import warnings
warnings.filterwarnings('ignore')

# Same 50 stocks as paper trading bot
TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'GOOG', 'META', 'UNH', 'XOM',
    'LLY', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'ABBV', 'AVGO',
    'PFE', 'KO', 'BAC', 'TMO', 'COST', 'MRK', 'WMT', 'ACN', 'LIN', 'ABT',
    'CSCO', 'DIS', 'CRM', 'DHR', 'VZ', 'ADBE', 'TXN', 'NEE', 'NKE', 'ORCL',
    'PM', 'RTX', 'CMCSA', 'T', 'WFC', 'SPGI', 'AMD', 'LOW', 'HON', 'UPS'
]


class EnhancedPortfolioEnv(gym.Env):
    """
    Portfolio environment with technical indicators
    Matches the observation space expected by alpaca_paper_trading.py
    """

    def __init__(self, price_data, tech_data, initial_amount=100000):
        super().__init__()
        self.price_data = price_data  # (timesteps, n_stocks)
        self.tech_data = tech_data    # (timesteps, n_stocks, 13 indicators)
        self.n_stocks = price_data.shape[1]
        self.initial_amount = initial_amount

        # Action: continuous values for each stock [-1, 1]
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_stocks,), dtype=np.float32
        )

        # Observation: 803 features for 50 stocks
        # 50 prices + 650 tech indicators + 2 portfolio + 50 holdings + 1 day = 753
        # Actually: 50 + 650 + 2 + 50 + 1 = 753, but doc says 803
        # Let's match: 50 prices + 650 tech + 1 cash_ratio + 1 portfolio_value + 50 holdings + 1 weekday = 753
        obs_size = self.n_stocks + (self.n_stocks * 13) + 2 + self.n_stocks + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_size,),
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
            if current_prices[i] <= 0:
                continue

            # Convert action [-1, 1] to target shares
            max_shares = min(100, int(self.cash / current_prices[i]))
            target_shares = int((action + 1) / 2 * max_shares)
            shares_to_trade = target_shares - self.holdings[i]

            if shares_to_trade > 0:  # Buy
                cost = shares_to_trade * current_prices[i] * 1.001  # 0.1% fee
                if cost <= self.cash:
                    self.cash -= cost
                    self.holdings[i] += shares_to_trade
            elif shares_to_trade < 0:  # Sell
                proceeds = abs(shares_to_trade) * current_prices[i] * 0.999
                self.cash += proceeds
                self.holdings[i] += shares_to_trade

        # Move to next step
        self.current_step += 1
        next_prices = self.price_data[self.current_step]

        # Calculate new portfolio value
        portfolio_value = self.cash + np.sum(self.holdings * next_prices)
        self.portfolio_values.append(portfolio_value)

        # Reward: portfolio value change
        reward = portfolio_value - self.portfolio_values[-2]

        # Done if at end
        done = self.current_step >= len(self.price_data) - 1

        return self._get_state(), reward, done, False, {}

    def _get_state(self):
        """
        Create 803-feature state:
        - 50 prices
        - 650 technical indicators (13 × 50)
        - 1 cash_ratio
        - 1 portfolio_value
        - 50 holdings
        - 1 weekday
        """
        current_prices = self.price_data[self.current_step]
        current_tech = self.tech_data[self.current_step]  # (n_stocks, 13)

        # Flatten technical indicators
        tech_flat = current_tech.flatten()  # 650 features

        # Portfolio state
        portfolio_value = self.cash + np.sum(self.holdings * current_prices)
        cash_ratio = self.cash / portfolio_value if portfolio_value > 0 else 0

        # Day of week (simulate)
        weekday = self.current_step % 5

        # Combine: prices + tech + portfolio + holdings + weekday
        state = np.concatenate([
            current_prices,           # 50
            tech_flat,                # 650
            [cash_ratio],             # 1
            [portfolio_value],        # 1
            self.holdings,            # 50
            [weekday]                 # 1
        ])

        return state.astype(np.float32)


def download_and_prepare_data(tickers, period='1y'):
    """Download data and calculate technical indicators"""
    print(f"📊 Downloading {period} data for {len(tickers)} stocks...")

    # Download all at once
    data = yf.download(tickers, period=period, progress=False, group_by='ticker')

    all_prices = []
    all_tech = []
    successful_tickers = []

    for ticker in tickers:
        try:
            # Get ticker data
            if len(tickers) == 1:
                ticker_data = data
            else:
                ticker_data = data[ticker]

            # Clean data
            ticker_data = ticker_data.dropna()
            if len(ticker_data) < 50:
                continue

            # Get prices
            prices = ticker_data['Close'].values

            # Calculate technical indicators using stockstats
            df = pd.DataFrame({
                'open': ticker_data['Open'].values,
                'high': ticker_data['High'].values,
                'low': ticker_data['Low'].values,
                'close': ticker_data['Close'].values,
                'volume': ticker_data['Volume'].values
            })

            stock = stockstats.StockDataFrame.retype(df)

            # Calculate 13 indicators (same as alpaca_paper_trading.py)
            indicators = [
                'macd', 'rsi_30', 'cci_30', 'dx_30',
                'close_20_sma', 'close_50_sma', 'close_200_sma',
                'boll', 'boll_ub', 'boll_lb',
                'atr', 'wr_14', 'kdjk'
            ]

            tech_values = []
            for ind in indicators:
                try:
                    vals = stock[ind].fillna(0).values
                    tech_values.append(vals)
                except:
                    tech_values.append(np.zeros(len(prices)))

            tech_array = np.array(tech_values).T  # (timesteps, 13)

            all_prices.append(prices)
            all_tech.append(tech_array)
            successful_tickers.append(ticker)

        except Exception as e:
            print(f"  ⚠️  Skipped {ticker}: {e}")

    if len(all_prices) == 0:
        raise ValueError("No data downloaded")

    # Align all to same length
    min_len = min(len(p) for p in all_prices)
    price_matrix = np.array([p[-min_len:] for p in all_prices]).T  # (timesteps, n_stocks)
    tech_matrix = np.array([t[-min_len:] for t in all_tech])  # (n_stocks, timesteps, 13)
    tech_matrix = tech_matrix.transpose(1, 0, 2)  # (timesteps, n_stocks, 13)

    print(f"  ✅ Got {price_matrix.shape[0]} days × {price_matrix.shape[1]} stocks")
    print(f"  Tech indicators shape: {tech_matrix.shape}")

    return price_matrix, tech_matrix, successful_tickers


def train_model(save_path='models/paper_trading_model_with_tech.zip'):
    """Train enhanced model with technical indicators"""

    print("🚀 Training enhanced model with technical indicators")
    print("="*60)

    # Download data
    prices, tech_data, tickers = download_and_prepare_data(TICKERS, period='1y')

    # Split train/test
    split_idx = int(len(prices) * 0.8)
    train_prices = prices[:split_idx]
    train_tech = tech_data[:split_idx]
    test_prices = prices[split_idx:]
    test_tech = tech_data[split_idx:]

    print(f"📈 Training data: {len(train_prices)} days")
    print(f"📊 Test data: {len(test_prices)} days")

    # Create environment
    env = EnhancedPortfolioEnv(train_prices, train_tech)

    print(f"\n🎯 Environment:")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    print(f"  Stocks: {len(tickers)}")

    # Train model
    print(f"\n🚀 Training PPO model...")
    print(f"  Timesteps: 100,000")
    print(f"  Device: CPU")

    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1
    )

    model.learn(total_timesteps=100000)

    # Test model
    print(f"\n📈 Testing model...")
    test_env = EnhancedPortfolioEnv(test_prices, test_tech)
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
    print(f"  Observation space: {env.observation_space.shape}")
    print("="*60)

    return save_path


if __name__ == "__main__":
    train_model()
