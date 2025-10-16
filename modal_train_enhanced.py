"""
Modal A100 training with enhanced model (500K timesteps)
Uses technical indicators, optimized for production paper trading
"""

import modal

# Create Modal app
app = modal.App("finrl-enhanced-training")

# Define the Modal image with all dependencies
# Skip FinRL since we don't use it and it conflicts with stockstats version
# numpy 1.26.4 for compatibility with local environment
image = (
    modal.Image.debian_slim()
    .pip_install([
        "yfinance>=0.2.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0,<2.0.0",  # Match local environment (1.26.4)
        "gymnasium>=0.28.0",
        "stable-baselines3>=2.0.0",
        "stockstats>=0.5.0,<0.6.0",  # Use version compatible with our code
        "torch>=2.0.0"
    ])
)

# Volume for persistent storage
volume = modal.Volume.from_name("finrl-volume", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100",
    memory=40960,  # 40GB RAM for safety
    timeout=3600,  # 1 hour
    volumes={"/data": volume}
)
def train_enhanced_model():
    """
    Train enhanced PPO model with 500K timesteps on A100
    - 50 S&P 500 stocks
    - 753 features (prices + technical indicators + portfolio state)
    - Optimized hyperparameters
    - Save to volume for download
    """
    import numpy as np
    import pandas as pd
    import yfinance as yf
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    import stockstats
    import warnings
    warnings.filterwarnings('ignore')

    print("🚀 ENHANCED TRAINING ON A100")
    print("="*70)
    print("Configuration:")
    print("  - Timesteps: 500,000")
    print("  - Stocks: 50 S&P 500")
    print("  - Features: 753 (with technical indicators)")
    print("  - GPU: NVIDIA A100")
    print("  - Data: 2 years (2023-2025)")
    print("="*70)

    # 50 S&P 500 stocks
    TICKERS = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'GOOG', 'META', 'UNH', 'XOM',
        'LLY', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'ABBV', 'AVGO',
        'PFE', 'KO', 'BAC', 'TMO', 'COST', 'MRK', 'WMT', 'ACN', 'LIN', 'ABT',
        'CSCO', 'DIS', 'CRM', 'DHR', 'VZ', 'ADBE', 'TXN', 'NEE', 'NKE', 'ORCL',
        'PM', 'RTX', 'CMCSA', 'T', 'WFC', 'SPGI', 'AMD', 'LOW', 'HON', 'UPS'
    ]

    class EnhancedPortfolioEnv(gym.Env):
        """Portfolio environment with technical indicators"""

        def __init__(self, price_data, tech_data, initial_amount=100000):
            super().__init__()
            self.price_data = price_data
            self.tech_data = tech_data
            self.n_stocks = price_data.shape[1]
            self.initial_amount = initial_amount

            self.action_space = spaces.Box(
                low=-1, high=1, shape=(self.n_stocks,), dtype=np.float32
            )

            # 753 features
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

            # Execute trades
            for i, action in enumerate(actions):
                if current_prices[i] <= 0:
                    continue

                max_shares = min(100, int(self.cash / current_prices[i]))
                target_shares = int((action + 1) / 2 * max_shares)
                shares_to_trade = target_shares - self.holdings[i]

                if shares_to_trade > 0:  # Buy
                    cost = shares_to_trade * current_prices[i] * 1.001
                    if cost <= self.cash:
                        self.cash -= cost
                        self.holdings[i] += shares_to_trade
                elif shares_to_trade < 0:  # Sell
                    proceeds = abs(shares_to_trade) * current_prices[i] * 0.999
                    self.cash += proceeds
                    self.holdings[i] += shares_to_trade

            # Next step
            self.current_step += 1
            next_prices = self.price_data[self.current_step]

            portfolio_value = self.cash + np.sum(self.holdings * next_prices)
            self.portfolio_values.append(portfolio_value)

            reward = portfolio_value - self.portfolio_values[-2]
            done = self.current_step >= len(self.price_data) - 1

            return self._get_state(), reward, done, False, {}

        def _get_state(self):
            """Create 753-feature state"""
            current_prices = self.price_data[self.current_step]
            current_tech = self.tech_data[self.current_step]

            tech_flat = current_tech.flatten()
            portfolio_value = self.cash + np.sum(self.holdings * current_prices)
            cash_ratio = self.cash / portfolio_value if portfolio_value > 0 else 0
            weekday = self.current_step % 5

            state = np.concatenate([
                current_prices,
                tech_flat,
                [cash_ratio],
                [portfolio_value],
                self.holdings,
                [weekday]
            ])

            return state.astype(np.float32)

    # Download and prepare data
    print("\n📊 Downloading 2 years of data...")
    data = yf.download(TICKERS, period='2y', progress=False, group_by='ticker')

    all_prices = []
    all_tech = []
    successful_tickers = []

    for ticker in TICKERS:
        try:
            if len(TICKERS) == 1:
                ticker_data = data
            else:
                ticker_data = data[ticker]

            ticker_data = ticker_data.dropna()
            if len(ticker_data) < 100:
                continue

            prices = ticker_data['Close'].values

            # Calculate technical indicators
            df = pd.DataFrame({
                'open': ticker_data['Open'].values,
                'high': ticker_data['High'].values,
                'low': ticker_data['Low'].values,
                'close': ticker_data['Close'].values,
                'volume': ticker_data['Volume'].values
            })

            stock = stockstats.StockDataFrame.retype(df)

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

            tech_array = np.array(tech_values).T

            all_prices.append(prices)
            all_tech.append(tech_array)
            successful_tickers.append(ticker)

        except Exception as e:
            print(f"  ⚠️ Skipped {ticker}: {e}")

    # Align data
    min_len = min(len(p) for p in all_prices)
    price_matrix = np.array([p[-min_len:] for p in all_prices]).T
    tech_matrix = np.array([t[-min_len:] for t in all_tech])
    tech_matrix = tech_matrix.transpose(1, 0, 2)

    print(f"  ✅ {price_matrix.shape[0]} days × {price_matrix.shape[1]} stocks")

    # Split train/test
    split_idx = int(len(price_matrix) * 0.8)
    train_prices = price_matrix[:split_idx]
    train_tech = tech_matrix[:split_idx]
    test_prices = price_matrix[split_idx:]
    test_tech = tech_matrix[split_idx:]

    print(f"\n📈 Training: {len(train_prices)} days")
    print(f"📊 Testing: {len(test_prices)} days")

    # Create environment
    env = EnhancedPortfolioEnv(train_prices, train_tech)

    print(f"\n🎯 Environment:")
    print(f"  Observation: {env.observation_space.shape}")
    print(f"  Actions: {env.action_space.shape}")

    # Train with optimized hyperparameters
    print(f"\n🚀 Training PPO with 500,000 timesteps on A100...")

    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,  # Larger batch size for A100
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=None,  # Disable tensorboard
        device='cuda'
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="/data/checkpoints/",
        name_prefix="finrl_enhanced"
    )

    eval_callback = EvalCallback(
        env,
        eval_freq=10000,
        deterministic=True,
        render=False,
        verbose=1
    )

    # Train
    model.learn(
        total_timesteps=500000,
        callback=[checkpoint_callback, eval_callback]
    )

    # Test
    print("\n📈 Testing on held-out data...")
    test_env = EnhancedPortfolioEnv(test_prices, test_tech)
    obs, _ = test_env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)

    final_value = test_env.portfolio_values[-1]
    returns = (final_value / 100000 - 1) * 100

    # Calculate Sharpe ratio
    portfolio_returns = np.diff(test_env.portfolio_values) / test_env.portfolio_values[:-1]
    sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252) if np.std(portfolio_returns) > 0 else 0

    # Benchmark
    equal_weights = np.ones(len(successful_tickers)) / len(successful_tickers)
    benchmark_return = np.sum(equal_weights * (test_prices[-1] / test_prices[0] - 1)) * 100

    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {returns:+.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Benchmark Return: {benchmark_return:+.2f}%")
    print(f"Excess Return: {returns - benchmark_return:+.2f}%")

    # Save final model
    model.save("/data/finrl_enhanced_500k.zip")
    print(f"\n💾 Model saved to: /data/finrl_enhanced_500k.zip")
    print("="*70)

    return {
        'final_value': final_value,
        'return_pct': returns,
        'sharpe_ratio': sharpe,
        'benchmark_return': benchmark_return
    }


@app.local_entrypoint()
def main():
    """Run training from command line"""
    result = train_enhanced_model.remote()
    print(f"\n✅ Training completed remotely!")
    print(f"Results: {result}")
