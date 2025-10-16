"""
Modal-powered FinRL training for large-scale portfolio optimization
Use your A100 credits for serious RL training
"""

import modal

# Create Modal app
app = modal.App("finrl-portfolio-training")

# Define the Modal image with all dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("git")  # Install git first
    .pip_install([
        "git+https://github.com/AI4Finance-Foundation/FinRL.git",  # Install FinRL first (brings its own dependencies)
        "optuna>=3.3.0",  # For hyperparameter optimization
        "plotly>=5.15.0", # For advanced plotting
        "seaborn>=0.12.0"
    ])
)

# Get volume for data persistence
volume = modal.Volume.from_name("finrl-volume")

@app.function(
    image=image,
    gpu="A100",  # Use your A100 credits!
    memory=32768,  # 32GB RAM
    timeout=3600,  # 1 hour limit
    volumes={"/data": volume}
)
def train_large_portfolio():
    """
    Train on large portfolio with full GPU power
    """
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO, SAC, A2C
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    import stockstats
    import warnings
    warnings.filterwarnings('ignore')
    
    print("🚀 Starting large-scale FinRL training on A100!")
    print("="*60)
    
    # S&P 500 top 50 stocks for serious training
    SP500_TOP50 = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'GOOG', 'META',
        'UNH', 'XOM', 'LLY', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX',
        'ABBV', 'AVGO', 'PFE', 'KO', 'BAC', 'TMO', 'COST', 'MRK', 'WMT',
        'ACN', 'LIN', 'ABT', 'CSCO', 'DIS', 'CRM', 'DHR', 'VZ', 'ADBE',
        'TXN', 'NEE', 'NKE', 'ORCL', 'PM', 'RTX', 'CMCSA', 'T', 'WFC',
        'SPGI', 'AMD', 'LOW', 'HON', 'UPS'
    ]
    
    # Configuration for serious training
    config = {
        'tickers': SP500_TOP50,
        'start_date': '2020-01-01',
        'end_date': '2024-10-01',  # 4+ years of data
        'indicators': [
            'macd', 'rsi_30', 'rsi_14', 'close_20_sma', 'close_50_sma', 
            'close_200_sma', 'boll_ub', 'boll_lb', 'atr_14', 'cci_30',
            'willr_14', 'stoch_k', 'stoch_d'
        ],
        'initial_amount': 1000000,  # $1M portfolio
        'train_timesteps': 500000,  # Serious training time
        'eval_freq': 10000
    }
    
    print(f"Target: {len(config['tickers'])} stocks")
    print(f"Period: {config['start_date']} to {config['end_date']}")
    print(f"Training timesteps: {config['train_timesteps']:,}")
    
    # Enhanced portfolio environment
    class ProPortfolioEnv(gym.Env):
        def __init__(self, price_data, tech_data, volume_data, initial_amount=1000000):
            super().__init__()
            
            self.price_data = price_data
            self.tech_data = tech_data
            self.volume_data = volume_data
            self.n_stocks = price_data.shape[1]
            self.initial_amount = initial_amount
            
            # Enhanced state space
            state_dim = (
                self.n_stocks +  # Current prices
                tech_data.shape[1] +  # Technical indicators
                self.n_stocks +  # Current holdings
                self.n_stocks +  # Previous returns
                3  # Cash ratio, portfolio value, day progress
            )
            
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(state_dim,)
            )
            
            # Action space: portfolio weights (including cash)
            self.action_space = spaces.Box(
                low=0, high=1, shape=(self.n_stocks + 1,)
            )
            
            self.reset()
        
        def reset(self, seed=None):
            super().reset(seed=seed)
            self.current_step = 1  # Start at 1 to calculate returns
            self.cash = self.initial_amount
            self.holdings = np.zeros(self.n_stocks)
            self.portfolio_values = [self.initial_amount]
            self.transaction_costs = 0
            self.num_trades = 0
            
            return self._get_state(), {}
        
        def step(self, action):
            # Normalize actions to sum to 1
            action = np.clip(action, 0, 1)
            action = action / (action.sum() + 1e-8)
            
            current_prices = self.price_data[self.current_step]
            current_portfolio_value = self.cash + np.sum(self.holdings * current_prices)
            
            # Rebalance portfolio
            target_cash_ratio = action[-1]
            target_stock_ratios = action[:-1]
            
            target_cash = target_cash_ratio * current_portfolio_value
            target_stock_values = target_stock_ratios * current_portfolio_value
            target_holdings = target_stock_values / (current_prices + 1e-8)
            
            # Execute trades with transaction costs
            total_trade_value = 0
            for i in range(self.n_stocks):
                trade_amount = abs(target_holdings[i] - self.holdings[i])
                if trade_amount > 0.01:  # Minimum trade threshold
                    trade_value = trade_amount * current_prices[i]
                    total_trade_value += trade_value
                    self.holdings[i] = target_holdings[i]
                    self.num_trades += 1
            
            # Apply transaction costs (0.1% of trade value)
            transaction_cost = total_trade_value * 0.001
            self.cash = target_cash - transaction_cost
            self.transaction_costs += transaction_cost
            
            # Move to next step
            self.current_step += 1
            
            if self.current_step >= len(self.price_data) - 1:
                done = True
                next_prices = current_prices
            else:
                done = False
                next_prices = self.price_data[self.current_step]
            
            # Calculate new portfolio value
            new_portfolio_value = self.cash + np.sum(self.holdings * next_prices)
            self.portfolio_values.append(new_portfolio_value)
            
            # Enhanced reward function
            reward = self._calculate_reward()
            
            info = {
                'portfolio_value': new_portfolio_value,
                'cash_ratio': self.cash / new_portfolio_value,
                'num_trades': self.num_trades,
                'transaction_costs': self.transaction_costs
            }
            
            return self._get_state(), reward, done, False, info
        
        def _calculate_reward(self):
            if len(self.portfolio_values) < 2:
                return 0
            
            # Portfolio returns
            returns = np.array(self.portfolio_values[1:]) / np.array(self.portfolio_values[:-1]) - 1
            
            if len(returns) == 0:
                return 0
            
            # Risk-adjusted return (Sharpe-like)
            avg_return = np.mean(returns)
            volatility = np.std(returns) if len(returns) > 1 else 0.01
            
            # Sharpe ratio with risk penalty
            sharpe = avg_return / (volatility + 1e-8)
            
            # Penalty for excessive trading
            trade_penalty = self.num_trades * 0.0001 if hasattr(self, 'num_trades') else 0
            
            return sharpe - trade_penalty
        
        def _get_state(self):
            if self.current_step >= len(self.price_data):
                self.current_step = len(self.price_data) - 1
            
            current_prices = self.price_data[self.current_step]
            current_tech = self.tech_data[self.current_step]
            current_volumes = self.volume_data[self.current_step]
            
            # Previous returns
            if self.current_step > 0:
                prev_prices = self.price_data[self.current_step - 1]
                returns = current_prices / prev_prices - 1
            else:
                returns = np.zeros(self.n_stocks)
            
            portfolio_value = self.cash + np.sum(self.holdings * current_prices)
            
            state = np.concatenate([
                current_prices / np.mean(current_prices),  # Normalized prices
                current_tech / (np.abs(current_tech).max() + 1e-8),  # Normalized tech
                self.holdings / (np.sum(self.holdings) + 1e-8),  # Normalized holdings
                returns,  # Previous returns
                [
                    self.cash / portfolio_value,  # Cash ratio
                    portfolio_value / self.initial_amount,  # Portfolio performance
                    self.current_step / len(self.price_data)  # Progress
                ]
            ])
            
            return state.astype(np.float32)
    
    # Download and process data
    print("\n📊 Downloading market data...")
    
    def download_stock_data(tickers, start_date, end_date):
        all_data = []
        failed = []
        
        for i, ticker in enumerate(tickers):
            try:
                print(f"  {i+1}/{len(tickers)}: {ticker}")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if len(data) > 100:  # Ensure enough data
                    data = data.reset_index()
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = [col[0] if col[1] == '' else col[0] for col in data.columns]
                    
                    data['tic'] = ticker
                    data = data.rename(columns={
                        'Date': 'date', 'Open': 'open', 'High': 'high',
                        'Low': 'low', 'Close': 'close', 'Volume': 'volume'
                    })
                    
                    if all(col in data.columns for col in ['date', 'open', 'high', 'low', 'close', 'volume']):
                        all_data.append(data[['date', 'open', 'high', 'low', 'close', 'volume', 'tic']])
                    else:
                        failed.append(ticker)
                else:
                    failed.append(ticker)
            except Exception as e:
                print(f"    Failed: {e}")
                failed.append(ticker)
        
        if failed:
            print(f"  Failed tickers: {failed}")
        
        df = pd.concat(all_data, ignore_index=True)
        successful_tickers = [t for t in tickers if t not in failed]
        
        print(f"  ✓ {len(successful_tickers)} stocks, {len(df)} total rows")
        return df, successful_tickers
    
    # Download data
    df, successful_tickers = download_stock_data(
        config['tickers'], config['start_date'], config['end_date']
    )
    
    # Add technical indicators
    print("\n🔧 Adding technical indicators...")
    
    def add_advanced_indicators(df, indicators):
        processed_dfs = []
        
        for ticker in df['tic'].unique():
            ticker_data = df[df['tic'] == ticker].copy().sort_values('date')
            
            stock_data = ticker_data.set_index('date')[['open', 'high', 'low', 'close', 'volume']]
            stock = stockstats.StockDataFrame.retype(stock_data)
            
            # Add all indicators
            for indicator in indicators:
                try:
                    if indicator in ['macd', 'rsi_30', 'rsi_14', 'close_20_sma', 'close_50_sma', 
                                   'close_200_sma', 'boll_ub', 'boll_lb', 'atr_14', 'cci_30',
                                   'willr_14', 'stoch_k', 'stoch_d']:
                        stock[indicator]  # This triggers calculation
                except:
                    pass
            
            result = stock.reset_index()
            result['tic'] = ticker
            processed_dfs.append(result)
        
        processed_df = pd.concat(processed_dfs, ignore_index=True)
        processed_df = processed_df.fillna(method='ffill').fillna(0)
        
        return processed_df
    
    df_with_tech = add_advanced_indicators(df, config['indicators'])
    
    # Prepare arrays for environment
    print("\n🏗️  Preparing environment data...")
    
    def create_environment_arrays(df, tickers, indicators):
        dates = sorted(df['date'].unique())
        
        price_array = []
        tech_array = []
        volume_array = []
        
        for date in dates:
            date_data = df[df['date'] == date]
            
            prices = []
            volumes = []
            tech_features = []
            
            for ticker in tickers:
                ticker_data = date_data[date_data['tic'] == ticker]
                
                if len(ticker_data) > 0:
                    prices.append(ticker_data['close'].iloc[0])
                    volumes.append(ticker_data['volume'].iloc[0])
                    
                    for indicator in indicators:
                        if indicator in ticker_data.columns:
                            tech_features.append(ticker_data[indicator].iloc[0])
                        else:
                            tech_features.append(0)
                else:
                    prices.append(0)
                    volumes.append(0)
                    tech_features.extend([0] * len(indicators))
            
            price_array.append(prices)
            volume_array.append(volumes)
            tech_array.append(tech_features)
        
        return np.array(price_array), np.array(tech_array), np.array(volume_array)
    
    price_array, tech_array, volume_array = create_environment_arrays(
        df_with_tech, successful_tickers, config['indicators']
    )
    
    print(f"  ✓ Arrays created: {price_array.shape}")
    
    # Split data
    split_point = int(len(price_array) * 0.8)
    train_prices = price_array[:split_point]
    train_tech = tech_array[:split_point]
    train_volumes = volume_array[:split_point]
    
    test_prices = price_array[split_point:]
    test_tech = tech_array[split_point:]
    test_volumes = volume_array[split_point:]
    
    print(f"  Train: {len(train_prices)} days")
    print(f"  Test: {len(test_prices)} days")
    
    # Create environment
    env = ProPortfolioEnv(train_prices, train_tech, train_volumes, config['initial_amount'])
    
    print(f"\n🎯 Environment created:")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    print(f"  Portfolio stocks: {len(successful_tickers)}")
    
    # Train advanced models with callbacks
    print(f"\n🚀 Training with {config['train_timesteps']:,} timesteps...")
    
    # Enhanced PPO with optimized hyperparameters
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="/tmp/finrl_tensorboard/",
        device='cuda'  # Use GPU!
    )
    
    # Training callbacks
    eval_callback = EvalCallback(
        env, 
        eval_freq=config['eval_freq'],
        deterministic=True,
        render=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="/tmp/finrl_models/",
        name_prefix="finrl_portfolio"
    )
    
    print("  🔥 Starting GPU training...")
    model.learn(
        total_timesteps=config['train_timesteps'],
        callback=[eval_callback, checkpoint_callback]
    )
    
    # Test the trained model
    print("\n📈 Testing trained model...")
    
    test_env = ProPortfolioEnv(test_prices, test_tech, test_volumes, config['initial_amount'])
    
    obs, _ = test_env.reset()
    done = False
    episode_rewards = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = test_env.step(action)
        episode_rewards.append(reward)
    
    final_value = test_env.portfolio_values[-1]
    total_return = (final_value / config['initial_amount'] - 1) * 100
    
    # Calculate benchmark (equal weight buy and hold)
    equal_weights = np.ones(len(successful_tickers)) / len(successful_tickers)
    benchmark_return = np.sum(equal_weights * (test_prices[-1] / test_prices[0] - 1)) * 100
    benchmark_value = config['initial_amount'] * (1 + benchmark_return/100)
    
    # Results summary
    results = {
        'final_portfolio_value': final_value,
        'total_return_pct': total_return,
        'benchmark_return_pct': benchmark_return,
        'excess_return': total_return - benchmark_return,
        'num_stocks': len(successful_tickers),
        'training_timesteps': config['train_timesteps'],
        'total_trades': test_env.num_trades,
        'transaction_costs': test_env.transaction_costs
    }
    
    print("\n" + "="*60)
    print("🎉 LARGE-SCALE TRAINING COMPLETE!")
    print("="*60)
    print(f"Portfolio Size: ${config['initial_amount']:,} → ${final_value:,.0f}")
    print(f"RL Return: {total_return:+.2f}%")
    print(f"Benchmark: {benchmark_return:+.2f}%") 
    print(f"Excess Return: {total_return - benchmark_return:+.2f}%")
    print(f"Stocks Traded: {len(successful_tickers)}")
    print(f"Total Trades: {test_env.num_trades:,}")
    print(f"Transaction Costs: ${test_env.transaction_costs:,.0f}")
    
    if total_return > benchmark_return:
        print("🏆 RL BEATS BENCHMARK!")
    else:
        print("📊 Benchmark wins this round")
    
    print(f"\nModel saved to: /tmp/finrl_models/")
    print(f"Tensorboard logs: /tmp/finrl_tensorboard/")
    
    return results

@app.function(
    image=image,
    gpu="A100",
    memory=32768,
    timeout=10800,  # 3 hours
    volumes={"/data": volume}
)
def hyperparameter_optimization():
    """
    Use Optuna for hyperparameter optimization on GPU
    Optimizes: learning_rate, n_steps, batch_size, gamma, gae_lambda
    Target: Maximize Sharpe ratio
    """
    import optuna
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    print("🔬 HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    print("Target: Maximize Sharpe Ratio")
    print("Trials: 20")
    print("Expected Duration: 2-3 hours")
    print("="*60)

    # Download and prepare data (same as train_large_portfolio)
    from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
    from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split

    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
        'V', 'WMT', 'JPM', 'MA', 'PG', 'XOM', 'HD', 'CVX', 'LLY', 'ABBV',
        'MRK', 'PEP', 'KO', 'COST', 'AVGO', 'TMO', 'MCD', 'CSCO', 'ACN', 'ABT',
        'NKE', 'DHR', 'VZ', 'ADBE', 'NFLX', 'CRM', 'CMCSA', 'TXN', 'PM', 'NEE',
        'UNP', 'ORCL', 'WFC', 'DIS', 'HON', 'BAC', 'MS', 'RTX', 'BMY', 'AMGN'
    ]

    print(f"Downloading data for {len(tickers)} stocks...")
    df = YahooDownloader(
        start_date='2020-01-01',
        end_date='2024-10-01',
        ticker_list=tickers
    ).fetch_data()

    print("Adding technical indicators...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=['macd', 'rsi_30', 'cci_30', 'dx_30'],
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False
    )

    processed = fe.preprocess_data(df)
    train = data_split(processed, '2020-01-01', '2023-07-01')

    print(f"Training data: {len(train)} rows")

    # Define objective function for Optuna
    def objective(trial):
        """Objective function to maximize Sharpe ratio"""

        # Sample hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        gamma = trial.suggest_float('gamma', 0.95, 0.9999)
        gae_lambda = trial.suggest_float('gae_lambda', 0.9, 0.99)
        clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
        ent_coef = trial.suggest_float('ent_coef', 0.0, 0.1)

        print(f"\nTrial {trial.number + 1}: lr={learning_rate:.2e}, n_steps={n_steps}, batch={batch_size}")

        try:
            # Create environment
            from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

            env_config = {
                'df': train,
                'stock_dim': len(tickers),
                'hmax': 100,
                'initial_amount': 1000000,
                'num_stock_shares': [0] * len(tickers),
                'buy_cost_pct': [0.001] * len(tickers),
                'sell_cost_pct': [0.001] * len(tickers),
                'reward_scaling': 1e-4,
                'state_space': 2 + len(tickers) * 6,  # Adjusted for 4 indicators
                'action_space': len(tickers),
                'tech_indicator_list': ['macd', 'rsi_30', 'cci_30', 'dx_30'],
                'print_verbosity': 0
            }

            env = DummyVecEnv([lambda: StockTradingEnv(**env_config)])

            # Train model with sampled hyperparameters
            model = PPO(
                'MlpPolicy',
                env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=10,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                verbose=0,
                device='cuda'
            )

            # Train for shorter time (100K steps for speed)
            model.learn(total_timesteps=100000)

            # Evaluate on validation set (last 20% of train data)
            val = data_split(processed, '2022-07-01', '2023-07-01')

            env_config['df'] = val
            val_env = DummyVecEnv([lambda: StockTradingEnv(**env_config)])

            obs = val_env.reset()
            returns = []

            for _ in range(len(val)):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = val_env.step(action)

                if 'portfolio_value' in info[0]:
                    returns.append(info[0]['portfolio_value'])

                if done[0]:
                    break

            # Calculate Sharpe ratio
            if len(returns) > 1:
                returns_pct = np.diff(returns) / returns[:-1]
                sharpe = np.mean(returns_pct) / (np.std(returns_pct) + 1e-8) * np.sqrt(252)
            else:
                sharpe = -10  # Penalty for failed trials

            print(f"  → Sharpe: {sharpe:.3f}")

            return sharpe

        except Exception as e:
            print(f"  → Trial failed: {e}")
            return -10  # Penalty for failed trials

    # Run optimization
    study = optuna.create_study(
        direction='maximize',
        study_name='finrl_sharpe_optimization',
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    study.optimize(objective, n_trials=20, show_progress_bar=True)

    # Print results
    print("\n" + "="*60)
    print("🏆 OPTIMIZATION COMPLETE!")
    print("="*60)
    print(f"Best Sharpe Ratio: {study.best_value:.3f}")
    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print("\nTop 5 Trials:")
    trials_df = study.trials_dataframe().sort_values('value', ascending=False).head(5)
    print(trials_df[['number', 'value', 'params_learning_rate', 'params_n_steps', 'params_batch_size']])

    # Save best parameters
    import json
    with open('/tmp/best_hyperparameters.json', 'w') as f:
        json.dump(study.best_params, f, indent=2)

    print("\n✅ Best hyperparameters saved to /tmp/best_hyperparameters.json")
    print("💡 Use these to retrain your model for better performance!")

    return study.best_params

@app.function(image=image, volumes={"/data": volume})
def generate_trading_report(results):
    """
    Generate comprehensive trading performance report
    """
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    
    # Create comprehensive report with visualizations
    print("📊 Generating trading performance report...")
    # ... reporting code
    pass

if __name__ == "__main__":
    # Run locally for testing or deploy to Modal
    print("🚀 Ready to deploy to Modal!")
    print("\nTo run:")
    print("1. modal run modal_finrl_training.py::train_large_portfolio")
    print("2. Watch your A100 credits work their magic!")
    print("3. Get professional-grade results in ~1 hour")