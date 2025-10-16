#!/usr/bin/env python3
"""
Enhanced FinRL Portfolio Trading System
Ready for local testing and Modal scaling
"""

import sys
sys.path.append('/Users/jonathanmuhire/finRL/FinRL')

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
import stockstats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PortfolioTradingEnv(gym.Env):
    """
    Advanced Portfolio Trading Environment with risk management
    """
    
    def __init__(self, price_data, tech_data, initial_amount=100000, 
                 transaction_cost=0.001, risk_penalty=0.1):
        super().__init__()
        
        self.price_data = price_data
        self.tech_data = tech_data
        self.n_stocks = price_data.shape[1]
        self.n_features = tech_data.shape[1]
        self.initial_amount = initial_amount
        self.transaction_cost = transaction_cost
        self.risk_penalty = risk_penalty
        
        # State: [cash, stocks_owned, prices, tech_indicators, portfolio_value, day]
        state_dim = 1 + self.n_stocks + self.n_stocks + self.n_features + 1 + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,))
        
        # Actions: portfolio weights (sum to 1)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_stocks + 1,))  # +1 for cash
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.cash = self.initial_amount
        self.stocks_owned = np.zeros(self.n_stocks)
        self.portfolio_values = [self.initial_amount]
        self.trades = 0
        return self._get_state(), {}
    
    def step(self, action):
        # Normalize actions to ensure they sum to 1
        action = np.clip(action, 0, 1)
        action = action / (action.sum() + 1e-8)
        
        # Current portfolio value
        current_prices = self.price_data[self.current_step]
        current_portfolio_value = self.cash + np.sum(self.stocks_owned * current_prices)
        
        # Calculate target allocations
        target_cash = action[-1] * current_portfolio_value
        target_stock_values = action[:-1] * current_portfolio_value
        target_stock_shares = target_stock_values / (current_prices + 1e-8)
        
        # Execute trades with transaction costs
        trades_made = 0
        for i in range(self.n_stocks):
            share_diff = target_stock_shares[i] - self.stocks_owned[i]
            if abs(share_diff) > 0.01:  # Minimum trade threshold
                trade_value = abs(share_diff * current_prices[i])
                cost = trade_value * self.transaction_cost
                self.cash -= cost
                self.stocks_owned[i] = target_stock_shares[i]
                trades_made += 1
        
        self.trades += trades_made
        
        # Move to next time step
        self.current_step += 1
        if self.current_step >= len(self.price_data) - 1:
            done = True
            next_prices = current_prices  # Use current prices if at end
        else:
            done = False
            next_prices = self.price_data[self.current_step]
        
        # Calculate new portfolio value
        new_portfolio_value = self.cash + np.sum(self.stocks_owned * next_prices)
        self.portfolio_values.append(new_portfolio_value)
        
        # Calculate reward with risk adjustment
        if len(self.portfolio_values) > 1:
            returns = np.array(self.portfolio_values[1:]) / np.array(self.portfolio_values[:-1]) - 1
            avg_return = np.mean(returns)
            volatility = np.std(returns) if len(returns) > 1 else 0
            sharpe_ratio = avg_return / (volatility + 1e-8)
            
            # Reward = return - risk penalty for high volatility
            reward = avg_return - self.risk_penalty * volatility
        else:
            reward = 0
        
        return self._get_state(), reward, done, False, {
            'portfolio_value': new_portfolio_value,
            'cash': self.cash,
            'trades': trades_made,
            'total_trades': self.trades
        }
    
    def _get_state(self):
        if self.current_step >= len(self.price_data):
            self.current_step = len(self.price_data) - 1
            
        current_prices = self.price_data[self.current_step]
        current_tech = self.tech_data[self.current_step]
        portfolio_value = self.cash + np.sum(self.stocks_owned * current_prices)
        
        state = np.concatenate([
            [self.cash / self.initial_amount],  # Normalized cash
            self.stocks_owned / 1000,  # Normalized stock holdings
            current_prices / current_prices.mean(),  # Normalized prices
            current_tech / (np.abs(current_tech).max() + 1e-8),  # Normalized tech indicators
            [portfolio_value / self.initial_amount],  # Normalized portfolio value
            [self.current_step / len(self.price_data)]  # Progress through data
        ])
        
        return state.astype(np.float32)

def download_portfolio_data(tickers, start_date, end_date):
    """Download and process data for multiple stocks"""
    print(f"Downloading data for {len(tickers)} tickers...")
    
    dfs = []
    failed_tickers = []
    
    for i, ticker in enumerate(tickers):
        try:
            print(f"  {i+1}/{len(tickers)}: {ticker}")
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if len(data) > 0:
                data = data.reset_index()
                
                # Handle MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] if col[1] == '' else col[0] for col in data.columns]
                
                data['tic'] = ticker
                data = data.rename(columns={
                    'Date': 'date',
                    'Open': 'open', 
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                if all(col in data.columns for col in ['date', 'open', 'high', 'low', 'close', 'volume']):
                    data = data[['date', 'open', 'high', 'low', 'close', 'volume', 'tic']]
                    dfs.append(data)
                else:
                    failed_tickers.append(ticker)
            else:
                failed_tickers.append(ticker)
                
        except Exception as e:
            print(f"    Failed to download {ticker}: {e}")
            failed_tickers.append(ticker)
    
    if failed_tickers:
        print(f"Failed to download: {failed_tickers}")
    
    if not dfs:
        raise ValueError("No data downloaded successfully")
    
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    
    print(f"✓ Downloaded {len(df)} rows for {len(dfs)} successful tickers")
    return df, [ticker for ticker in tickers if ticker not in failed_tickers]

def add_technical_indicators(df, indicators):
    """Add technical indicators to the dataframe"""
    print("Adding technical indicators...")
    
    processed_dfs = []
    tickers = df['tic'].unique()
    
    for ticker in tickers:
        ticker_data = df[df['tic'] == ticker].copy()
        
        # Convert to stockstats format
        stock_data = ticker_data.set_index('date')[['open', 'high', 'low', 'close', 'volume']]
        stock = stockstats.StockDataFrame.retype(stock_data)
        
        # Add indicators
        for indicator in indicators:
            try:
                if indicator == 'macd':
                    stock['macd']
                elif indicator == 'rsi_30':
                    stock['rsi_30']
                elif indicator == 'sma_20':
                    stock['close_20_sma']
                elif indicator == 'sma_50':
                    stock['close_50_sma']
                elif indicator == 'bb_upper':
                    stock['boll_ub']
                elif indicator == 'bb_lower':
                    stock['boll_lb']
                elif indicator == 'atr_14':
                    stock['atr_14']
            except Exception as e:
                print(f"    Warning: Failed to add {indicator} for {ticker}: {e}")
        
        # Back to dataframe
        result = stock.reset_index()
        result['tic'] = ticker
        processed_dfs.append(result)
    
    processed_df = pd.concat(processed_dfs, ignore_index=True)
    processed_df = processed_df.fillna(method='ffill').fillna(0)
    
    print(f"✓ Added indicators: {indicators}")
    return processed_df

def prepare_environment_data(df, tech_indicators):
    """Convert dataframe to arrays for environment"""
    print("Preparing environment data...")
    
    tickers = sorted(df['tic'].unique())
    dates = sorted(df['date'].unique())
    
    price_array = []
    tech_array = []
    
    for date in dates:
        date_data = df[df['date'] == date]
        
        prices = []
        techs = []
        
        for ticker in tickers:
            ticker_data = date_data[date_data['tic'] == ticker]
            
            if len(ticker_data) > 0:
                prices.append(ticker_data['close'].iloc[0])
                
                tech_values = []
                for indicator in tech_indicators:
                    if indicator in ticker_data.columns:
                        tech_values.append(ticker_data[indicator].iloc[0])
                    else:
                        tech_values.append(0)
                techs.extend(tech_values)
            else:
                prices.append(0)
                techs.extend([0] * len(tech_indicators))
        
        price_array.append(prices)
        tech_array.append(techs)
    
    price_array = np.array(price_array)
    tech_array = np.array(tech_array)
    
    print(f"✓ Created arrays: prices {price_array.shape}, tech {tech_array.shape}")
    return price_array, tech_array, tickers

def train_ensemble_models(env, model_configs, total_timesteps=50000):
    """Train multiple RL models for ensemble trading"""
    print("Training ensemble models...")
    
    models = {}
    
    for name, config in model_configs.items():
        print(f"  Training {name}...")
        
        if name == 'PPO':
            model = PPO('MlpPolicy', env, **config, verbose=0)
        elif name == 'SAC':
            model = SAC('MlpPolicy', env, **config, verbose=0)
        elif name == 'A2C':
            model = A2C('MlpPolicy', env, **config, verbose=0)
        
        model.learn(total_timesteps=total_timesteps)
        models[name] = model
        print(f"    ✓ {name} training complete")
    
    return models

def evaluate_ensemble(models, env, episodes=10):
    """Evaluate ensemble performance"""
    print("Evaluating ensemble performance...")
    
    results = {}
    
    for name, model in models.items():
        episode_returns = []
        
        for episode in range(episodes):
            obs, _ = env.reset()
            done = False
            episode_return = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                episode_return += reward
            
            episode_returns.append(env.portfolio_values[-1])
        
        avg_return = np.mean(episode_returns)
        std_return = np.std(episode_returns)
        
        results[name] = {
            'avg_portfolio_value': avg_return,
            'std_portfolio_value': std_return,
            'sharpe_ratio': (avg_return - env.initial_amount) / (std_return + 1e-8)
        }
        
        print(f"  {name}: Avg Portfolio ${avg_return:,.0f}, Sharpe {results[name]['sharpe_ratio']:.3f}")
    
    return results

def main():
    print("Enhanced FinRL Portfolio Trading System")
    print("="*60)
    
    # Configuration
    config = {
        'tickers': [
            # Tech giants
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS',
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'ABBV',
            # Consumer
            'KO', 'PG', 'WMT', 'HD', 'MCD',
            # Industrial
            'CAT', 'BA', 'GE', 'MMM',
            # Energy
            'XOM', 'CVX'
        ],
        'start_date': '2023-01-01',
        'end_date': '2024-10-01',
        'indicators': ['macd', 'rsi_30', 'sma_20', 'sma_50', 'bb_upper', 'bb_lower'],
        'initial_amount': 100000,
        'train_split': 0.8
    }
    
    # Download data
    df, successful_tickers = download_portfolio_data(
        config['tickers'], 
        config['start_date'], 
        config['end_date']
    )
    
    # Add technical indicators
    df_with_tech = add_technical_indicators(df, config['indicators'])
    
    # Prepare environment data
    price_array, tech_array, tickers = prepare_environment_data(df_with_tech, config['indicators'])
    
    # Split data
    split_point = int(len(price_array) * config['train_split'])
    train_prices = price_array[:split_point]
    train_tech = tech_array[:split_point]
    test_prices = price_array[split_point:]
    test_tech = tech_array[split_point:]
    
    print(f"Train data: {train_prices.shape[0]} days")
    print(f"Test data: {test_prices.shape[0]} days")
    
    # Create environments
    train_env = PortfolioTradingEnv(train_prices, train_tech, config['initial_amount'])
    test_env = PortfolioTradingEnv(test_prices, test_tech, config['initial_amount'])
    
    # Check environment
    check_env(train_env)
    print("✓ Environment validation passed")
    
    # Model configurations
    model_configs = {
        'PPO': {'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64},
        'SAC': {'learning_rate': 3e-4, 'buffer_size': 100000, 'batch_size': 256},
        'A2C': {'learning_rate': 3e-4, 'n_steps': 5}
    }
    
    # Train ensemble
    models = train_ensemble_models(train_env, model_configs, total_timesteps=20000)
    
    # Evaluate on test set
    results = evaluate_ensemble(models, test_env, episodes=5)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE - SUMMARY:")
    print(f"✓ Processed {len(successful_tickers)} stocks")
    print(f"✓ Generated {len(config['indicators'])} technical indicators")
    print(f"✓ Trained {len(models)} RL models")
    print(f"✓ Test period performance:")
    
    best_model = max(results.keys(), key=lambda x: results[x]['sharpe_ratio'])
    print(f"  Best Model: {best_model}")
    print(f"  Sharpe Ratio: {results[best_model]['sharpe_ratio']:.3f}")
    print(f"  Portfolio Value: ${results[best_model]['avg_portfolio_value']:,.0f}")
    
    print("\nNext Steps:")
    print("1. Scale to Modal for full S&P 500 training")
    print("2. Add paper trading with Alpaca API")
    print("3. Implement real-time monitoring dashboard")
    
    return models, results, successful_tickers

if __name__ == "__main__":
    models, results, tickers = main()