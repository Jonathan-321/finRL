#!/usr/bin/env python3
"""
Quick test version - 5 stocks, fast training
"""

import sys
sys.path.append('/Users/jonathanmuhire/finRL/FinRL')

import pandas as pd
import numpy as np
import yfinance as yf
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
import stockstats
import warnings
warnings.filterwarnings('ignore')

class QuickPortfolioEnv(gym.Env):
    def __init__(self, price_data, tech_data, initial_amount=100000):
        super().__init__()
        
        self.price_data = price_data
        self.tech_data = tech_data
        self.n_stocks = price_data.shape[1]
        self.initial_amount = initial_amount
        
        # Simple state: [prices, tech_indicators, portfolio_weights, portfolio_value]
        state_dim = self.n_stocks + tech_data.shape[1] + (self.n_stocks + 1) + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,))
        
        # Actions: portfolio weights
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_stocks + 1,))
        
        self.reset()
    
    def reset(self, seed=None):
        self.current_step = 0
        self.portfolio_value = self.initial_amount
        self.weights = np.array([0.2] * self.n_stocks + [0.0])  # Equal weights + cash
        self.portfolio_history = [self.initial_amount]
        return self._get_state(), {}
    
    def step(self, action):
        # Normalize actions
        action = np.clip(action, 0, 1)
        action = action / (action.sum() + 1e-8)
        
        # Calculate returns
        if self.current_step > 0:
            price_returns = self.price_data[self.current_step] / self.price_data[self.current_step - 1] - 1
            portfolio_return = np.sum(self.weights[:-1] * price_returns)
            self.portfolio_value *= (1 + portfolio_return)
        
        self.weights = action
        self.current_step += 1
        
        done = self.current_step >= len(self.price_data) - 1
        
        # Simple reward: portfolio return
        if len(self.portfolio_history) > 0:
            reward = (self.portfolio_value / self.portfolio_history[-1]) - 1
        else:
            reward = 0
            
        self.portfolio_history.append(self.portfolio_value)
        
        return self._get_state(), reward, done, False, {
            'portfolio_value': self.portfolio_value
        }
    
    def _get_state(self):
        if self.current_step >= len(self.price_data):
            self.current_step = len(self.price_data) - 1
            
        prices = self.price_data[self.current_step]
        tech = self.tech_data[self.current_step]
        
        # Normalize
        normalized_prices = prices / prices.mean()
        normalized_tech = tech / (np.abs(tech).max() + 1e-8)
        portfolio_value_norm = [self.portfolio_value / self.initial_amount]
        
        state = np.concatenate([
            normalized_prices,
            normalized_tech, 
            self.weights,
            portfolio_value_norm
        ])
        
        return state.astype(np.float32)

def quick_test():
    print("Quick FinRL Portfolio Test")
    print("="*40)
    
    # Small test with 5 tech stocks
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
    start_date = '2024-01-01'
    end_date = '2024-06-01'
    
    print(f"Downloading {len(tickers)} stocks...")
    
    # Download data
    dfs = []
    for ticker in tickers:
        print(f"  {ticker}...")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if len(data) > 0:
            data = data.reset_index()
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] if col[1] == '' else col[0] for col in data.columns]
            data['tic'] = ticker
            data = data.rename(columns={'Date': 'date', 'Close': 'close'})
            dfs.append(data[['date', 'close', 'tic']])
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Create price matrix
    pivot_df = df.pivot(index='date', columns='tic', values='close')
    price_array = pivot_df.values
    
    # Simple tech indicators
    tech_features = []
    for i in range(len(price_array)):
        if i < 5:
            # Not enough history for indicators
            tech_features.append([0] * len(tickers) * 2)  # 2 indicators per stock
        else:
            features = []
            for j in range(len(tickers)):
                # Simple moving average
                sma_5 = np.mean(price_array[i-5:i, j])
                momentum = price_array[i, j] / price_array[i-1, j] - 1 if i > 0 else 0
                features.extend([sma_5, momentum])
            tech_features.append(features)
    
    tech_array = np.array(tech_features)
    
    print(f"✓ Data prepared: {price_array.shape[0]} days, {price_array.shape[1]} stocks")
    
    # Split data
    split = int(len(price_array) * 0.8)
    train_prices = price_array[:split]
    train_tech = tech_array[:split]
    test_prices = price_array[split:]
    test_tech = tech_array[split:]
    
    # Create environment
    train_env = QuickPortfolioEnv(train_prices, train_tech)
    test_env = QuickPortfolioEnv(test_prices, test_tech)
    
    print(f"Train: {len(train_prices)} days, Test: {len(test_prices)} days")
    
    # Train models
    print("\nTraining models...")
    
    # PPO
    print("  PPO...")
    ppo_model = PPO('MlpPolicy', train_env, verbose=0, learning_rate=3e-4)
    ppo_model.learn(total_timesteps=5000)
    
    # SAC  
    print("  SAC...")
    sac_model = SAC('MlpPolicy', train_env, verbose=0, learning_rate=3e-4)
    sac_model.learn(total_timesteps=5000)
    
    # Test models
    print("\nTesting models...")
    
    models = {'PPO': ppo_model, 'SAC': sac_model}
    results = {}
    
    for name, model in models.items():
        obs, _ = test_env.reset()
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = test_env.step(action)
        
        final_value = test_env.portfolio_value
        total_return = (final_value / test_env.initial_amount - 1) * 100
        
        results[name] = {
            'final_value': final_value,
            'return_pct': total_return
        }
        
        print(f"  {name}: ${final_value:,.0f} ({total_return:+.1f}%)")
    
    # Benchmark: Buy and hold equal weights
    equal_weights = np.array([0.2] * len(tickers))
    benchmark_return = np.sum(equal_weights * (test_prices[-1] / test_prices[0] - 1)) * 100
    benchmark_value = test_env.initial_amount * (1 + benchmark_return/100)
    
    print(f"  Benchmark (Equal Weight): ${benchmark_value:,.0f} ({benchmark_return:+.1f}%)")
    
    # Summary
    print("\n" + "="*40)
    print("QUICK TEST COMPLETE")
    
    best_model = max(results.keys(), key=lambda x: results[x]['return_pct'])
    best_return = results[best_model]['return_pct']
    
    if best_return > benchmark_return:
        print(f"✅ RL beats benchmark! {best_model}: {best_return:+.1f}% vs {benchmark_return:+.1f}%")
    else:
        print(f"📊 Benchmark wins: {benchmark_return:+.1f}% vs best RL: {best_return:+.1f}%")
    
    print(f"\nReady for scaling:")
    print(f"- More stocks (S&P 500)")
    print(f"- Longer training (100k+ timesteps)")
    print(f"- Better features (sentiment, options data)")
    print(f"- GPU training with Modal")
    
    return models, results

if __name__ == "__main__":
    models, results = quick_test()