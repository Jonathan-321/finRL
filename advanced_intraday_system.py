"""
Advanced 5-Minute Intraday Trading System
Addresses real market microstructure and regime detection
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import stockstats
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class MarketRegimeDetector:
    """Detect market regimes (bull/bear/sideways) for adaptive trading"""

    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods
        self.regimes = {
            'strong_bull': {'threshold': 0.02, 'volatility': (0, 0.15)},
            'bull': {'threshold': 0.005, 'volatility': (0, 0.20)},
            'sideways': {'threshold': 0.0, 'volatility': (0, 0.25)},
            'bear': {'threshold': -0.005, 'volatility': (0.15, 1.0)},
            'crash': {'threshold': -0.02, 'volatility': (0.25, 1.0)}
        }

    def detect_regime(self, returns: np.ndarray, volatility: float) -> str:
        """Detect current market regime based on returns and volatility"""
        mean_return = np.mean(returns[-self.lookback_periods:])

        if mean_return > 0.02 and volatility < 0.15:
            return 'strong_bull'
        elif mean_return > 0.005 and volatility < 0.20:
            return 'bull'
        elif mean_return < -0.02 and volatility > 0.25:
            return 'crash'
        elif mean_return < -0.005 and volatility > 0.15:
            return 'bear'
        else:
            return 'sideways'


class MicrostructureModeler:
    """Model realistic market microstructure"""

    @staticmethod
    def calculate_market_impact(order_size: float, avg_volume: float,
                                volatility: float) -> float:
        """
        Calculate market impact using square-root law
        Impact = spread_cost + temporary_impact + permanent_impact
        """
        participation_rate = abs(order_size) / avg_volume if avg_volume > 0 else 0.1

        # Square-root law for temporary impact
        temporary_impact = 0.1 * np.sqrt(participation_rate) * volatility

        # Permanent impact (smaller)
        permanent_impact = 0.01 * participation_rate * volatility

        # Bid-ask spread cost (baseline)
        spread_cost = 0.0005  # 5 bps

        return spread_cost + temporary_impact + permanent_impact

    @staticmethod
    def calculate_slippage(order_size: float, current_price: float,
                           bid_ask_spread: float = 0.01) -> float:
        """Calculate realistic slippage based on order size"""
        base_slippage = bid_ask_spread / 2
        size_impact = abs(order_size) * 0.0001  # 1bp per 100 shares
        return current_price * (base_slippage + size_impact)


class IntradayFeatureEngine:
    """Advanced feature engineering for intraday trading"""

    @staticmethod
    def calculate_vwap(prices: np.ndarray, volumes: np.ndarray) -> float:
        """Volume Weighted Average Price"""
        if len(prices) == 0 or np.sum(volumes) == 0:
            return np.mean(prices) if len(prices) > 0 else 0
        return np.sum(prices * volumes) / np.sum(volumes)

    @staticmethod
    def calculate_order_flow_imbalance(bid_volume: float, ask_volume: float) -> float:
        """Order flow imbalance - predicts short-term direction"""
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0
        return (bid_volume - ask_volume) / total_volume

    @staticmethod
    def calculate_momentum_features(prices: np.ndarray) -> Dict[str, float]:
        """Multiple timeframe momentum indicators"""
        if len(prices) < 30:
            return {'mom_5': 0, 'mom_15': 0, 'mom_30': 0, 'mom_accel': 0}

        current_price = prices[-1]

        # Different timeframe momentums
        mom_5 = (current_price / prices[-5] - 1) if len(prices) >= 5 else 0
        mom_15 = (current_price / prices[-15] - 1) if len(prices) >= 15 else 0
        mom_30 = (current_price / prices[-30] - 1) if len(prices) >= 30 else 0

        # Momentum acceleration (second derivative)
        if len(prices) >= 10:
            recent_mom = prices[-1] / prices[-5] - 1
            older_mom = prices[-5] / prices[-10] - 1
            mom_accel = recent_mom - older_mom
        else:
            mom_accel = 0

        return {
            'mom_5': mom_5,
            'mom_15': mom_15,
            'mom_30': mom_30,
            'mom_accel': mom_accel
        }

    @staticmethod
    def calculate_microstructure_features(prices: np.ndarray, volumes: np.ndarray,
                                         high: np.ndarray, low: np.ndarray) -> Dict[str, float]:
        """Market microstructure features"""
        if len(prices) < 20:
            return {'spread': 0, 'depth_imbalance': 0, 'trade_intensity': 0}

        # Approximate bid-ask spread from high-low
        spread = np.mean((high[-20:] - low[-20:]) / prices[-20:])

        # Volume imbalance as proxy for depth
        recent_vol = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0
        avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
        depth_imbalance = (recent_vol - avg_vol) / avg_vol if avg_vol > 0 else 0

        # Trade intensity
        trade_intensity = np.sum(volumes[-5:]) / np.sum(volumes[-20:]) if np.sum(volumes[-20:]) > 0 else 0

        return {
            'spread': spread,
            'depth_imbalance': depth_imbalance,
            'trade_intensity': trade_intensity
        }

    @staticmethod
    def calculate_time_features(timestamp: datetime) -> Dict[str, float]:
        """Time-based features for market patterns"""
        hour = timestamp.hour
        minute = timestamp.minute

        # Market session indicators
        is_open = 1 if hour == 9 and minute >= 30 else 0
        is_close = 1 if hour == 15 and minute >= 30 else 0
        is_lunch = 1 if 12 <= hour <= 13 else 0

        # Time until close (normalized)
        minutes_until_close = (16 - hour) * 60 - minute
        time_until_close = minutes_until_close / 390  # Normalize by trading day length

        return {
            'is_open': is_open,
            'is_close': is_close,
            'is_lunch': is_lunch,
            'time_until_close': time_until_close
        }


class AdvancedIntradayEnvironment(gym.Env):
    """
    Sophisticated intraday trading environment with:
    - Market microstructure modeling
    - Regime detection
    - Realistic execution costs
    - Advanced features
    """

    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000,
                 max_position_size: float = 0.1, use_microstructure: bool = True):
        super().__init__()

        self.data = data
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size  # Max 10% per position
        self.use_microstructure = use_microstructure

        # Components
        self.regime_detector = MarketRegimeDetector()
        self.microstructure = MicrostructureModeler()
        self.feature_engine = IntradayFeatureEngine()

        # Get unique tickers and timestamps
        self.tickers = sorted(data['tic'].unique())
        self.timestamps = sorted(data['Datetime'].unique())
        self.n_stocks = len(self.tickers)

        # Precompute features for efficiency
        self._precompute_features()

        # Environment setup
        self.current_step = 0
        self.max_steps = len(self.timestamps) - 1

        # Calculate observation size
        # Per stock: price(1) + momentum(4) + microstructure(3) + technicals(6) = 14
        # Global: portfolio(2) + regime(5) + time(4) = 11
        # Holdings: n_stocks
        obs_dim = self.n_stocks * 14 + 11 + self.n_stocks

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1, high=1,
            shape=(self.n_stocks,),
            dtype=np.float32
        )

        self.reset()

    def _precompute_features(self):
        """Precompute all features for faster training"""
        print("📊 Precomputing advanced features...")

        self.price_data = np.zeros((len(self.timestamps), self.n_stocks))
        self.volume_data = np.zeros((len(self.timestamps), self.n_stocks))
        self.high_data = np.zeros((len(self.timestamps), self.n_stocks))
        self.low_data = np.zeros((len(self.timestamps), self.n_stocks))
        self.feature_data = np.zeros((len(self.timestamps), self.n_stocks, 14))

        for t, timestamp in enumerate(self.timestamps):
            day_data = self.data[self.data['Datetime'] == timestamp]

            for i, ticker in enumerate(self.tickers):
                ticker_data = day_data[day_data['tic'] == ticker]

                if len(ticker_data) > 0:
                    self.price_data[t, i] = ticker_data['Close'].iloc[0]
                    self.volume_data[t, i] = ticker_data['Volume'].iloc[0]
                    self.high_data[t, i] = ticker_data['High'].iloc[0]
                    self.low_data[t, i] = ticker_data['Low'].iloc[0]

                    # Compute features if we have enough history
                    if t >= 30:
                        # Momentum features
                        mom_features = self.feature_engine.calculate_momentum_features(
                            self.price_data[:t+1, i]
                        )

                        # Microstructure features
                        micro_features = self.feature_engine.calculate_microstructure_features(
                            self.price_data[:t+1, i],
                            self.volume_data[:t+1, i],
                            self.high_data[:t+1, i],
                            self.low_data[:t+1, i]
                        )

                        # Technical indicators (simplified for speed)
                        returns = np.diff(self.price_data[:t+1, i]) / self.price_data[:t, i]
                        volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.01

                        # VWAP
                        vwap = self.feature_engine.calculate_vwap(
                            self.price_data[max(0, t-20):t+1, i],
                            self.volume_data[max(0, t-20):t+1, i]
                        )
                        vwap_ratio = self.price_data[t, i] / vwap if vwap > 0 else 1

                        # RSI (simplified)
                        if len(returns) >= 14:
                            gains = returns[-14:]
                            gains[gains < 0] = 0
                            losses = -returns[-14:]
                            losses[losses < 0] = 0
                            avg_gain = np.mean(gains)
                            avg_loss = np.mean(losses)
                            rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))
                        else:
                            rsi = 50

                        # Pack features
                        self.feature_data[t, i, :] = [
                            mom_features['mom_5'],
                            mom_features['mom_15'],
                            mom_features['mom_30'],
                            mom_features['mom_accel'],
                            micro_features['spread'],
                            micro_features['depth_imbalance'],
                            micro_features['trade_intensity'],
                            volatility,
                            vwap_ratio,
                            rsi / 100,
                            np.mean(returns[-5:]) if len(returns) >= 5 else 0,  # Short MA
                            np.mean(returns[-20:]) if len(returns) >= 20 else 0,  # Long MA
                            self.volume_data[t, i] / np.mean(self.volume_data[max(0, t-20):t+1, i]),  # Volume ratio
                            (self.high_data[t, i] - self.low_data[t, i]) / self.price_data[t, i]  # Daily range
                        ]

        print(f"✅ Features computed: {self.feature_data.shape}")

    def reset(self, seed=None):
        super().reset(seed=seed)

        # Start from a random point (but with enough history)
        self.current_step = np.random.randint(50, min(100, self.max_steps // 2))

        # Portfolio state
        self.cash = self.initial_capital
        self.holdings = np.zeros(self.n_stocks)
        self.portfolio_value_history = [self.initial_capital]
        self.trade_count = 0
        self.transaction_costs = 0

        return self._get_observation(), {}

    def _get_observation(self):
        """Build comprehensive observation"""
        obs = []

        # Current prices and features for each stock
        prices = self.price_data[self.current_step]
        features = self.feature_data[self.current_step]

        # Per-stock observations
        for i in range(self.n_stocks):
            obs.extend([
                prices[i] / 1000,  # Normalize price
                *features[i]
            ])

        # Portfolio state
        portfolio_value = self.cash + np.sum(self.holdings * prices)
        cash_ratio = self.cash / portfolio_value if portfolio_value > 0 else 1

        obs.extend([
            cash_ratio,
            portfolio_value / self.initial_capital  # Normalized portfolio value
        ])

        # Market regime (one-hot encoded)
        if self.current_step >= 100:
            returns = np.diff(np.mean(self.price_data[self.current_step-100:self.current_step+1], axis=1))
            volatility = np.std(returns)
            regime = self.regime_detector.detect_regime(returns, volatility)
        else:
            regime = 'sideways'

        regime_encoding = {'strong_bull': 0, 'bull': 1, 'sideways': 2, 'bear': 3, 'crash': 4}
        regime_one_hot = np.zeros(5)
        regime_one_hot[regime_encoding[regime]] = 1
        obs.extend(regime_one_hot)

        # Time features
        timestamp = self.timestamps[self.current_step]
        time_features = self.feature_engine.calculate_time_features(timestamp)
        obs.extend(time_features.values())

        # Current holdings (normalized)
        obs.extend(self.holdings / 1000)  # Normalize by typical holding size

        return np.array(obs, dtype=np.float32)

    def step(self, action):
        """Execute trades with realistic microstructure"""
        prices = self.price_data[self.current_step]
        volumes = self.volume_data[self.current_step]

        # Current portfolio value
        old_portfolio_value = self.cash + np.sum(self.holdings * prices)

        # Execute trades
        executed_trades = 0
        total_impact_cost = 0

        for i in range(self.n_stocks):
            if prices[i] <= 0:
                continue

            # Convert action to target allocation
            target_allocation = (action[i] + 1) / 2  # Scale from [-1,1] to [0,1]
            target_allocation *= self.max_position_size  # Apply position limit

            target_value = target_allocation * old_portfolio_value
            target_shares = target_value / prices[i]

            shares_to_trade = target_shares - self.holdings[i]

            if abs(shares_to_trade) < 0.1:  # Minimum trade size
                continue

            # Calculate execution costs
            if self.use_microstructure:
                avg_volume = np.mean(self.volume_data[max(0, self.current_step-20):self.current_step+1, i])
                volatility = np.std(self.price_data[max(0, self.current_step-20):self.current_step+1, i]) / prices[i]

                market_impact = self.microstructure.calculate_market_impact(
                    shares_to_trade, avg_volume, volatility
                )
                slippage = self.microstructure.calculate_slippage(
                    shares_to_trade, prices[i]
                )
            else:
                market_impact = 0.001  # Fixed 10bps
                slippage = 0

            # Execute trade
            if shares_to_trade > 0:  # Buy
                cost_per_share = prices[i] * (1 + market_impact) + slippage
                max_shares = self.cash / cost_per_share
                shares_to_buy = min(shares_to_trade, max_shares)

                if shares_to_buy > 0:
                    total_cost = shares_to_buy * cost_per_share
                    self.cash -= total_cost
                    self.holdings[i] += shares_to_buy
                    executed_trades += 1
                    total_impact_cost += shares_to_buy * prices[i] * market_impact

            else:  # Sell
                shares_to_sell = min(abs(shares_to_trade), self.holdings[i])

                if shares_to_sell > 0:
                    proceeds_per_share = prices[i] * (1 - market_impact) - slippage
                    total_proceeds = shares_to_sell * proceeds_per_share
                    self.cash += total_proceeds
                    self.holdings[i] -= shares_to_sell
                    executed_trades += 1
                    total_impact_cost += shares_to_sell * prices[i] * market_impact

        # Move to next timestep
        self.current_step += 1

        # Calculate new portfolio value
        new_prices = self.price_data[self.current_step]
        new_portfolio_value = self.cash + np.sum(self.holdings * new_prices)

        # Track metrics
        self.portfolio_value_history.append(new_portfolio_value)
        self.trade_count += executed_trades
        self.transaction_costs += total_impact_cost

        # Calculate reward (risk-adjusted)
        raw_return = (new_portfolio_value - old_portfolio_value) / old_portfolio_value

        # Risk adjustment
        if len(self.portfolio_value_history) >= 20:
            recent_returns = np.diff(self.portfolio_value_history[-20:]) / self.portfolio_value_history[-20:-1]
            volatility = np.std(recent_returns) + 1e-10
            sharpe_component = raw_return / volatility
        else:
            sharpe_component = raw_return

        # Penalize excessive trading
        trade_penalty = -0.0001 * executed_trades

        # Final reward
        reward = sharpe_component * 10000 + trade_penalty

        # Check if done
        done = self.current_step >= self.max_steps - 1

        # Additional stopping conditions
        if new_portfolio_value < self.initial_capital * 0.8:  # Stop if 20% loss
            done = True
            reward -= 1000  # Large penalty for big losses

        info = {
            'portfolio_value': new_portfolio_value,
            'trades': executed_trades,
            'transaction_costs': total_impact_cost,
            'total_return': (new_portfolio_value - self.initial_capital) / self.initial_capital
        }

        return self._get_observation(), reward, done, False, info


def train_advanced_intraday():
    """Train the advanced intraday model"""

    print("="*80)
    print("ADVANCED 5-MINUTE INTRADAY TRAINING")
    print("="*80)

    # Download data for top 20 stocks
    TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM',
               'UNH', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'HD', 'DIS', 'BAC', 'ADBE', 'CRM', 'NFLX']

    print(f"📥 Downloading 5-minute data for {len(TICKERS)} stocks...")

    all_data = []
    for ticker in TICKERS:
        try:
            data = yf.download(ticker, period='60d', interval='5m', progress=False)
            if len(data) > 0:
                data['tic'] = ticker
                data = data.reset_index()
                all_data.append(data)
                print(f"  ✓ {ticker}: {len(data)} bars")
        except:
            print(f"  ✗ {ticker}: Failed")

    df = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ Total data: {len(df):,} 5-minute bars")

    # Create environment
    print("\n🏗️ Creating advanced environment with microstructure...")
    env = AdvancedIntradayEnvironment(df, use_microstructure=True)

    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.shape}")
    print(f"  Timesteps available: {len(env.timestamps)}")

    # Custom callback for monitoring
    class TradingCallback(BaseCallback):
        def __init__(self, verbose=0):
            super(TradingCallback, self).__init__(verbose)
            self.episode_returns = []
            self.episode_lengths = []

        def _on_step(self):
            if self.locals.get('dones')[0]:
                info = self.locals['infos'][0]
                self.episode_returns.append(info['total_return'])
                self.episode_lengths.append(info.get('trades', 0))

                if len(self.episode_returns) % 10 == 0:
                    recent_returns = self.episode_returns[-10:]
                    avg_return = np.mean(recent_returns) * 100
                    print(f"  Episode {len(self.episode_returns)}: Avg Return: {avg_return:.2f}%")

            return True

    # Train model
    print("\n🚀 Training PPO with advanced features...")
    print("   This will take 20-30 minutes on CPU")
    print("   For production: use Modal A100 (5-10 minutes)")
    print()

    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1
    )

    callback = TradingCallback()
    model.learn(total_timesteps=100000, callback=callback)

    # Save model
    model_path = "models/advanced_intraday_5min.zip"
    model.save(model_path)

    print(f"\n💾 Model saved: {model_path}")
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)

    # Quick evaluation
    obs, _ = env.reset()
    total_return = 0
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if done:
            total_return = info['total_return']
            break

    print(f"\nTest run return: {total_return*100:.2f}%")
    print(f"Ready for live testing tomorrow at 9:30 AM ET!")

    return model_path


if __name__ == "__main__":
    train_advanced_intraday()