"""
Production 5-Minute Intraday Paper Trading Bot
Trades every 5 minutes with advanced market microstructure modeling
"""

import os
import time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from alpaca_trade_api import REST
import pickle
import warnings
warnings.filterwarnings('ignore')

# Alpaca credentials
API_KEY = os.environ.get('ALPACA_API_KEY', 'your_api_key')
SECRET_KEY = os.environ.get('ALPACA_SECRET_KEY', 'your_secret_key')
BASE_URL = 'https://paper-api.alpaca.markets'

# Trading configuration
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'JPM',
    'UNH', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'HD', 'DIS', 'BAC', 'ADBE',
    'CRM', 'NFLX', 'INTC', 'ORCL', 'AMD', 'IBM', 'CSCO', 'PFE', 'TMO',
    'ABT', 'CVX', 'LLY'
]

MODEL_PATH = 'models/advanced_intraday_5min.zip'  # Or use Modal-trained model
MAX_POSITION_SIZE = 0.1  # Max 10% per position
TRADE_INTERVAL = 300  # 5 minutes in seconds


class IntradayFeatureCalculator:
    """Calculate real-time features for intraday trading"""

    def __init__(self):
        self.price_history = {}
        self.volume_history = {}
        self.feature_cache = {}

    def update_history(self, ticker, price, volume, high, low):
        """Update price and volume history"""
        if ticker not in self.price_history:
            self.price_history[ticker] = []
            self.volume_history[ticker] = []

        self.price_history[ticker].append({
            'price': price,
            'high': high,
            'low': low,
            'timestamp': datetime.now()
        })
        self.volume_history[ticker].append(volume)

        # Keep only last 100 data points
        if len(self.price_history[ticker]) > 100:
            self.price_history[ticker].pop(0)
            self.volume_history[ticker].pop(0)

    def calculate_features(self, ticker):
        """Calculate all features for a ticker"""
        if ticker not in self.price_history or len(self.price_history[ticker]) < 5:
            return np.zeros(14)  # Return zeros if insufficient history

        prices = [p['price'] for p in self.price_history[ticker]]
        highs = [p['high'] for p in self.price_history[ticker]]
        lows = [p['low'] for p in self.price_history[ticker]]
        volumes = self.volume_history[ticker]

        features = []

        # Momentum features
        current_price = prices[-1]
        mom_5 = (current_price / prices[-5] - 1) if len(prices) >= 5 else 0
        mom_15 = (current_price / prices[-15] - 1) if len(prices) >= 15 else 0
        mom_30 = (current_price / prices[-30] - 1) if len(prices) >= 30 else 0

        # Momentum acceleration
        if len(prices) >= 10:
            recent_mom = prices[-1] / prices[-5] - 1
            older_mom = prices[-5] / prices[-10] - 1
            mom_accel = recent_mom - older_mom
        else:
            mom_accel = 0

        features.extend([mom_5, mom_15, mom_30, mom_accel])

        # Microstructure features
        if len(prices) >= 20:
            spread = np.mean([(h - l) / p for h, l, p in
                            zip(highs[-20:], lows[-20:], prices[-20:])])

            recent_vol = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0
            avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else 0
            depth_imbalance = (recent_vol - avg_vol) / avg_vol if avg_vol > 0 else 0

            trade_intensity = np.sum(volumes[-5:]) / np.sum(volumes[-20:]) if np.sum(volumes[-20:]) > 0 else 0
        else:
            spread = 0.001
            depth_imbalance = 0
            trade_intensity = 1

        features.extend([spread, depth_imbalance, trade_intensity])

        # Technical features
        returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else [0]
        volatility = np.std(returns[-20:]) if len(returns) >= 20 else 0.01

        # VWAP
        if len(prices) >= 20 and sum(volumes[-20:]) > 0:
            vwap = np.sum(np.array(prices[-20:]) * np.array(volumes[-20:])) / np.sum(volumes[-20:])
            vwap_ratio = current_price / vwap
        else:
            vwap_ratio = 1

        # RSI
        if len(returns) >= 14:
            gains = [r if r > 0 else 0 for r in returns[-14:]]
            losses = [-r if r < 0 else 0 for r in returns[-14:]]
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10))) if avg_loss > 0 else 50
        else:
            rsi = 50

        features.extend([
            volatility,
            vwap_ratio,
            rsi / 100,
            np.mean(returns[-5:]) if len(returns) >= 5 else 0,
            np.mean(returns[-20:]) if len(returns) >= 20 else 0,
            volumes[-1] / np.mean(volumes[-20:]) if len(volumes) >= 20 and np.mean(volumes[-20:]) > 0 else 1,
            (highs[-1] - lows[-1]) / prices[-1] if prices[-1] > 0 else 0
        ])

        return np.array(features, dtype=np.float32)


class MarketRegimeDetector:
    """Detect current market regime"""

    def __init__(self):
        self.market_returns = []

    def update(self, market_prices):
        """Update with latest market prices"""
        avg_price = np.mean(market_prices)
        self.market_returns.append(avg_price)

        # Keep last 100 observations
        if len(self.market_returns) > 100:
            self.market_returns.pop(0)

    def get_regime(self):
        """Get current market regime"""
        if len(self.market_returns) < 20:
            return 'sideways'

        returns = np.diff(self.market_returns[-100:]) / self.market_returns[-100:-1]
        mean_return = np.mean(returns)
        volatility = np.std(returns)

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


class IntradayTradingBot:
    """5-Minute Intraday Trading Bot"""

    def __init__(self, api_key, secret_key, base_url, model_path):
        """Initialize the trading bot"""
        print("\n🤖 Initializing 5-Minute Intraday Trading Bot...")

        self.api = REST(api_key, secret_key, base_url, api_version='v2')
        self.tickers = TICKERS
        self.model_path = model_path
        self.model = None

        # Components
        self.feature_calculator = IntradayFeatureCalculator()
        self.regime_detector = MarketRegimeDetector()

        # Trading state
        self.positions = {}
        self.last_prices = {}
        self.trade_count = 0
        self.start_value = None

        # Load model
        self._load_model()

        # Initialize account
        self._initialize_account()

    def _load_model(self):
        """Load the trained model"""
        try:
            # Check if it's a .pkl file (Modal output) or .zip (SB3 format)
            if self.model_path.endswith('.pkl'):
                print(f"📂 Loading Modal-trained model: {self.model_path}")
                with open(self.model_path, 'rb') as f:
                    model_params = pickle.load(f)
                    # Reconstruct model from parameters
                    from stable_baselines3 import PPO
                    import gymnasium as gym
                    from gymnasium import spaces

                    # Create dummy environment for model structure
                    obs_dim = len(self.tickers) * 14 + 11 + len(self.tickers)
                    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
                    act_space = spaces.Box(low=-1, high=1, shape=(len(self.tickers),), dtype=np.float32)

                    self.model = PPO('MlpPolicy', None)
                    self.model.set_parameters(model_params)
            else:
                print(f"📂 Loading SB3 model: {self.model_path}")
                from stable_baselines3 import PPO
                self.model = PPO.load(self.model_path)

            print("✅ Model loaded successfully")

        except FileNotFoundError:
            print(f"⚠️  Model not found at {self.model_path}")
            print("   Using random actions for demonstration")
            self.model = None

    def _initialize_account(self):
        """Initialize account and clear positions"""
        try:
            account = self.api.get_account()
            self.start_value = float(account.portfolio_value)
            print(f"💰 Account Value: ${self.start_value:,.2f}")

            # Get current positions
            positions = self.api.list_positions()
            if positions:
                print(f"📊 Current positions: {len(positions)}")
                for pos in positions:
                    self.positions[pos.symbol] = int(pos.qty)
            else:
                print("📊 No open positions")

        except Exception as e:
            print(f"❌ Error initializing account: {e}")

    def get_market_data(self):
        """Get real-time market data"""
        market_data = {}

        for ticker in self.tickers:
            try:
                # Get latest bar
                bars = self.api.get_bars(
                    ticker,
                    '5Min',
                    limit=1
                ).df

                if not bars.empty:
                    latest_bar = bars.iloc[-1]
                    market_data[ticker] = {
                        'price': float(latest_bar['close']),
                        'volume': float(latest_bar['volume']),
                        'high': float(latest_bar['high']),
                        'low': float(latest_bar['low'])
                    }

                    # Update feature calculator
                    self.feature_calculator.update_history(
                        ticker,
                        float(latest_bar['close']),
                        float(latest_bar['volume']),
                        float(latest_bar['high']),
                        float(latest_bar['low'])
                    )

            except Exception as e:
                print(f"  ⚠️  Error getting data for {ticker}: {e}")

        return market_data

    def build_observation(self, market_data):
        """Build observation for model"""
        obs = []

        # Get current account info
        try:
            account = self.api.get_account()
            portfolio_value = float(account.portfolio_value)
            cash = float(account.cash)
        except:
            portfolio_value = self.start_value
            cash = self.start_value

        # Per-stock features
        for ticker in self.tickers:
            if ticker in market_data:
                price = market_data[ticker]['price']
                features = self.feature_calculator.calculate_features(ticker)

                obs.extend([price / 1000])  # Normalized price
                obs.extend(features)
            else:
                obs.extend(np.zeros(15))

        # Portfolio state
        cash_ratio = cash / portfolio_value if portfolio_value > 0 else 1
        obs.extend([
            cash_ratio,
            portfolio_value / self.start_value if self.start_value > 0 else 1
        ])

        # Market regime
        if market_data:
            prices = [market_data[t]['price'] for t in market_data.keys()]
            self.regime_detector.update(prices)
            regime = self.regime_detector.get_regime()
        else:
            regime = 'sideways'

        # One-hot encode regime
        regime_encoding = {'strong_bull': 0, 'bull': 1, 'sideways': 2, 'bear': 3, 'crash': 4}
        regime_one_hot = np.zeros(5)
        regime_one_hot[regime_encoding[regime]] = 1
        obs.extend(regime_one_hot)

        # Time features
        now = datetime.now()
        is_open = 1 if now.hour == 9 and now.minute >= 30 else 0
        is_close = 1 if now.hour == 15 and now.minute >= 30 else 0
        is_lunch = 1 if 12 <= now.hour <= 13 else 0
        minutes_until_close = (16 - now.hour) * 60 - now.minute
        time_until_close = minutes_until_close / 390

        obs.extend([is_open, is_close, is_lunch, time_until_close])

        # Current holdings
        holdings = []
        for ticker in self.tickers:
            holdings.append(self.positions.get(ticker, 0) / 1000)
        obs.extend(holdings)

        return np.array(obs, dtype=np.float32)

    def execute_trades(self, actions, market_data):
        """Execute trades based on model actions"""
        if not market_data:
            print("  ⚠️  No market data available")
            return

        # Get account info
        try:
            account = self.api.get_account()
            portfolio_value = float(account.portfolio_value)
            cash = float(account.cash)
        except:
            print("  ❌ Error getting account info")
            return

        trades_executed = []

        for i, ticker in enumerate(self.tickers):
            if ticker not in market_data:
                continue

            try:
                # Convert action to target allocation
                target_allocation = (actions[i] + 1) / 2  # Scale from [-1,1] to [0,1]
                target_allocation *= MAX_POSITION_SIZE  # Apply position limit

                target_value = target_allocation * portfolio_value
                current_price = market_data[ticker]['price']
                target_shares = int(target_value / current_price)

                current_shares = self.positions.get(ticker, 0)
                shares_to_trade = target_shares - current_shares

                # Execute trade if significant
                if abs(shares_to_trade) >= 1:
                    if shares_to_trade > 0:  # Buy
                        # Check cash available
                        cost = shares_to_trade * current_price
                        if cost <= cash * 0.95:  # Leave 5% buffer
                            order = self.api.submit_order(
                                symbol=ticker,
                                qty=shares_to_trade,
                                side='buy',
                                type='market',
                                time_in_force='day'
                            )
                            trades_executed.append(f"BUY {shares_to_trade} {ticker}")
                            cash -= cost

                    else:  # Sell
                        if current_shares >= abs(shares_to_trade):
                            order = self.api.submit_order(
                                symbol=ticker,
                                qty=abs(shares_to_trade),
                                side='sell',
                                type='market',
                                time_in_force='day'
                            )
                            trades_executed.append(f"SELL {abs(shares_to_trade)} {ticker}")

                    # Update local position tracking
                    self.positions[ticker] = target_shares

            except Exception as e:
                print(f"  ❌ Error trading {ticker}: {e}")

        if trades_executed:
            print(f"  ✅ Executed {len(trades_executed)} trades:")
            for trade in trades_executed[:5]:  # Show first 5 trades
                print(f"     {trade}")
            if len(trades_executed) > 5:
                print(f"     ... and {len(trades_executed) - 5} more")

        self.trade_count += len(trades_executed)

    def run(self):
        """Main trading loop - runs every 5 minutes"""
        print("\n" + "="*80)
        print("🚀 STARTING 5-MINUTE INTRADAY PAPER TRADING")
        print("="*80)
        print(f"Stocks: {len(self.tickers)}")
        print(f"Trade Frequency: Every 5 minutes (78 times/day)")
        print(f"Max Position Size: {MAX_POSITION_SIZE*100:.0f}% per stock")
        print("="*80)
        print("\nPress Ctrl+C to stop\n")

        iteration = 0

        while True:
            try:
                # Check if market is open
                clock = self.api.get_clock()
                if not clock.is_open:
                    next_open = clock.next_open
                    print(f"🕐 Market closed. Next open: {next_open}")
                    time.sleep(60)
                    continue

                iteration += 1
                timestamp = datetime.now().strftime('%H:%M:%S')

                print(f"\n[{timestamp}] Iteration {iteration}")
                print("-" * 50)

                # Get market data
                print("  📊 Getting market data...")
                market_data = self.get_market_data()

                if not market_data:
                    print("  ⚠️  No market data available, skipping...")
                    time.sleep(TRADE_INTERVAL)
                    continue

                # Build observation
                observation = self.build_observation(market_data)

                # Get model prediction
                if self.model is not None:
                    actions, _ = self.model.predict(observation, deterministic=True)
                    print(f"  🤖 Model prediction received")
                else:
                    # Random actions for testing
                    actions = np.random.uniform(-1, 1, len(self.tickers))
                    print(f"  🎲 Using random actions (no model)")

                # Execute trades
                self.execute_trades(actions, market_data)

                # Display status
                try:
                    account = self.api.get_account()
                    current_value = float(account.portfolio_value)
                    day_pl = float(account.equity) - float(account.last_equity)
                    total_return = (current_value - self.start_value) / self.start_value * 100

                    print(f"\n  💼 Portfolio Status:")
                    print(f"     Value: ${current_value:,.2f}")
                    print(f"     Day P&L: ${day_pl:+,.2f}")
                    print(f"     Total Return: {total_return:+.2f}%")
                    print(f"     Total Trades: {self.trade_count}")
                except:
                    pass

                # Wait for next 5-minute interval
                print(f"\n  ⏰ Next trade in {TRADE_INTERVAL} seconds...")
                time.sleep(TRADE_INTERVAL)

            except KeyboardInterrupt:
                print("\n\n🛑 Stopping trading bot...")
                break

            except Exception as e:
                print(f"\n❌ Error in main loop: {e}")
                print("  Retrying in 60 seconds...")
                time.sleep(60)

        # Final summary
        try:
            account = self.api.get_account()
            final_value = float(account.portfolio_value)
            total_return = (final_value - self.start_value) / self.start_value * 100

            print("\n" + "="*80)
            print("📊 TRADING SESSION SUMMARY")
            print("="*80)
            print(f"Start Value:    ${self.start_value:,.2f}")
            print(f"Final Value:    ${final_value:,.2f}")
            print(f"Total Return:   {total_return:+.2f}%")
            print(f"Total Trades:   {self.trade_count}")
            print(f"Trade Intervals: {iteration}")
            print("="*80)

        except:
            pass


if __name__ == "__main__":
    # Check for API keys
    if API_KEY == 'your_api_key' or SECRET_KEY == 'your_secret_key':
        print("⚠️  Please set your Alpaca API credentials:")
        print("   export ALPACA_API_KEY='your_key'")
        print("   export ALPACA_SECRET_KEY='your_secret'")
    else:
        # Create and run bot
        bot = IntradayTradingBot(API_KEY, SECRET_KEY, BASE_URL, MODEL_PATH)
        bot.run()