# Alpaca Paper Trading Integration Guide

**Last Updated:** October 12, 2024
**Status:** Ready for implementation
**Integration Time:** 3-5 days

---

## 🔐 Alpaca API Credentials

**IMPORTANT: Keep these credentials secure!**

```
API Key: PK59TECEVDLEY6E5DSIY
API Secret: KiyhaPQbgWgPwlMI4cngc0nxlchuf9GkWD8siqjn
Endpoint: https://paper-api.alpaca.markets/v2
```

**Security Notes:**
- These are paper trading credentials (virtual $100K)
- No real money at risk
- Do NOT commit these to public repositories
- Store in environment variables or `.env` file (gitignored)

---

## 📋 What Is Paper Trading?

Paper trading = **Real-time trading simulation with fake money**

### Key Benefits
- ✅ **Zero Financial Risk** - $100,000 virtual cash
- ✅ **Real Market Data** - Live prices from NYSE/NASDAQ
- ✅ **Real Execution Logic** - Actual order routing and fills
- ✅ **Market Conditions** - News, volatility, gaps, halts
- ✅ **Validation** - Proves model works in current market

### What You Get
- Real-time portfolio value
- Actual fill prices and slippage
- Transaction cost simulation
- Market hours constraints (9:30 AM - 4:00 PM ET)
- Pre-market and after-hours trading

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│  FinRL PPO Model (Trained on Historical Data)  │
│  - Observes market state every 5 minutes         │
│  - Makes portfolio allocation decisions          │
│  - Generates buy/sell orders                     │
└──────────────┬──────────────────────────────────┘
               │ Actions (buy/sell orders)
               ↓
┌─────────────────────────────────────────────────┐
│  Alpaca Paper Trading API                      │
│  - Executes orders with virtual money          │
│  - Routes through real market infrastructure   │
│  - Tracks P&L and positions                    │
│  - Provides real-time market data              │
└──────────────┬──────────────────────────────────┘
               │ Market data & fills
               ↓
┌─────────────────────────────────────────────────┐
│  Real Stock Market (NYSE, NASDAQ)             │
│  - Live bid/ask spreads                         │
│  - Order book dynamics                          │
│  - Market hours (9:30 AM - 4 PM ET)           │
│  - Real-time price movements                    │
└─────────────────────────────────────────────────┘
```

---

## 🎯 Trading Strategy Configuration

### Timeframe Decision
Based on our research guide and model design:

**Recommended: 5-Minute Bars (Intraday Trading)**

#### Rationale:
1. **Model Training**: Trained on daily data (2020-2024)
2. **Portfolio Size**: 50 stocks requires multiple rebalancing opportunities
3. **Transaction Costs**: 0.1% per trade means need profitable moves
4. **Market Efficiency**: Daily bars allow time for fundamentals to play out
5. **Execution Quality**: Reduces market impact vs. high-frequency

#### Alternative Timeframes:

| Timeframe | Pros | Cons | Best For |
|-----------|------|------|----------|
| **1-Minute** | More signals, quick adaptation | High costs, noisy | HFT research |
| **5-Minute** | Balanced, good execution | Moderate complexity | Intraday strategies |
| **15-Minute** | Smoother signals, lower costs | Fewer opportunities | Swing trading |
| **1-Hour** | Very low costs, stable | Slow adaptation | Position trading |
| **Daily** | Matches training data | Slow, missed intraday moves | Long-term investing |

**Our Choice: 5-Minute bars** - Best balance of execution quality and opportunity frequency.

---

## 📊 Model Architecture & Parameters

### PPO Model Specifications

#### Model Size
```
Total Parameters: ~50,000 - 100,000
Breakdown:
  - Policy Network: MLP with 2 hidden layers
    Layer 1: [state_size] → 64 neurons (~40K params)
    Layer 2: 64 → 64 neurons (~4K params)
    Output: 64 → [action_size] (~3K params)

  - Value Network: Similar architecture
    Total: ~3K additional params

Model Size: ~400 KB (small, fast inference)
```

**Context:**
- GPT-3: 175 billion parameters (400,000x larger!)
- Our model: 50-100K parameters
- **This is a small model** - trains fast, infers fast

#### State Space (Input)
```python
State Dimensions: 803 features
Breakdown:
  - Prices: 50 stocks × 1 close price = 50
  - Technical Indicators: 50 stocks × 13 indicators = 650
    - MACD: 50
    - RSI: 50
    - CCI: 50
    - DX: 50
    - SMA (20, 50, 200): 150
    - Bollinger Bands (upper, middle, lower): 150
    - ATR: 50
    - Williams %R: 50
    - Stochastic: 50
  - Portfolio State: 2 features
    - Current cash balance: 1
    - Portfolio value: 1
  - Holdings: 50 (shares owned per stock)

Total: 50 + 650 + 2 + 50 + 1 (day of week) = 803
```

#### Action Space (Output)
```python
Actions: Continuous values in [-1, +1] for each stock
Dimension: 50 (one per stock)

Interpretation:
  -1.0 = Sell all shares of this stock
   0.0 = Hold current position
  +1.0 = Buy maximum shares (up to hmax)

Constraints:
  - Total portfolio value must equal $1M
  - Cannot short (negative positions)
  - Max position per stock: hmax = 100 shares
```

#### Reward Function
```python
Reward = (portfolio_value_t - portfolio_value_t-1) * 1e-4
       - transaction_cost_penalty

Where:
  transaction_cost = 0.001 * |trade_value|  # 0.1% per trade
```

#### Training Hyperparameters (Current)
```python
Algorithm: PPO (Proximal Policy Optimization)
Learning Rate: 3e-4
Batch Size: 64
N Steps: 2,048 (steps before update)
N Epochs: 10 (updates per batch)
Gamma: 0.99 (discount factor)
GAE Lambda: 0.95 (advantage estimation)
Clip Range: 0.2 (policy update limit)

Training Time: ~50 minutes on A100 GPU
Training Timesteps: 500,000
```

---

## 🎓 Industry-Standard Explanation

### For Quantitative Researchers

**"We use Proximal Policy Optimization (PPO) - a state-of-the-art reinforcement learning algorithm - to learn portfolio allocation strategies across 50 S&P 500 stocks. The model observes a 803-dimensional state space consisting of prices, 13 technical indicators, and portfolio state, then outputs continuous actions representing target portfolio weights. We train on 4 years of historical data (2020-2024) using risk-adjusted rewards and validate with walk-forward analysis."**

### For Portfolio Managers

**"This is an AI-driven portfolio management system that learns optimal trading strategies from historical market data. Instead of hard-coding rules, the algorithm discovers patterns by trying different trades and learning from outcomes (similar to how AlphaGo learned chess). It manages a 50-stock portfolio with automatic rebalancing based on market conditions and technical signals."**

### For Software Engineers

**"We've built a reinforcement learning agent using the PPO algorithm from Stable-Baselines3. The model is a small MLP (~50K parameters) that takes market state (prices + indicators) as input and outputs trading actions (buy/sell quantities). We train it on GPUs via Modal.com, then deploy it to trade via Alpaca's paper trading API. The system monitors real-time market data and executes trades automatically."**

### For Executives/Business

**"We're developing AI technology that automates stock portfolio management. The system learns trading strategies from 4 years of historical data, then applies those strategies to make real-time trading decisions. We validate it using paper trading (fake money) before any real capital deployment. The technology is similar to what hedge funds use, but we're building it as open-source software."**

### Key Benchmarks

| Metric | Our Model | Industry Standard | Interpretation |
|--------|-----------|-------------------|----------------|
| **Sharpe Ratio** | 0.51 | >1.0 desirable | Need improvement |
| **Annual Return** | 17.32% | 10-15% typical | Competitive |
| **Max Drawdown** | -20.66% | <20% preferred | Acceptable but high |
| **Win Rate** | 58.3% | 55-60% typical | Good |
| **Volatility** | 22.96% | <25% target | Acceptable |

**Assessment:** Model shows promise but needs optimization for risk-adjusted returns.

---

## 🔧 Implementation

### Setup (5 minutes)

```bash
# 1. Install dependencies
pip install alpaca-trade-api stable-baselines3 pandas numpy

# 2. Create .env file (NEVER commit this!)
cat > .env <<EOF
ALPACA_API_KEY=PK59TECEVDLEY6E5DSIY
ALPACA_SECRET_KEY=KiyhaPQbgWgPwlMI4cngc0nxlchuf9GkWD8siqjn
ALPACA_BASE_URL=https://paper-api.alpaca.markets
EOF

# 3. Test connection
python test_alpaca_connection.py
```

### Integration Code

```python
# paper_trading_bot.py
import os
import time
import numpy as np
import pandas as pd
from alpaca_trade_api import REST
from stable_baselines3 import PPO
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class AlpacaTradingBot:
    def __init__(self, model_path, trading_interval_minutes=5):
        # Connect to Alpaca
        self.api = REST(
            key_id=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL')
        )

        # Load trained model
        self.model = PPO.load(model_path)

        # Configuration
        self.trading_interval = trading_interval_minutes * 60  # Convert to seconds
        self.tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
            'BRK-B', 'UNH', 'JNJ', 'V', 'WMT', 'JPM', 'MA', 'PG',
            'XOM', 'HD', 'CVX', 'LLY', 'ABBV', 'MRK', 'PEP', 'KO',
            'COST', 'AVGO', 'TMO', 'MCD', 'CSCO', 'ACN', 'ABT',
            'NKE', 'DHR', 'VZ', 'ADBE', 'NFLX', 'CRM', 'CMCSA',
            'TXN', 'PM', 'NEE', 'UNP', 'ORCL', 'WFC', 'DIS', 'HON',
            'BAC', 'MS', 'RTX', 'BMY', 'AMGN'
        ]

        print(f"✅ Connected to Alpaca Paper Trading")
        print(f"📊 Portfolio Value: ${self.get_portfolio_value():,.2f}")
        print(f"⏰ Trading Interval: {trading_interval_minutes} minutes")

    def get_portfolio_value(self):
        """Get current portfolio value"""
        account = self.api.get_account()
        return float(account.portfolio_value)

    def get_current_state(self):
        """Fetch current market data and construct state vector"""
        # Get current prices
        bars = self.api.get_bars(
            self.tickers,
            '5Min',
            limit=200  # Last 200 bars for technical indicators
        ).df

        # Calculate technical indicators
        # (Simplified - in production, use full indicator suite)
        state = []
        for ticker in self.tickers:
            ticker_data = bars[bars.symbol == ticker]

            # Price
            current_price = ticker_data['close'].iloc[-1]
            state.append(current_price)

            # Simple indicators (in production, add all 13)
            state.append(ticker_data['close'].rolling(20).mean().iloc[-1])  # SMA20
            state.append(ticker_data['close'].rolling(50).mean().iloc[-1])  # SMA50

            # More indicators...

        # Add portfolio state
        account = self.api.get_account()
        state.append(float(account.cash))
        state.append(float(account.portfolio_value))

        # Add positions
        positions = self.api.list_positions()
        position_dict = {p.symbol: float(p.qty) for p in positions}
        for ticker in self.tickers:
            state.append(position_dict.get(ticker, 0))

        return np.array(state)

    def execute_trades(self, actions):
        """Execute trades based on model actions"""
        positions = self.api.list_positions()
        position_dict = {p.symbol: float(p.qty) for p in positions}

        for ticker, action in zip(self.tickers, actions):
            current_qty = position_dict.get(ticker, 0)

            # Convert action to target shares
            if action > 0.1:  # Buy signal
                target_qty = int(action * 100)  # Scale to shares
                if target_qty > current_qty:
                    buy_qty = target_qty - current_qty
                    print(f"🟢 BUY {buy_qty} shares of {ticker}")
                    try:
                        self.api.submit_order(
                            symbol=ticker,
                            qty=buy_qty,
                            side='buy',
                            type='market',
                            time_in_force='gtc'
                        )
                    except Exception as e:
                        print(f"❌ Error buying {ticker}: {e}")

            elif action < -0.1:  # Sell signal
                if current_qty > 0:
                    sell_qty = int(abs(action) * current_qty)
                    print(f"🔴 SELL {sell_qty} shares of {ticker}")
                    try:
                        self.api.submit_order(
                            symbol=ticker,
                            qty=sell_qty,
                            side='sell',
                            type='market',
                            time_in_force='gtc'
                        )
                    except Exception as e:
                        print(f"❌ Error selling {ticker}: {e}")

    def is_market_open(self):
        """Check if market is currently open"""
        clock = self.api.get_clock()
        return clock.is_open

    def run_trading_loop(self):
        """Main trading loop"""
        print("\n🚀 Starting Paper Trading Bot...")
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        iteration = 0
        while True:
            try:
                # Check if market is open
                if not self.is_market_open():
                    print("🌙 Market closed. Waiting...")
                    time.sleep(60)
                    continue

                iteration += 1
                print(f"\n📊 Iteration #{iteration} - {datetime.now().strftime('%H:%M:%S')}")

                # Get current state
                state = self.get_current_state()

                # Get model prediction
                action, _ = self.model.predict(state, deterministic=True)

                # Execute trades
                self.execute_trades(action)

                # Log portfolio value
                portfolio_value = self.get_portfolio_value()
                print(f"💰 Portfolio Value: ${portfolio_value:,.2f}")

                # Wait for next interval
                time.sleep(self.trading_interval)

            except KeyboardInterrupt:
                print("\n🛑 Stopping bot...")
                break
            except Exception as e:
                print(f"❌ Error in trading loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

if __name__ == "__main__":
    # Load trained model
    bot = AlpacaTradingBot(
        model_path="trained_models/ppo_portfolio.zip",
        trading_interval_minutes=5
    )

    # Start trading
    bot.run_trading_loop()
```

---

## 📈 Monitoring & Logging

### Real-Time Dashboard

Access Alpaca's paper trading dashboard:
- URL: https://app.alpaca.markets/paper/dashboard/overview
- Login with your Alpaca account
- View real-time P&L, positions, order history

### Performance Tracking

```python
# performance_tracker.py
import pandas as pd
from datetime import datetime

class PerformanceTracker:
    def __init__(self, api):
        self.api = api
        self.initial_value = float(api.get_account().portfolio_value)
        self.log_file = f"trading_log_{datetime.now().strftime('%Y%m%d')}.csv"

    def log_iteration(self):
        """Log current portfolio state"""
        account = self.api.get_account()

        log_entry = {
            'timestamp': datetime.now(),
            'portfolio_value': float(account.portfolio_value),
            'cash': float(account.cash),
            'return_pct': (float(account.portfolio_value) - self.initial_value) / self.initial_value * 100
        }

        # Append to CSV
        pd.DataFrame([log_entry]).to_csv(
            self.log_file,
            mode='a',
            header=not os.path.exists(self.log_file),
            index=False
        )

        return log_entry
```

---

## 🎯 Success Metrics

### Daily Tracking
- [ ] Portfolio value at market open
- [ ] Portfolio value at market close
- [ ] Number of trades executed
- [ ] Total transaction costs
- [ ] Win/loss ratio

### Weekly Analysis
- [ ] Weekly return vs S&P 500
- [ ] Maximum drawdown
- [ ] Sharpe ratio (rolling 7-day)
- [ ] Portfolio turnover rate

### Monthly Review
- [ ] Compare to backtest results
- [ ] Analyze worst performing days
- [ ] Review model predictions vs actual outcomes
- [ ] Adjust hyperparameters if needed

---

## 🚨 Risk Management

### Position Limits
```python
MAX_POSITION_SIZE = 0.10  # 10% of portfolio per stock
MAX_DAILY_LOSS = 0.02     # 2% daily stop loss
MAX_LEVERAGE = 1.0        # No leverage (paper trading)
```

### Circuit Breakers
```python
def check_circuit_breakers(current_value, initial_value):
    """Stop trading if losses exceed threshold"""
    daily_return = (current_value - initial_value) / initial_value

    if daily_return < -MAX_DAILY_LOSS:
        print("🚨 CIRCUIT BREAKER: Daily loss limit exceeded")
        return False  # Stop trading

    return True  # Continue trading
```

---

## 🔄 Next Steps

### Phase 1: Initial Deployment (Week 1)
1. ✅ Setup Alpaca account and credentials
2. ⏳ Implement basic trading bot (code above)
3. ⏳ Test with 1-2 stocks first
4. ⏳ Verify orders execute correctly
5. ⏳ Monitor for 1 week

### Phase 2: Full Deployment (Week 2-3)
1. ⏳ Scale to all 50 stocks
2. ⏳ Add full technical indicator suite
3. ⏳ Implement performance tracking
4. ⏳ Set up automated alerts (email/Slack)
5. ⏳ Run continuously during market hours

### Phase 3: Optimization (Week 4+)
1. ⏳ Analyze paper trading results
2. ⏳ Compare to backtest performance
3. ⏳ Identify model weaknesses
4. ⏳ Retrain with updated hyperparameters
5. ⏳ Consider moving to real money (if >3 months profitable)

---

## 📞 Support & Resources

- **Alpaca Docs:** https://alpaca.markets/docs/
- **Alpaca Discord:** https://alpaca.markets/discord
- **Our Codebase:** See `finRL/` directory
- **Research Guide:** See `FINRL_RESEARCH_GUIDE.md`

---

## ⚠️ Important Disclaimers

1. **No Financial Advice** - This is educational/research software
2. **Past Performance ≠ Future Results** - Backtest results don't guarantee live performance
3. **Paper Trading ≠ Real Trading** - Execution quality differs
4. **Risk of Loss** - Real money trading can result in substantial losses
5. **Start Small** - If moving to real money, start with <$1,000

---

**Last Updated:** October 12, 2024
**Next Review:** After 2 weeks of paper trading data
