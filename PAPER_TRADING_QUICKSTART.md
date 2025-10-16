# Paper Trading Quick Start Guide

**Status:** ✅ Ready to trade
**Model:** Trained (500K timesteps, 16.5 minutes)
**Account:** Alpaca Paper Trading ($100,000 virtual)
**Connection:** Verified

---

## ✅ What's Ready

1. **Trained Model**: PPO model trained on 50 S&P 500 stocks (2020-2024 data)
2. **Alpaca Account**: Paper trading account connected and verified
3. **Trading Bot**: `alpaca_paper_trading.py` ready to execute trades
4. **Market Data**: 5-minute bar strategy configured

---

## 🚀 Quick Start (3 Steps)

### Step 1: Verify Setup

```bash
# Test connection (already done!)
python test_alpaca_connection.py
```

**Expected Output:**
```
✅ CONNECTION SUCCESSFUL!
Account Number: PA31TZQPPHEF
Portfolio Value: $100,000.00
Buying Power: $200,000.00
```

### Step 2: Get Trained Model

The trained model from Modal needs to be downloaded:

```bash
# Option A: If model was saved to Modal volume
modal volume get finrl-volume /tmp/finrl_models/finrl_portfolio_500000_steps.zip ./trained_model.zip

# Option B: Retrain quickly (16 minutes on A100)
modal run modal_finrl_training.py::train_large_portfolio
```

### Step 3: Start Paper Trading

```bash
# Run the bot (will trade every 5 minutes during market hours)
python alpaca_paper_trading.py
```

**Market Hours:** 9:30 AM - 4:00 PM ET (Monday-Friday)

---

## 📊 What Happens When You Run It

### During Market Hours:

```
🚀 Starting paper trading bot
   Interval: 5 minutes
   Market hours: 9:30 AM - 4:00 PM ET

============================================================
Iteration 1 - 2024-10-13 09:35:00
============================================================

📊 Executing trades at 2024-10-13 09:35:23
  🟢 BUY 15 AAPL @ $178.42
  🟢 BUY 8 MSFT @ $412.56
  🔴 SELL 5 TSLA @ $242.18
  ...

💰 Portfolio Value: $100,234.56
   Cash: $42,156.78
   P&L: $234.56 (+0.23%)

⏳ Waiting 5 minutes until next cycle...
```

### Outside Market Hours:

```
⏸️  Market closed. Next open: 2024-10-13 09:30:00
```

---

## 🎯 Strategy Details

**Timeframe:** 5-minute bars
**Stocks:** 50 S&P 500 stocks
**Capital:** $100,000 (paper money)
**Rebalancing:** Every 5 minutes during market hours
**Transaction Costs:** 0.1% per trade (simulated)

**Risk Management:**
- Max 100 shares per stock
- No short selling (long-only)
- Automatic position sizing based on model output

---

## 📈 Monitoring Performance

### Real-time Dashboard
Check Alpaca dashboard: https://app.alpaca.markets/paper/dashboard

### Performance Metrics

The bot tracks:
- Portfolio value over time
- Win rate and Sharpe ratio
- Transaction costs
- Position sizes

### Example Performance Report

```
============================================================
FINAL TRADING REPORT
============================================================
Portfolio Value: $102,456.78
Cash: $38,234.12
P&L: $2,456.78
Return: +2.46%

Open Positions: 35
  AAPL: 12 shares @ $180.25
  MSFT: 6 shares @ $415.30
  ...
============================================================
```

---

## 🛠️ Customization

### Change Trading Interval

Edit `alpaca_paper_trading.py`:

```python
# Trade every 15 minutes instead of 5
bot.run(interval_minutes=15)
```

### Trade Different Stocks

```python
# Smaller portfolio of tech stocks
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
bot = AlpacaTradingBot(model_path=MODEL_PATH, tickers=TICKERS)
```

### Adjust Capital

```python
# Start with $50K instead of $100K
bot = AlpacaTradingBot(
    model_path=MODEL_PATH,
    tickers=TICKERS,
    initial_capital=50000
)
```

---

## ⚠️ Important Notes

### 1. Model-Data Mismatch
The model was trained on **daily** data but we're trading on **5-minute** bars. This means:
- ✅ Technical indicators still work
- ⚠️ Model may need retraining on intraday data for optimal performance
- 💡 Consider this a proof-of-concept / validation phase

### 2. Market Hours Only
- Trading only happens during market hours (9:30 AM - 4 PM ET)
- Bot sleeps outside trading hours
- Weekend = no trading

### 3. Paper Money Only
- This is **simulation only** - no real money at risk
- Perfect for testing and validation
- Transition to real trading requires additional risk controls

### 4. Internet Connection
- Bot needs continuous internet connection
- If disconnected, positions remain but bot stops trading
- Restart bot to resume (it will sync with current positions)

---

## 🐛 Troubleshooting

### "Model file not found"
Download trained model from Modal or retrain:
```bash
modal run modal_finrl_training.py::train_large_portfolio
```

### "Connection failed"
Check `.env` file has correct credentials:
```bash
cat .env
# Should show ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL
```

### "Market data not available"
Some stocks may not have 5-minute data. Bot will skip them and continue.

### Bot crashes
Check error message. Common issues:
- API rate limits (wait a minute and restart)
- Invalid stock ticker (update TICKERS list)
- Network timeout (check internet connection)

---

## 📚 Next Steps

### Short Term (This Week)
1. ✅ Run bot during market hours
2. ✅ Monitor performance dashboard
3. ✅ Collect real trading data

### Medium Term (2-4 Weeks)
1. 🔄 Retrain model on 5-minute historical data
2. 📊 Analyze paper trading results
3. 🎯 Optimize hyperparameters based on live performance

### Long Term (1-3 Months)
1. 🧪 A/B test different strategies
2. 📈 Compare to buy-and-hold benchmark
3. 🔐 Add advanced risk controls
4. 💼 Consider real trading (if performance is good)

---

## 🎓 Learning Resources

**Reinforcement Learning:**
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [PPO Algorithm Explained](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

**Alpaca API:**
- [Paper Trading Guide](https://alpaca.markets/docs/trading/paper-trading/)
- [API Documentation](https://alpaca.markets/docs/api-references/trading-api/)

**FinRL Framework:**
- [FinRL GitHub](https://github.com/AI4Finance-Foundation/FinRL)
- [Research Papers](https://github.com/AI4Finance-Foundation/FinRL/tree/master/FinRL_Papers)

---

## 📞 Support

**Issues?** Check:
1. `test_alpaca_connection.py` - Verify API access
2. Market hours - Bot only trades 9:30 AM - 4 PM ET
3. Internet connection - Required for real-time data

**Questions?** Review:
- `ALPACA_PAPER_TRADING_GUIDE.md` - Full integration details
- `MODEL_ARCHITECTURE_EXPLAINED.md` - Model technical specs
- `FINRL_RESEARCH_GUIDE.md` - FinRL framework insights

---

**Ready to trade?** Run `python alpaca_paper_trading.py` when market opens! 🚀
