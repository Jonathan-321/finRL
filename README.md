# FinRL Portfolio Training System

**Status:** ✅ Environment Validated | ⚠️ Modal Volume Setup Required | 🚀 Ready for A100 Deployment

A comprehensive Financial Reinforcement Learning system for portfolio optimization using Modal.com cloud infrastructure and HuggingFace datasets.

## 📋 Current Status (Oct 12, 2024)

✅ **All Dependencies Installed & Tested** (17/17 tests passed)
- Python 3.12.11, pandas 2.2.3, numpy 1.26.4
- yfinance 0.2.58, gymnasium 0.29.1, stable-baselines3 2.4.1
- stockstats 0.5.4, datasets 3.6.0, finrl 0.3.8

✅ **Data Pipeline Operational**
- yfinance API tested (AAPL, MSFT, GOOGL ✓)
- HuggingFace datasets loaded (1,961+ samples ✓)
- Technical indicators working (MACD, RSI, SMA, Bollinger ✓)

✅ **Modal Infrastructure Ready**
- Modal CLI v1.1.1 installed ✓
- Authentication configured ✓
- Secrets available: `hf-token`, `aws-secret`, `llm-keys` ✓

⚠️ **Action Required Before Deployment:**
```bash
# Create Modal volume for data persistence
modal volume create finrl-volume
```

📖 **Documentation:** See [`PROGRESS.md`](PROGRESS.md), [`MODAL_SETUP.md`](MODAL_SETUP.md), [`DEBUGGING_SUMMARY.md`](DEBUGGING_SUMMARY.md)

---

## 🚀 Quick Start

### Prerequisites
```bash
# Install core dependencies
pip install modal datasets huggingface_hub yfinance pandas numpy \
    gymnasium stable-baselines3 stockstats matplotlib

# Authenticate with Modal
modal token new

# Create required volume
modal volume create finrl-volume
```

### Validate Environment (Recommended First Step)
```bash
# Run comprehensive diagnostics
python3 quick_diagnostic.py

# Test data pipeline
python3 test_data_download.py

# Test HuggingFace integration
python3 test_massive_datasets.py
```

### Local Training (No Cloud Costs)
```bash
# Train locally on 30 stocks (15-20 mins)
python3 finrl_portfolio_system.py
```

### Deploy to Modal A100 (Production)
```bash
# Option 1: Standard Training (50 stocks, ~$2-3, 1 hour)
modal run modal_finrl_training.py::train_large_portfolio

# Option 2: Massive Training (500+ assets, ~$6-15, 2 hours)
modal run massive_finrl_hf.py::build_massive_dataset
modal run massive_finrl_hf.py::train_massive_portfolio

# Option 3: Interactive Deployment
python3 deploy_modal.py  # Choose options from menu
```

## 📊 Dataset Sources

The system integrates multiple financial datasets:

- **Twitter Financial Sentiment**: `zeroshot/twitter-financial-news-sentiment`
- **Financial Q&A**: `pauri32/fiqa-2018`
- **Yahoo Finance OHLCV**: Via yfinance API
- **Technical Indicators**: Generated using stockstats/ta-lib

**Estimated Training Data**: ~60K samples across 4 datasets

## 🏗️ Project Structure

```
finRL/
├── 📁 FinRL/                          # Core FinRL library (v0.3.8)
│
├── 📄 Documentation
│   ├── README.md                      # This file - project overview
│   ├── PROGRESS.md                    # Comprehensive progress tracking (500+ lines)
│   ├── MODAL_SETUP.md                 # Complete Modal deployment guide (500+ lines)
│   └── DEBUGGING_SUMMARY.md           # Oct 12 debugging session results
│
├── 🧪 Validation & Testing
│   ├── quick_diagnostic.py            # Environment validator (11 tests)
│   ├── test_data_download.py          # Data pipeline tester (4 tests)
│   ├── test_massive_datasets.py       # HuggingFace dataset tester
│   ├── test_finrl_basic.py            # Basic FinRL functionality
│   └── test_finrl_simple.py           # Simple trading scenarios
│
├── 🚀 Training Scripts
│   ├── finrl_portfolio_system.py      # Local portfolio training (30 stocks)
│   ├── modal_finrl_training.py        # Modal A100 standard (50 stocks, $2-3)
│   ├── massive_finrl_hf.py            # Modal A100 massive (500+ assets, $6-15)
│   └── run_massive_training.py        # Training orchestration
│
├── 🔧 Utilities
│   ├── deploy_modal.py                # Interactive deployment tool
│   ├── search_hf_datasets.py          # HF dataset discovery
│   ├── debug_yfinance.py              # yfinance API debugging
│   └── finrl_quick_test.py            # Quick validation
│
└── 📊 Data & Models (generated during training)
    ├── /tmp/finrl_models/             # Trained model checkpoints
    ├── /tmp/finrl_tensorboard/        # TensorBoard logs
    └── trained_models/                # Downloaded models from Modal
```

## 🎯 Training Configurations

### Local Training
- **CPU-based** (No cloud costs)
- **Stocks:** 30 (customizable)
- **Time:** 15-20 minutes
- **Memory:** ~2-4GB RAM
- **Timesteps:** 20,000 per model
- **Algorithms:** PPO, SAC, A2C ensemble
- **Use Case:** Testing, prototyping, small portfolios

### Modal Standard Training
- **GPU:** A100 (40GB VRAM)
- **Stocks:** 50 (S&P 500 top 50)
- **Memory:** 32GB RAM
- **Time:** 45-60 minutes
- **Timesteps:** 500,000
- **Cost:** ~$2-3 per run
- **Use Case:** Production models, research

### Modal Massive Training
- **GPU:** A100 (40GB VRAM)
- **Assets:** 500+ (stocks, crypto, forex, commodities)
- **Memory:** 64GB RAM
- **Time:** 2-3 hours
- **Timesteps:** 2,000,000
- **Cost:** ~$6-15 per run
- **Use Case:** Research-grade multi-asset optimization

### Algorithms Supported
- **PPO** (Proximal Policy Optimization) - Best for discrete decisions
- **SAC** (Soft Actor-Critic) - Best for continuous actions
- **A2C** (Advantage Actor-Critic) - Fast baseline
- **Ensemble** - Combined predictions from all three

## 🔧 Configuration

### Required Secrets
```bash
# HuggingFace token for dataset access
modal secret create hf-token HF_TOKEN=your_token_here
```

### Environment Variables
- `HF_TOKEN`: HuggingFace API token
- `MODAL_TOKEN`: Modal authentication token

## 📈 Expected Outcomes

- Multi-asset portfolio optimization
- Sentiment-enhanced trading decisions
- Risk-adjusted performance metrics
- Production-ready trading models
- Benchmark comparison vs buy-and-hold

## 🧪 Testing Workflow

1. **Local Testing**: Run `test_massive_datasets.py` to verify data access
2. **Dataset Building**: Use `build_massive_dataset()` to prepare training data
3. **Model Training**: Deploy RL algorithms with `train_massive_portfolio()`
4. **Evaluation**: Compare against benchmarks and validate performance

## 💡 Usage Examples

### Basic Training
```python
# Test datasets locally first
python test_massive_datasets.py

# Deploy to Modal
modal run massive_finrl_hf.py::build_massive_dataset
modal run massive_finrl_hf.py::train_massive_portfolio
```

### Custom Portfolio
```python
# Modify symbols in finrl_portfolio_system.py
SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']

# Run custom training
python finrl_portfolio_system.py
```

## 📋 Troubleshooting

### Environment Issues
```bash
# Run comprehensive diagnostics
python3 quick_diagnostic.py
# Should show: 11/11 tests passed

# If imports fail, reinstall dependencies
pip install -r requirements.txt  # (if you create one)
```

### Modal Issues
```bash
# Volume not found error
modal volume create finrl-volume

# Check Modal authentication
modal token set

# List available secrets
modal secret list

# View Modal apps
modal app list

# Monitor training logs
modal app logs finrl-portfolio-training --follow
```

### Data Download Issues
```bash
# Test data pipeline
python3 test_data_download.py
# Should show: 4/4 tests passed

# If yfinance fails, check rate limits
# Wait a few minutes and retry
```

### HuggingFace Issues
```bash
# Create HF secret if missing
modal secret create hf-token HUGGINGFACE_HUB_TOKEN=hf_your_token_here

# Test HF access
python3 test_massive_datasets.py
# Should show: 2/2 datasets loaded
```

### GPU/Memory Issues
- **Out of Memory:** Reduce batch size or number of stocks
- **Timeout:** Increase timeout in function decorator
- **GPU Unavailable:** Try T4 GPU for testing or schedule for off-peak hours

### Common Error Solutions
See [`MODAL_SETUP.md`](MODAL_SETUP.md) for detailed troubleshooting guide

## 🔮 Future Enhancements

- [ ] Real-time trading integration
- [ ] Multi-exchange support (crypto/forex)
- [ ] Advanced risk management
- [ ] Hyperparameter optimization with Optuna
- [ ] Model versioning and A/B testing
- [ ] Production deployment pipelines

## 📚 Resources

- [FinRL Documentation](https://github.com/AI4Finance-Foundation/FinRL)
- [Modal.com Docs](https://modal.com/docs)
- [HuggingFace Datasets](https://huggingface.co/datasets)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)

---

## 🎉 Ready to Deploy!

Your system is **fully validated and ready** for production training:

✅ **Environment:** 17/17 tests passed
✅ **Data Pipeline:** All sources operational
✅ **Modal Setup:** CLI configured, secrets available
⚠️ **Final Step:** `modal volume create finrl-volume`

### Recommended Next Action

```bash
# 1. Create Modal volume
modal volume create finrl-volume

# 2. Run validation (optional but recommended)
python3 quick_diagnostic.py && python3 test_data_download.py

# 3. Start with local training (free)
python3 finrl_portfolio_system.py

# 4. Deploy to Modal when ready
modal run modal_finrl_training.py::train_large_portfolio
```

**Questions?** Check out the detailed docs:
- [`PROGRESS.md`](PROGRESS.md) - Complete project status
- [`MODAL_SETUP.md`](MODAL_SETUP.md) - Modal deployment guide
- [`DEBUGGING_SUMMARY.md`](DEBUGGING_SUMMARY.md) - Recent validation results

---

**Last Updated:** October 12, 2024
**Status:** Ready for Modal A100 training | All systems operational 🚀# finRL
