# Modal Setup Documentation

**Last Updated:** October 12, 2024
**Status:** ✅ Modal CLI Installed | ⚠️ Volume Setup Required

---

## 📋 Current Modal Configuration

### ✅ Completed
- [x] Modal CLI installed (v1.1.1)
- [x] Modal authentication configured
- [x] Secrets available:
  - `hf-token` (HuggingFace)
  - `aws-secret`
  - `llm-keys`

### ⚠️ Requires Setup
- [ ] Create `finrl-volume` for data persistence
- [ ] Test Modal deployment scripts
- [ ] Validate A100 GPU access
- [ ] Configure results storage

---

## 🛠️ Modal Scripts Overview

### 1. **`modal_finrl_training.py`** (Main Training Script)
**Purpose:** Large-scale FinRL training on A100 GPU
**Configuration:**
```python
App: "finrl-portfolio-training"
GPU: A100
Memory: 32GB
Timeout: 1 hour
Training: 500,000 timesteps
Portfolio: $1M (50 stocks)
```

**Key Functions:**
- `train_large_portfolio()` - Main training function (A100)
- `hyperparameter_optimization()` - Optuna-based tuning (A100)
- `generate_trading_report()` - Results visualization

**Training Setup:**
- **Stocks:** S&P 500 top 50
- **Period:** 2020-01-01 to 2024-10-01 (4+ years)
- **Indicators:** 13 advanced technical indicators
  - MACD, RSI (14, 30), SMA (20, 50, 200)
  - Bollinger Bands, ATR, CCI, Williams %R
  - Stochastic oscillators
- **Initial Capital:** $1,000,000
- **Transaction Cost:** 0.1%

### 2. **`massive_finrl_hf.py`** (HuggingFace Integration)
**Purpose:** Massive multi-asset training with HF datasets
**Configuration:**
```python
App: "massive-finrl-hf"
GPU: A100
Memory: 64GB
Timeout: 2 hours
Training: 2,000,000 timesteps
Portfolio: $10M (500+ assets)
```

**Key Functions:**
- `build_massive_dataset()` - Load/create massive dataset (A100, 64GB)
- `train_massive_portfolio()` - Train on multi-asset universe (A100, 64GB)

**Asset Universe:**
- **Stocks:** S&P 500 (all 500)
- **Crypto:** Top 50 cryptocurrencies
- **Forex:** Major pairs (10+)
- **Commodities:** Metals, energy, agriculture (10+)
- **Total:** 570+ assets

**Data Sources:**
- Yahoo Finance (OHLCV)
- HuggingFace datasets (sentiment, context)
- CCXT (crypto data)
- Alpha Vantage (alternative data)

### 3. **`deploy_modal.py`** (Deployment Helper)
**Purpose:** Interactive deployment tool
**Features:**
- Environment validation
- Quick local testing
- Modal deployment wrapper
- Setup guidance

**Usage:**
```bash
python3 deploy_modal.py
# Options:
# 1. Quick local test
# 2. Deploy to Modal A100
# 3. Setup Modal CLI
# 4. Exit
```

---

## 🚀 Setup Instructions

### Step 1: Create Modal Volume
The scripts reference a volume that doesn't exist yet:

```bash
# Create the volume
modal volume create finrl-volume

# Verify creation
modal volume list | grep finrl
```

### Step 2: Verify Modal Secrets
Check that required secrets are accessible:

```bash
# List secrets
modal secret list

# Verify HF token exists (required for massive_finrl_hf.py)
# Should show: hf-token
```

### Step 3: Test Basic Modal Deployment
Create a simple test to verify Modal is working:

```bash
# Create test_modal.py
modal run test_modal.py::hello
```

Example `test_modal.py`:
```python
import modal

app = modal.App("test-finrl")
image = modal.Image.debian_slim().pip_install("numpy")

@app.function(image=image)
def hello():
    import numpy as np
    print("✓ Modal working!")
    print(f"✓ NumPy version: {np.__version__}")
    return "Success"
```

### Step 4: Run Small-Scale Test
Before using expensive A100 GPU, test with smaller config:

```bash
# Edit modal_finrl_training.py
# Change: gpu="A100" → gpu="T4"
# Change: train_timesteps=500000 → 50000
# Change: tickers=SP500_TOP50[:5] (only 5 stocks)

# Run test
modal run modal_finrl_training.py::train_large_portfolio
```

### Step 5: Full Deployment
Once tests pass, run full training:

```bash
# Option 1: Standard training (50 stocks)
modal run modal_finrl_training.py::train_large_portfolio

# Option 2: Massive training (500+ assets)
modal run massive_finrl_hf.py::build_massive_dataset
modal run massive_finrl_hf.py::train_massive_portfolio
```

---

## 💰 Cost Estimates

### GPU Pricing (Modal)
- **T4 GPU:** ~$0.60/hour (for testing)
- **A100 (40GB):** ~$2.50-3.00/hour (for production)
- **A100 (80GB):** ~$4.00-5.00/hour (for massive datasets)

### Training Cost Estimates

#### Small Test (T4, 50K timesteps, 5 stocks)
- **Time:** ~10-15 minutes
- **Cost:** ~$0.15-0.30
- **Purpose:** Validation

#### Standard Training (A100, 500K timesteps, 50 stocks)
- **Time:** ~45-60 minutes
- **Cost:** ~$2.00-3.00
- **Purpose:** Production model

#### Massive Training (A100, 2M timesteps, 500+ assets)
- **Time:** ~2-3 hours
- **Cost:** ~$6.00-15.00
- **Purpose:** Research-grade model

### Monthly Costs (Typical Usage)
- **Development:** 10 test runs @ T4 = ~$3.00
- **Production:** 4 retraining @ A100 = ~$12.00
- **Research:** 2 massive runs = ~$30.00
- **Total:** ~$45.00/month

---

## 🐛 Common Issues & Solutions

### Issue 1: Volume Not Found
**Error:** `Volume 'finrl-volume' not found`
**Solution:**
```bash
modal volume create finrl-volume
```

### Issue 2: Secret Not Found
**Error:** `Secret 'hf-token' not found`
**Solution:**
```bash
# Get HF token from: https://huggingface.co/settings/tokens
modal secret create hf-token HUGGINGFACE_HUB_TOKEN=hf_your_token_here
```

### Issue 3: GPU Not Available
**Error:** `A100 GPU not available`
**Solution:**
- Check Modal dashboard for GPU availability
- Try different region
- Use T4 as fallback
- Schedule run for off-peak hours

### Issue 4: Import Errors in Modal
**Error:** `ModuleNotFoundError: No module named 'X'`
**Solution:**
- Add package to image definition:
```python
image = modal.Image.debian_slim().pip_install([
    "yfinance",
    "pandas",
    # ... your missing package
])
```

### Issue 5: Timeout During Training
**Error:** `Function timed out after 3600s`
**Solution:**
- Increase timeout in function decorator:
```python
@app.function(
    timeout=7200,  # 2 hours instead of 1
    # ...
)
```

### Issue 6: Out of Memory
**Error:** `OutOfMemoryError` or `CUDA out of memory`
**Solution:**
- Increase memory allocation:
```python
@app.function(
    memory=65536,  # 64GB instead of 32GB
    # ...
)
```
- Reduce batch size
- Reduce number of assets
- Use gradient checkpointing

---

## 📊 Monitoring & Debugging

### Check Modal Logs
```bash
# View recent logs
modal app logs finrl-portfolio-training

# Follow logs in real-time
modal app logs finrl-portfolio-training --follow
```

### Check GPU Usage
```bash
# View GPU stats during training
modal app logs finrl-portfolio-training | grep GPU
```

### TensorBoard Integration
Results are logged to `/tmp/finrl_tensorboard/` in the Modal container.

To view locally:
```bash
# Download logs from Modal (if volume mounted)
modal volume get finrl-volume /tmp/finrl_tensorboard ./tensorboard_logs

# Run TensorBoard
tensorboard --logdir ./tensorboard_logs
```

### Save and Download Models
Models are saved to `/tmp/finrl_models/` during training.

To retrieve:
```bash
# Download trained models
modal volume get finrl-volume /tmp/finrl_models ./trained_models

# List model checkpoints
ls -lh trained_models/
```

---

## 🔧 Configuration Reference

### Modal Function Parameters

```python
@app.function(
    image=image,              # Docker image with dependencies
    gpu="A100",               # GPU type: None, T4, A100, H100
    memory=32768,             # RAM in MB (32GB = 32768)
    timeout=3600,             # Max runtime in seconds (1hr = 3600)
    cpu=8,                    # Number of CPU cores
    mounts=[volume],          # Persistent storage volumes
    secrets=[secret],         # Environment secrets
    retries=2,                # Number of retry attempts
    concurrency_limit=1,      # Max concurrent runs
)
```

### Environment Hyperparameters

**PPO Configuration:**
```python
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=3e-4,       # Learning rate
    n_steps=2048,             # Steps before update
    batch_size=64,            # Batch size for training
    n_epochs=10,              # Gradient steps per update
    gamma=0.99,               # Discount factor
    gae_lambda=0.95,          # GAE parameter
    clip_range=0.2,           # PPO clip range
    verbose=1,
    tensorboard_log="/tmp/finrl_tensorboard/",
    device='cuda'             # Use GPU
)
```

**Training Configuration:**
```python
config = {
    'tickers': [...],                # Stock list
    'start_date': '2020-01-01',     # Training start
    'end_date': '2024-10-01',       # Training end
    'initial_amount': 1000000,       # Portfolio size
    'train_timesteps': 500000,       # Training iterations
    'eval_freq': 10000,             # Evaluation frequency
}
```

---

## 📈 Expected Results

### Standard Training (50 stocks, A100)
**Expected Metrics:**
- Training time: 45-60 minutes
- Final portfolio value: $1.0M - $1.5M (0-50% return)
- Sharpe ratio: 0.5 - 2.0
- Max drawdown: 10-30%
- Trades executed: 500-2000

**Benchmark Comparison:**
- Equal-weight portfolio: Baseline
- RL should achieve: 5-20% excess return
- Risk-adjusted: Better Sharpe than benchmark

### Massive Training (500+ assets, A100)
**Expected Metrics:**
- Training time: 2-3 hours
- Portfolio size: $10M
- Asset coverage: 500+ instruments
- Features per sample: 50+
- State space dimension: 2500+

**Advanced Capabilities:**
- Multi-asset allocation
- Cross-asset correlations
- Regime detection
- Dynamic rebalancing

---

## ✅ Pre-Deployment Checklist

Before deploying to Modal A100 (expensive!), verify:

- [ ] Local testing complete (`finrl_portfolio_system.py` runs)
- [ ] Data download working (tested with `test_data_download.py`)
- [ ] Modal CLI authenticated (`modal token set`)
- [ ] Modal volume created (`finrl-volume`)
- [ ] Secrets configured (if using `massive_finrl_hf.py`)
- [ ] Scripts validated (no import errors)
- [ ] Configuration reviewed (tickers, dates, parameters)
- [ ] Cost estimate acceptable (~$2-15 per run)
- [ ] Monitoring plan in place (logs, metrics)

### Quick Validation Command
```bash
# Run all validation checks
python3 quick_diagnostic.py && \
python3 test_data_download.py && \
modal volume list | grep finrl && \
modal secret list && \
echo "✅ Ready for Modal deployment!"
```

---

## 🎯 Next Steps

### Immediate
1. **Create Modal Volume:**
   ```bash
   modal volume create finrl-volume
   ```

2. **Run Small Test:**
   ```bash
   # Edit modal_finrl_training.py: gpu="T4", 5 stocks, 50K steps
   modal run modal_finrl_training.py::train_large_portfolio
   ```

3. **Monitor Results:**
   ```bash
   modal app logs finrl-portfolio-training --follow
   ```

### Short-term
4. **Full A100 Training:**
   ```bash
   modal run modal_finrl_training.py::train_large_portfolio
   ```

5. **Download Results:**
   ```bash
   modal volume get finrl-volume /tmp/finrl_models ./trained_models
   ```

6. **Analyze Performance:**
   - Review TensorBoard logs
   - Compare to benchmark
   - Evaluate risk metrics

### Long-term
7. **Massive Training:**
   ```bash
   modal run massive_finrl_hf.py::build_massive_dataset
   modal run massive_finrl_hf.py::train_massive_portfolio
   ```

8. **Hyperparameter Tuning:**
   ```bash
   modal run modal_finrl_training.py::hyperparameter_optimization
   ```

9. **Production Deployment:**
   - Schedule periodic retraining
   - Integrate with paper trading
   - Build monitoring dashboard

---

## 📞 Support Resources

- [Modal Documentation](https://modal.com/docs)
- [Modal Discord](https://discord.gg/modal)
- [FinRL Documentation](https://finrl.readthedocs.io/)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)

---

## 🎉 Ready to Deploy?

Your Modal setup is **90% complete**! Just need to:
1. Create the `finrl-volume`
2. Run a small test
3. Deploy to A100

**Recommended First Command:**
```bash
modal volume create finrl-volume && \
echo "✅ Volume created! Ready for deployment."
```

Then run:
```bash
python3 deploy_modal.py
# Choose option 1: Quick local test
# Then choose option 2: Deploy to Modal A100
```

---

**Last Updated:** October 12, 2024
**Status:** Ready for deployment (volume creation pending)
**Estimated Time to Production:** 30 minutes
