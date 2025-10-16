#!/bin/bash

# Production Trading Bot Launcher
# Automatically downloads latest model and starts trading

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║    FinRL Production Trading Bot - Deployment Script           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if Modal is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Install with: pip install modal"
    exit 1
fi

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "📁 Creating models directory..."
    mkdir models
fi

# Check for production model
if [ ! -f "models/finrl_production_100stocks_500k.zip" ]; then
    echo "📥 Production model not found locally"
    echo "   Checking Modal volume..."

    if modal volume ls finrl-volume | grep -q "finrl_production_100stocks_500k.zip"; then
        echo "✅ Found production model on Modal, downloading..."
        modal volume get finrl-volume finrl_production_100stocks_500k.zip models/
        echo "✅ Production model downloaded"
    else
        echo "⚠️  Production model not available yet"

        # Check for enhanced model
        if [ ! -f "models/finrl_enhanced_500k.zip" ]; then
            echo "📥 Downloading enhanced model (50 stocks) as fallback..."
            modal volume get finrl-volume finrl_enhanced_500k.zip models/
            echo "✅ Enhanced model downloaded"
        else
            echo "✅ Using existing enhanced model (50 stocks)"
        fi
    fi
else
    echo "✅ Production model found locally"
fi

# Check .env file
if [ ! -f ".env" ]; then
    echo "❌ .env file not found"
    echo "   Please create .env with Alpaca credentials:"
    echo "   ALPACA_API_KEY=your_key"
    echo "   ALPACA_SECRET_KEY=your_secret"
    echo "   ALPACA_BASE_URL=https://paper-api.alpaca.markets"
    exit 1
fi

# Test Alpaca connection
echo ""
echo "🔌 Testing Alpaca connection..."
python3 -c "
from alpaca_trade_api import REST
from dotenv import load_dotenv
import os

load_dotenv()
api = REST(
    key_id=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    base_url=os.getenv('ALPACA_BASE_URL')
)
account = api.get_account()
print(f'✅ Connected to Alpaca')
print(f'   Account: {account.account_number}')
print(f'   Portfolio Value: \${float(account.portfolio_value):,.2f}')
" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "❌ Failed to connect to Alpaca. Check your .env credentials."
    exit 1
fi

# Check market status
echo ""
echo "📊 Checking market status..."
python3 -c "
from alpaca_trade_api import REST
from dotenv import load_dotenv
import os

load_dotenv()
api = REST(
    key_id=os.getenv('ALPACA_API_KEY'),
    secret_key=os.getenv('ALPACA_SECRET_KEY'),
    base_url=os.getenv('ALPACA_BASE_URL')
)
clock = api.get_clock()
if clock.is_open:
    print('✅ Market is OPEN - trading will begin immediately')
else:
    print(f'⏸️  Market is CLOSED')
    print(f'   Next open: {clock.next_open}')
    print(f'   Bot will wait until market opens')
"

# Ask for confirmation
echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Ready to start production trading bot                        ║"
echo "║                                                                ║"
echo "║  Bot will:                                                     ║"
echo "║  • Trade 100 stocks (or 50 with fallback model)               ║"
echo "║  • Execute trades every 5 minutes during market hours          ║"
echo "║  • Use Alpaca paper trading (no real money)                   ║"
echo "║                                                                ║"
echo "║  Press Ctrl+C to stop trading at any time                     ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
read -p "Start trading? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled by user"
    exit 0
fi

# Start the bot
echo ""
echo "🚀 Starting production trading bot..."
echo ""

python3 alpaca_paper_trading_production.py
