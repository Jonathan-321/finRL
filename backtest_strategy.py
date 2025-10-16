"""
Backtest Trading Strategy on Historical Data
Tests the trained model on past data to evaluate performance before live trading
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from stable_baselines3 import PPO
import yfinance as yf
import stockstats
import matplotlib.pyplot as plt

# 50-stock list (same as trading bot)
# BRK.B replaced with GOOG (BRK.B has yfinance issues)
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'GOOG', 'UNH', 'JNJ',
    'V', 'XOM', 'WMT', 'JPM', 'LLY', 'MA', 'PG', 'AVGO', 'HD', 'CVX',
    'MRK', 'ABBV', 'KO', 'PEP', 'COST', 'ADBE', 'MCD', 'TMO', 'CSCO', 'ACN',
    'CRM', 'NFLX', 'ABT', 'LIN', 'DHR', 'AMD', 'NKE', 'TXN', 'DIS', 'PM',
    'VZ', 'INTU', 'CMCSA', 'WFC', 'NEE', 'COP', 'ORCL', 'IBM', 'QCOM', 'RTX'
]


def download_backtest_data(tickers, start_date, end_date):
    """Download historical data for backtesting"""
    print(f"📥 Downloading data: {start_date.date()} to {end_date.date()}")

    all_data = []
    failed = []

    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"  [{i}/{len(tickers)}] {ticker}...", end=" ")
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if len(data) > 0:
                # Flatten columns
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] if col[1] == '' else col[0] for col in data.columns]

                data['tic'] = ticker
                data = data.reset_index()

                # Ensure Close is Series
                if 'Close' in data.columns and isinstance(data['Close'], pd.DataFrame):
                    data['Close'] = data['Close'].iloc[:, 0]

                all_data.append(data)
                print("✓")
            else:
                failed.append(ticker)
                print("✗")
        except Exception as e:
            failed.append(ticker)
            print(f"✗ ({e})")

    if failed:
        print(f"\n⚠️  Failed: {', '.join(failed)}")

    df = pd.concat(all_data, ignore_index=True)
    print(f"✅ Downloaded {len(df):,} rows for {len(tickers) - len(failed)} stocks\n")

    return df, [t for t in tickers if t not in failed]


def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    print("📊 Calculating technical indicators...")

    result_dfs = []

    for ticker in df['tic'].unique():
        ticker_df = df[df['tic'] == ticker].copy()

        # Save Date
        date_col = ticker_df['Date'].copy()

        # Lowercase for stockstats
        ticker_df.columns = [c.lower() if c not in ['tic', 'Date'] else c for c in ticker_df.columns]
        stock = stockstats.StockDataFrame.retype(ticker_df)

        # 13 indicators (matching training)
        indicators = [
            'macd', 'rsi_30', 'cci_30', 'dx_30',
            'close_20_sma', 'close_50_sma', 'close_200_sma',
            'boll', 'boll_ub', 'boll_lb',
            'atr', 'wr_14', 'kdjk'
        ]

        for indicator in indicators:
            try:
                _ = stock[indicator]
            except:
                pass

        # Convert back
        ticker_df = pd.DataFrame(stock)
        ticker_df['Date'] = date_col.values
        ticker_df.columns = [c.capitalize() if c not in ['tic', 'Date'] else c for c in ticker_df.columns]
        result_dfs.append(ticker_df)

    df_with_tech = pd.concat(result_dfs, ignore_index=True)
    df_with_tech = df_with_tech.dropna()

    print(f"✅ Indicators calculated ({len(df_with_tech):,} rows)\n")
    return df_with_tech


def prepare_state(df, date, tickers, cash, holdings, portfolio_value):
    """Prepare state observation for the model"""
    day_data = df[df['Date'] == date]

    # Prices
    prices = []
    for ticker in tickers:
        ticker_data = day_data[day_data['tic'] == ticker]
        if len(ticker_data) > 0:
            price = ticker_data['Close'].iloc[0]
            prices.append(float(price))
        else:
            prices.append(0.0)

    # Technical indicators (13 per stock)
    tech_cols = [
        'Macd', 'Rsi_30', 'Cci_30', 'Dx_30',
        'Close_20_sma', 'Close_50_sma', 'Close_200_sma',
        'Boll', 'Boll_ub', 'Boll_lb',
        'Atr', 'Wr_14', 'Kdjk'
    ]

    tech_indicators = []
    for col in tech_cols:
        for ticker in tickers:
            ticker_data = day_data[day_data['tic'] == ticker]
            if len(ticker_data) > 0 and col in ticker_data.columns:
                val = ticker_data[col].iloc[0]
                tech_indicators.append(float(val) if not pd.isna(val) else 0.0)
            else:
                tech_indicators.append(0.0)

    # Portfolio state
    cash_ratio = cash / portfolio_value if portfolio_value > 0 else 0
    holdings_list = [holdings.get(ticker, 0.0) for ticker in tickers]
    weekday = date.weekday()

    # Build state (753 features)
    state = np.concatenate([
        prices,                 # 50
        tech_indicators,        # 650
        [cash_ratio],           # 1
        [portfolio_value],      # 1
        holdings_list,          # 50
        [weekday]               # 1
    ])

    return state.astype(np.float32), prices


def backtest(model_path, df, tickers, initial_capital=100000):
    """Run backtest simulation"""
    print("🎯 Loading model...")
    model = PPO.load(model_path)
    print(f"✅ Model loaded\n")

    # Initialize portfolio
    cash = initial_capital
    holdings = {ticker: 0.0 for ticker in tickers}
    portfolio_values = []
    dates = []
    trades_log = []

    # Get trading dates
    trading_dates = sorted(df['Date'].unique())

    print("="*80)
    print(f"BACKTESTING SIMULATION")
    print("="*80)
    print(f"Period: {trading_dates[0].date()} to {trading_dates[-1].date()}")
    print(f"Trading Days: {len(trading_dates)}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print("="*80)
    print()

    # Simulate trading
    for day_idx, date in enumerate(trading_dates, 1):
        # Calculate current portfolio value
        day_data = df[df['Date'] == date]
        current_prices = {}

        for ticker in tickers:
            ticker_data = day_data[day_data['tic'] == ticker]
            if len(ticker_data) > 0:
                current_prices[ticker] = float(ticker_data['Close'].iloc[0])
            else:
                current_prices[ticker] = 0.0

        # Portfolio value
        stock_value = sum(holdings[t] * current_prices[t] for t in tickers)
        portfolio_value = cash + stock_value

        # Prepare state
        state, prices = prepare_state(df, date, tickers, cash, holdings, portfolio_value)

        # Get model action
        action, _ = model.predict(state, deterministic=True)

        # Execute trades
        trades_today = 0
        for i, ticker in enumerate(tickers):
            price = prices[i]
            if price <= 0:
                continue

            # Convert action to target shares
            target_value = (action[i] + 1) / 2 * portfolio_value
            target_shares = target_value / price

            # Calculate trade
            shares_to_trade = target_shares - holdings[ticker]

            if abs(shares_to_trade) < 0.01:
                continue

            # Transaction cost (0.1%)
            if shares_to_trade > 0:  # Buy
                max_shares = cash / (price * 1.001)
                shares_to_trade = min(shares_to_trade, max_shares)
                cost = shares_to_trade * price * 1.001

                if cost > cash:
                    continue

                cash -= cost
                holdings[ticker] += shares_to_trade
                trades_today += 1
                trades_log.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'BUY',
                    'shares': shares_to_trade,
                    'price': price,
                    'cost': cost
                })

            else:  # Sell
                shares_to_trade = max(shares_to_trade, -holdings[ticker])
                proceeds = abs(shares_to_trade) * price * 0.999
                cash += proceeds
                holdings[ticker] += shares_to_trade
                trades_today += 1
                trades_log.append({
                    'date': date,
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': abs(shares_to_trade),
                    'price': price,
                    'proceeds': proceeds
                })

        # Record portfolio value
        portfolio_values.append(portfolio_value)
        dates.append(date)

        # Progress update
        if day_idx % 20 == 0 or day_idx == len(trading_dates):
            pnl = portfolio_value - initial_capital
            pnl_pct = (pnl / initial_capital) * 100
            print(f"Day {day_idx:3d}/{len(trading_dates)} | {date.date()} | "
                  f"Portfolio: ${portfolio_value:,.2f} | P&L: {pnl_pct:+.2f}% | "
                  f"Trades: {trades_today}")

    print()
    print("="*80)

    # Calculate statistics
    final_value = portfolio_values[-1]
    total_return = ((final_value - initial_capital) / initial_capital) * 100

    # Buy and hold benchmark (just AAPL)
    aapl_data = df[df['tic'] == 'AAPL'].sort_values('Date')
    aapl_start = aapl_data.iloc[0]['Close']
    aapl_end = aapl_data.iloc[-1]['Close']
    aapl_return = ((aapl_end - aapl_start) / aapl_start) * 100

    # Daily returns
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252) if len(daily_returns) > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = np.min(drawdown) * 100

    print("BACKTEST RESULTS")
    print("="*80)
    print(f"Initial Capital:    ${initial_capital:,.2f}")
    print(f"Final Portfolio:    ${final_value:,.2f}")
    print(f"Total Return:       {total_return:+.2f}%")
    print(f"AAPL Buy & Hold:    {aapl_return:+.2f}%")
    print(f"Sharpe Ratio:       {sharpe_ratio:.2f}")
    print(f"Max Drawdown:       {max_drawdown:.2f}%")
    print(f"Total Trades:       {len(trades_log)}")
    print(f"Avg Trades/Day:     {len(trades_log)/len(trading_dates):.1f}")
    print("="*80)

    # Plot results
    plot_backtest_results(dates, portfolio_values, initial_capital, trades_log, tickers)

    return {
        'portfolio_values': portfolio_values,
        'dates': dates,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'trades': trades_log
    }


def plot_backtest_results(dates, portfolio_values, initial_capital, trades_log, tickers):
    """Plot backtest results"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Portfolio value over time
    axes[0].plot(dates, portfolio_values, linewidth=2, color='blue', label='Portfolio Value')
    axes[0].axhline(y=initial_capital, color='red', linestyle='--', alpha=0.5, label='Initial Capital')
    axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Returns
    returns = [(v - initial_capital) / initial_capital * 100 for v in portfolio_values]
    axes[1].plot(dates, returns, linewidth=2, color='green', label='Cumulative Return (%)')
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title('Cumulative Returns', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Return (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Chart saved: backtest_results.png")

    # Trade analysis
    if trades_log:
        trades_df = pd.DataFrame(trades_log)
        print(f"\n📈 Most Traded Stocks:")
        print(trades_df['ticker'].value_counts().head(10))
        print(f"\n💰 Buy vs Sell:")
        print(trades_df['action'].value_counts())


if __name__ == "__main__":
    import sys

    # Configuration - check command line args
    if len(sys.argv) > 1 and sys.argv[1] == '--model':
        MODEL_PATH = sys.argv[2]
    else:
        MODEL_PATH = "models/paper_trading_model_with_tech.zip"

    BACKTEST_MONTHS = 6  # Test on last 6 months

    # Date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=BACKTEST_MONTHS * 30)

    print("="*80)
    print("BACKTESTING FINRL TRADING STRATEGY")
    print("="*80)
    print(f"Model: {MODEL_PATH}")
    print(f"Stocks: {len(TICKERS)}")
    print(f"Period: {BACKTEST_MONTHS} months")
    print("="*80)
    print()

    # Download data
    df, successful_tickers = download_backtest_data(TICKERS, start_date, end_date)

    # Calculate indicators
    df = calculate_technical_indicators(df)

    # Run backtest
    results = backtest(MODEL_PATH, df, successful_tickers, initial_capital=100000)

    print("\n✅ Backtest complete!")
