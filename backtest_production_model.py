"""
Backtest Production Model (99 stocks) on Historical Data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from stable_baselines3 import PPO
import yfinance as yf
import stockstats
import matplotlib.pyplot as plt

# 99-stock production list (from Modal training - PXD excluded)
PRODUCTION_TICKERS = [
    # Tech (20 stocks)
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'TSLA', 'AVGO', 'ORCL',
    'ADBE', 'CRM', 'CSCO', 'ACN', 'AMD', 'INTC', 'TXN', 'QCOM', 'IBM', 'INTU',

    # Financials (15 stocks)
    'JPM', 'BAC', 'WFC', 'MS', 'GS', 'BLK', 'C', 'SCHW', 'USB', 'AXP',
    'PNC', 'TFC', 'COF', 'BK', 'STT',

    # Healthcare (15 stocks)
    'UNH', 'JNJ', 'LLY', 'PFE', 'ABBV', 'TMO', 'ABT', 'MRK', 'DHR', 'BMY',
    'AMGN', 'GILD', 'CVS', 'CI', 'ELV',

    # Consumer (15 stocks)
    'WMT', 'HD', 'PG', 'COST', 'KO', 'PEP', 'NKE', 'MCD', 'DIS', 'SBUX',
    'LOW', 'TGT', 'TJX', 'BKNG', 'CMG',

    # Energy (9 stocks - PXD excluded)
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY',

    # Industrials (10 stocks)
    'BA', 'HON', 'UPS', 'RTX', 'LMT', 'CAT', 'DE', 'GE', 'MMM', 'UNP',

    # Utilities & Real Estate (8 stocks)
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'PLD', 'AMT', 'CCI',

    # Materials & Other (7 stocks)
    'LIN', 'APD', 'SHW', 'NEM', 'FCX', 'ECL', 'DD'
]


def download_backtest_data(tickers, start_date, end_date):
    """Download historical data for backtesting"""
    print(f"📥 Downloading data: {start_date.date()} to {end_date.date()}")

    all_data = []
    failed = []

    for i, ticker in enumerate(tickers, 1):
        try:
            print(f"  [{i}/{len(tickers)}] {ticker}...", end=" ", flush=True)
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)

            if len(data) > 0:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = [col[0] if col[1] == '' else col[0] for col in data.columns]

                data['tic'] = ticker
                data = data.reset_index()

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
    """Calculate 20 technical indicators (matching production training)"""
    print("📊 Calculating 20 technical indicators...")

    result_dfs = []

    for ticker in df['tic'].unique():
        ticker_df = df[df['tic'] == ticker].copy()
        date_col = ticker_df['Date'].copy()

        ticker_df.columns = [c.lower() if c not in ['tic', 'Date'] else c for c in ticker_df.columns]
        stock = stockstats.StockDataFrame.retype(ticker_df)

        # 20 indicators (matching Modal training)
        indicators = [
            'macd', 'rsi_6', 'rsi_12', 'rsi_30', 'cci_14', 'cci_30', 'dx_14', 'dx_30',
            'close_5_sma', 'close_10_sma', 'close_20_sma', 'close_50_sma',
            'close_100_sma', 'close_200_sma',
            'boll', 'boll_ub', 'boll_lb', 'atr',
            'wr_14', 'kdjk'
        ]

        for indicator in indicators:
            try:
                _ = stock[indicator]
            except:
                pass

        ticker_df = pd.DataFrame(stock)
        ticker_df['Date'] = date_col.values
        ticker_df.columns = [c.capitalize() if c not in ['tic', 'Date'] else c for c in ticker_df.columns]
        result_dfs.append(ticker_df)

    df_with_tech = pd.concat(result_dfs, ignore_index=True)
    df_with_tech = df_with_tech.dropna()

    print(f"✅ Indicators calculated ({len(df_with_tech):,} rows)\n")
    return df_with_tech


def prepare_state(df, date, tickers, cash, holdings, portfolio_value):
    """Prepare 2181-feature state for 99 stocks"""
    day_data = df[df['Date'] == date]

    # Prices (99)
    prices = []
    for ticker in tickers:
        ticker_data = day_data[day_data['tic'] == ticker]
        if len(ticker_data) > 0:
            price = ticker_data['Close'].iloc[0]
            prices.append(float(price))
        else:
            prices.append(0.0)

    # Technical indicators (20 per stock = 1980)
    tech_cols = [
        'Macd', 'Rsi_6', 'Rsi_12', 'Rsi_30', 'Cci_14', 'Cci_30', 'Dx_14', 'Dx_30',
        'Close_5_sma', 'Close_10_sma', 'Close_20_sma', 'Close_50_sma',
        'Close_100_sma', 'Close_200_sma',
        'Boll', 'Boll_ub', 'Boll_lb', 'Atr',
        'Wr_14', 'Kdjk'
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
    portfolio_val = cash + sum(holdings.get(ticker, 0) * prices[i] for i, ticker in enumerate(tickers))
    cash_ratio = cash / portfolio_val if portfolio_val > 0 else 0
    holdings_list = [holdings.get(ticker, 0.0) for ticker in tickers]
    weekday = date.weekday()

    # Build state (99 + 1980 + 1 + 1 + 99 + 1 = 2181)
    state = np.concatenate([
        prices,           # 99
        tech_indicators,  # 1980
        [cash_ratio],     # 1
        [portfolio_val],  # 1
        holdings_list,    # 99
        [weekday]         # 1
    ])

    return state.astype(np.float32), prices


def backtest(model_path, df, tickers, initial_capital=100000):
    """Run backtest simulation"""
    print("🎯 Loading production model...")
    model = PPO.load(model_path)
    print(f"✅ Model loaded (obs: {model.observation_space.shape}, action: {model.action_space.shape})\n")

    cash = initial_capital
    holdings = {ticker: 0.0 for ticker in tickers}
    portfolio_values = []
    dates = []
    trades_log = []

    trading_dates = sorted(df['Date'].unique())

    print("="*80)
    print(f"BACKTESTING PRODUCTION MODEL (99 STOCKS)")
    print("="*80)
    print(f"Period: {trading_dates[0].date()} to {trading_dates[-1].date()}")
    print(f"Trading Days: {len(trading_dates)}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print("="*80)
    print()

    for day_idx, date in enumerate(trading_dates, 1):
        day_data = df[df['Date'] == date]
        current_prices = {}

        for ticker in tickers:
            ticker_data = day_data[day_data['tic'] == ticker]
            if len(ticker_data) > 0:
                current_prices[ticker] = float(ticker_data['Close'].iloc[0])
            else:
                current_prices[ticker] = 0.0

        stock_value = sum(holdings[t] * current_prices[t] for t in tickers)
        portfolio_value = cash + stock_value

        state, prices = prepare_state(df, date, tickers, cash, holdings, portfolio_value)

        action, _ = model.predict(state, deterministic=True)

        trades_today = 0
        for i, ticker in enumerate(tickers):
            price = prices[i]
            if price <= 0:
                continue

            target_value = (action[i] + 1) / 2 * portfolio_value
            target_shares = target_value / price
            shares_to_trade = target_shares - holdings[ticker]

            if abs(shares_to_trade) < 0.01:
                continue

            if shares_to_trade > 0:  # Buy
                max_shares = cash / (price * 1.001)
                shares_to_trade = min(shares_to_trade, max_shares)
                cost = shares_to_trade * price * 1.001

                if cost > cash:
                    continue

                cash -= cost
                holdings[ticker] += shares_to_trade
                trades_today += 1

            else:  # Sell
                shares_to_trade = max(shares_to_trade, -holdings[ticker])
                proceeds = abs(shares_to_trade) * price * 0.999
                cash += proceeds
                holdings[ticker] += shares_to_trade
                trades_today += 1

        portfolio_values.append(portfolio_value)
        dates.append(date)

        if day_idx % 20 == 0 or day_idx == len(trading_dates):
            pnl = portfolio_value - initial_capital
            pnl_pct = (pnl / initial_capital) * 100
            print(f"Day {day_idx:3d}/{len(trading_dates)} | {date.date()} | "
                  f"Portfolio: ${portfolio_value:,.2f} | P&L: {pnl_pct:+.2f}% | "
                  f"Trades: {trades_today}")

    print()
    print("="*80)

    final_value = portfolio_values[-1]
    total_return = ((final_value - initial_capital) / initial_capital) * 100

    # Benchmark
    aapl_data = df[df['tic'] == 'AAPL'].sort_values('Date')
    aapl_start = aapl_data.iloc[0]['Close']
    aapl_end = aapl_data.iloc[-1]['Close']
    aapl_return = ((aapl_end - aapl_start) / aapl_start) * 100

    # Metrics
    daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252) if len(daily_returns) > 0 else 0
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = np.min(drawdown) * 100

    print("BACKTEST RESULTS - PRODUCTION MODEL (99 STOCKS)")
    print("="*80)
    print(f"Initial Capital:    ${initial_capital:,.2f}")
    print(f"Final Portfolio:    ${final_value:,.2f}")
    print(f"Total Return:       {total_return:+.2f}%")
    print(f"AAPL Buy & Hold:    {aapl_return:+.2f}%")
    print(f"Sharpe Ratio:       {sharpe_ratio:.2f}")
    print(f"Max Drawdown:       {max_drawdown:.2f}%")
    print(f"Total Trades:       {len(trades_log)}")
    print("="*80)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes[0].plot(dates, portfolio_values, linewidth=2, color='blue', label='Portfolio Value')
    axes[0].axhline(y=initial_capital, color='red', linestyle='--', alpha=0.5, label='Initial Capital')
    axes[0].set_title('Production Model (99 Stocks) - Portfolio Value Over Time', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    returns = [(v - initial_capital) / initial_capital * 100 for v in portfolio_values]
    axes[1].plot(dates, returns, linewidth=2, color='green', label='Cumulative Return (%)')
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title('Cumulative Returns', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Return (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('backtest_production_99stocks.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Chart saved: backtest_production_99stocks.png")

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }


if __name__ == "__main__":
    MODEL_PATH = "models/finrl_production_100stocks_500k.zip"
    BACKTEST_MONTHS = 6

    end_date = datetime.now()
    start_date = end_date - timedelta(days=BACKTEST_MONTHS * 30)

    print("="*80)
    print("BACKTESTING PRODUCTION MODEL - 99 STOCKS")
    print("="*80)
    print(f"Model: {MODEL_PATH}")
    print(f"Stocks: {len(PRODUCTION_TICKERS)}")
    print(f"Period: {BACKTEST_MONTHS} months")
    print("="*80)
    print()

    df, successful_tickers = download_backtest_data(PRODUCTION_TICKERS, start_date, end_date)
    df = calculate_technical_indicators(df)
    results = backtest(MODEL_PATH, df, successful_tickers, initial_capital=100000)

    print("\n✅ Backtest complete!")
