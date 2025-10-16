"""
Production Alpaca Paper Trading Bot
- Supports 100 stocks with 20 technical indicators
- Dynamic model/feature detection
- Enhanced monitoring and risk management
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca_trade_api import REST, TimeFrame
from stable_baselines3 import PPO
import yfinance as yf
from dotenv import load_dotenv
import stockstats

# Load environment variables
load_dotenv()

# 100-stock production portfolio
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

    # Energy (10 stocks)
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'PSX', 'VLO', 'OXY',

    # Industrials (10 stocks)
    'BA', 'HON', 'UPS', 'RTX', 'LMT', 'CAT', 'DE', 'GE', 'MMM', 'UNP',

    # Utilities & Real Estate (8 stocks)
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'PLD', 'AMT', 'CCI',

    # Materials & Other (7 stocks)
    'LIN', 'APD', 'SHW', 'NEM', 'FCX', 'ECL', 'DD'
]


class ProductionTradingBot:
    """
    Production paper trading bot with 100 stocks
    """

    def __init__(self, model_path, tickers=None, initial_capital=100000):
        """
        Initialize production trading bot

        Args:
            model_path: Path to trained PPO model
            tickers: List of tickers (defaults to PRODUCTION_TICKERS)
            initial_capital: Starting capital
        """
        # Alpaca API
        self.api = REST(
            key_id=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        )

        # Load model
        print(f"Loading model from {model_path}...")
        self.model = PPO.load(model_path)

        # Configuration
        self.tickers = tickers if tickers else PRODUCTION_TICKERS
        self.initial_capital = initial_capital
        self.current_holdings = {ticker: 0 for ticker in self.tickers}
        self.cash = initial_capital

        # Detect model type based on observation space
        obs_dim = self.model.observation_space.shape[0]
        n_stocks = len(self.tickers)

        # Calculate expected dimensions for different configurations
        # Old: 753 = 50 + 650 (50*13) + 2 + 50 + 1
        # Production: 2203 = 100 + 2000 (100*20) + 2 + 100 + 1

        if obs_dim == 753:
            self.n_indicators = 13
            self.indicator_cols = [
                'Macd', 'Rsi_30', 'Cci_30', 'Dx_30',
                'Close_20_sma', 'Close_50_sma', 'Close_200_sma',
                'Boll', 'Boll_ub', 'Boll_lb',
                'Atr', 'Wr_14', 'Kdjk'
            ]
        elif obs_dim == 2203:
            self.n_indicators = 20
            self.indicator_cols = [
                'Macd', 'Rsi_6', 'Rsi_12', 'Rsi_30', 'Cci_14', 'Cci_30', 'Dx_14', 'Dx_30',
                'Close_5_sma', 'Close_10_sma', 'Close_20_sma', 'Close_50_sma',
                'Close_100_sma', 'Close_200_sma',
                'Boll', 'Boll_ub', 'Boll_lb', 'Atr',
                'Wr_14', 'Kdjk'
            ]
        else:
            # Auto-detect
            # obs_dim = n_stocks + (n_stocks * n_indicators) + 2 + n_stocks + 1
            # obs_dim = 2*n_stocks + n_stocks*n_indicators + 3
            # obs_dim - 3 - 2*n_stocks = n_stocks * n_indicators
            self.n_indicators = (obs_dim - 3 - 2*n_stocks) // n_stocks
            print(f"⚠️  Auto-detected {self.n_indicators} indicators per stock")

            # Default to expanded set
            self.indicator_cols = [
                'Macd', 'Rsi_6', 'Rsi_12', 'Rsi_30', 'Cci_14', 'Cci_30', 'Dx_14', 'Dx_30',
                'Close_5_sma', 'Close_10_sma', 'Close_20_sma', 'Close_50_sma',
                'Close_100_sma', 'Close_200_sma',
                'Boll', 'Boll_ub', 'Boll_lb', 'Atr',
                'Wr_14', 'Kdjk'
            ][:self.n_indicators]

        # Verify connection
        self._verify_connection()

        print(f"✅ Production Trading Bot Initialized")
        print(f"   Stocks: {len(self.tickers)}")
        print(f"   Indicators: {self.n_indicators} per stock")
        print(f"   Observation dim: {obs_dim}")
        print(f"   Capital: ${initial_capital:,.2f}")

    def _verify_connection(self):
        """Test Alpaca API connection"""
        try:
            account = self.api.get_account()
            print(f"✅ Connected to Alpaca Paper Trading")
            print(f"   Account: {account.account_number}")
            print(f"   Buying Power: ${float(account.buying_power):,.2f}")
            print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Alpaca: {e}")

    def get_market_data(self, lookback_days=5):
        """
        Fetch recent market data

        Args:
            lookback_days: Days of historical data

        Returns:
            DataFrame with OHLCV data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        all_data = []
        failed_tickers = []

        for ticker in self.tickers:
            try:
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False
                )

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
            except Exception as e:
                print(f"⚠️  Failed to fetch {ticker}: {e}")
                failed_tickers.append(ticker)

        if not all_data:
            raise ValueError("No market data retrieved")

        df = pd.concat(all_data, ignore_index=True)
        return df

    def calculate_technical_indicators(self, df):
        """
        Calculate technical indicators

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with indicators
        """
        result_dfs = []

        for ticker in df['tic'].unique():
            ticker_df = df[df['tic'] == ticker].copy()

            # Save Date
            date_col = ticker_df['Date'].copy() if 'Date' in ticker_df.columns else None

            # Lowercase for stockstats
            ticker_df.columns = [c.lower() if c not in ['tic', 'Date'] else c for c in ticker_df.columns]
            stock = stockstats.StockDataFrame.retype(ticker_df)

            # Calculate all indicators
            lowercase_indicators = [col.lower() for col in self.indicator_cols]

            for indicator in lowercase_indicators:
                try:
                    _ = stock[indicator]
                except:
                    pass

            # Convert back
            ticker_df = pd.DataFrame(stock)

            # Restore Date
            if date_col is not None:
                ticker_df['Date'] = date_col.values

            ticker_df.columns = [c.capitalize() if c not in ['tic', 'Date'] else c for c in ticker_df.columns]
            result_dfs.append(ticker_df)

        return pd.concat(result_dfs, ignore_index=True)

    def prepare_state(self, df):
        """
        Convert market data to model input state

        Args:
            df: DataFrame with prices and indicators

        Returns:
            numpy array representing market state
        """
        # Get latest data point for each ticker
        latest = df.groupby('tic').tail(1).sort_values('tic')

        # Extract prices
        prices = []
        for ticker in self.tickers:
            ticker_data = latest[latest['tic'] == ticker]
            if len(ticker_data) > 0:
                price_val = ticker_data['Close'].iloc[0]
                if isinstance(price_val, (list, tuple, np.ndarray)):
                    price_val = price_val[0]
                prices.append(float(price_val))
            else:
                prices.append(0.0)

        # Extract technical indicators
        tech_indicators = []
        for col in self.indicator_cols:
            for ticker in self.tickers:
                ticker_data = latest[latest['tic'] == ticker]
                if len(ticker_data) > 0 and col in ticker_data.columns:
                    val = ticker_data[col].iloc[0]
                    tech_indicators.append(float(val) if not pd.isna(val) else 0.0)
                else:
                    tech_indicators.append(0.0)

        # Portfolio state
        portfolio_value = self.cash + sum(
            self.current_holdings[ticker] * prices[i]
            for i, ticker in enumerate(self.tickers)
        )

        cash_ratio = self.cash / portfolio_value if portfolio_value > 0 else 0
        holdings = [self.current_holdings[ticker] for ticker in self.tickers]

        # Build state
        state_list = []
        state_list.extend(prices)           # n_stocks
        state_list.extend(tech_indicators)  # n_stocks * n_indicators
        state_list.append(cash_ratio)       # 1
        state_list.append(portfolio_value)  # 1
        state_list.extend(holdings)         # n_stocks
        state_list.append(float(datetime.now().weekday()))  # 1

        state = np.array(state_list, dtype=np.float32)

        return state

    def execute_trades(self, actions, current_prices):
        """
        Execute trades on Alpaca

        Args:
            actions: Model action outputs
            current_prices: Current prices for each stock
        """
        print(f"\n📊 Executing trades at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        trades_executed = 0

        for i, ticker in enumerate(self.tickers):
            action = actions[i]
            price = current_prices[i]
            current_shares = self.current_holdings[ticker]

            if price <= 0:
                continue

            # Convert action to target shares
            max_shares = min(100, int(self.cash / price)) if price > 0 else 0
            target_shares = int((action + 1) / 2 * max_shares)

            # Calculate order size
            order_size = target_shares - current_shares

            if abs(order_size) < 1:
                continue

            try:
                if order_size > 0:
                    # BUY
                    self.api.submit_order(
                        symbol=ticker,
                        qty=order_size,
                        side='buy',
                        type='market',
                        time_in_force='day'
                    )
                    print(f"  🟢 BUY {order_size} {ticker} @ ${price:.2f}")
                    self.cash -= order_size * price * 1.001
                    self.current_holdings[ticker] += order_size
                    trades_executed += 1

                elif order_size < 0:
                    # SELL
                    self.api.submit_order(
                        symbol=ticker,
                        qty=abs(order_size),
                        side='sell',
                        type='market',
                        time_in_force='day'
                    )
                    print(f"  🔴 SELL {abs(order_size)} {ticker} @ ${price:.2f}")
                    self.cash += abs(order_size) * price * 0.999
                    self.current_holdings[ticker] += order_size
                    trades_executed += 1

            except Exception as e:
                print(f"  ⚠️  Failed to trade {ticker}: {e}")

        if trades_executed == 0:
            print("  ⏸️  No trades executed (holding)")

    def get_portfolio_value(self):
        """Get current portfolio value"""
        try:
            account = self.api.get_account()
            return float(account.portfolio_value)
        except:
            return 0.0

    def run(self, interval_minutes=5, max_iterations=None):
        """
        Run trading bot continuously

        Args:
            interval_minutes: Minutes between trading decisions
            max_iterations: Max trading cycles (None = infinite)
        """
        print(f"\n🚀 Starting production paper trading bot")
        print(f"   Interval: {interval_minutes} minutes")
        print(f"   Stocks: {len(self.tickers)}")
        print(f"   Market hours: 9:30 AM - 4:00 PM ET")
        print("   Press Ctrl+C to stop\n")

        iteration = 0

        try:
            while max_iterations is None or iteration < max_iterations:
                iteration += 1

                # Check market status
                clock = self.api.get_clock()
                if not clock.is_open:
                    print(f"⏸️  Market closed. Next open: {clock.next_open}")
                    time.sleep(300)
                    continue

                # Trading cycle
                print(f"\n{'='*60}")
                print(f"Iteration {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*60}")

                # Get data
                df = self.get_market_data(lookback_days=5)
                df = self.calculate_technical_indicators(df)

                # Prepare state
                state = self.prepare_state(df)

                # Get model prediction
                actions, _ = self.model.predict(state, deterministic=True)

                # Get current prices
                latest = df.groupby('tic').tail(1)
                current_prices = [
                    latest[latest['tic']==ticker]['Close'].values[0] if ticker in latest['tic'].values else 0.0
                    for ticker in self.tickers
                ]

                # Execute trades
                self.execute_trades(actions, current_prices)

                # Report status
                portfolio_value = self.get_portfolio_value()
                pnl = portfolio_value - self.initial_capital
                pnl_pct = (pnl / self.initial_capital) * 100

                print(f"\n💰 Portfolio Value: ${portfolio_value:,.2f}")
                print(f"   Cash: ${self.cash:,.2f}")
                print(f"   P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")

                # Wait
                print(f"\n⏳ Waiting {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            print("\n\n🛑 Trading bot stopped by user")
            self._print_final_report()

    def _print_final_report(self):
        """Print final performance summary"""
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()

            final_value = float(account.portfolio_value)
            pnl = final_value - self.initial_capital
            pnl_pct = (pnl / self.initial_capital) * 100

            print(f"\n{'='*60}")
            print("FINAL TRADING REPORT")
            print(f"{'='*60}")
            print(f"Portfolio Value: ${final_value:,.2f}")
            print(f"Cash: ${float(account.cash):,.2f}")
            print(f"P&L: ${pnl:,.2f}")
            print(f"Return: {pnl_pct:+.2f}%")
            print(f"\nOpen Positions: {len(positions)}")
            for pos in positions[:10]:  # Show first 10
                print(f"  {pos.symbol}: {pos.qty} shares @ ${float(pos.current_price):.2f}")
            if len(positions) > 10:
                print(f"  ... and {len(positions)-10} more")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"Could not generate final report: {e}")


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "models/finrl_production_100stocks_500k.zip"  # Production model

    # Fallback to enhanced model if production not available
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = "models/finrl_enhanced_500k.zip"
        print(f"⚠️  Production model not found, using enhanced model")

    # Create and run bot
    bot = ProductionTradingBot(
        model_path=MODEL_PATH,
        tickers=PRODUCTION_TICKERS,
        initial_capital=100000
    )

    # Run with 5-minute intervals
    bot.run(interval_minutes=5)
