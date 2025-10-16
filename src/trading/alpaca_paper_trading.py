"""
Alpaca Paper Trading Bot for FinRL Model
Integrates trained PPO model with Alpaca paper trading API
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

# Load environment variables
load_dotenv()

class AlpacaTradingBot:
    """
    Paper trading bot that uses trained FinRL PPO model
    Executes trades on Alpaca paper trading account
    """

    def __init__(self, model_path, tickers, initial_capital=100000):
        """
        Initialize the trading bot

        Args:
            model_path: Path to trained PPO model (.zip file)
            tickers: List of stock tickers to trade
            initial_capital: Starting portfolio value
        """
        # Alpaca credentials from .env
        self.api = REST(
            key_id=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        )

        # Load trained model
        print(f"Loading model from {model_path}...")
        self.model = PPO.load(model_path)

        # Trading configuration
        self.tickers = tickers
        self.initial_capital = initial_capital
        self.current_holdings = {ticker: 0 for ticker in tickers}
        self.cash = initial_capital

        # Verify connection
        self._verify_connection()

        print(f"✅ Trading bot initialized")
        print(f"   Portfolio: {len(tickers)} stocks")
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
        Fetch recent market data for all tickers

        Args:
            lookback_days: Number of days of historical data

        Returns:
            DataFrame with OHLCV data
        """
        # Get recent daily data (not 5-minute since model trained on daily)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        all_data = []
        for ticker in self.tickers:
            try:
                # Get daily data from yfinance
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False
                )

                if len(data) > 0:
                    # Flatten multi-level columns if present
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = [col[0] if col[1] == '' else col[0] for col in data.columns]
                    # Add ticker column
                    data['tic'] = ticker
                    data = data.reset_index()
                    # Ensure Close is a simple Series
                    if 'Close' in data.columns and isinstance(data['Close'], pd.DataFrame):
                        data['Close'] = data['Close'].iloc[:, 0]
                    all_data.append(data)
            except Exception as e:
                print(f"⚠️  Failed to fetch {ticker}: {e}")

        if not all_data:
            raise ValueError("No market data retrieved")

        df = pd.concat(all_data, ignore_index=True)
        return df

    def calculate_technical_indicators(self, df):
        """
        Add technical indicators to market data

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with technical indicators added
        """
        import stockstats

        # Group by ticker and calculate indicators for each
        result_dfs = []
        for ticker in df['tic'].unique():
            ticker_df = df[df['tic'] == ticker].copy()

            # Prepare for stockstats (needs lowercase column names)
            ticker_df.columns = [c.lower() if c != 'tic' else c for c in ticker_df.columns]

            stock = stockstats.StockDataFrame.retype(ticker_df)

            # Calculate indicators
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

            # Convert back and restore original column names
            ticker_df = pd.DataFrame(stock)
            ticker_df.columns = [c.capitalize() if c != 'tic' else c for c in ticker_df.columns]
            result_dfs.append(ticker_df)

        return pd.concat(result_dfs, ignore_index=True)

    def prepare_state(self, df):
        """
        Convert market data to model input state

        Args:
            df: DataFrame with prices and technical indicators

        Returns:
            numpy array representing current market state (753 features)
        """
        # Get latest data point for each ticker
        latest = df.groupby('tic').tail(1).sort_values('tic')

        # Extract features in correct order
        prices = []
        for ticker in self.tickers:
            ticker_data = latest[latest['tic'] == ticker]
            if len(ticker_data) > 0:
                price_series = ticker_data['Close']
                if hasattr(price_series, 'iloc'):
                    price_val = price_series.iloc[0]
                else:
                    price_val = price_series[0]
                if isinstance(price_val, (list, tuple, np.ndarray)):
                    price_val = price_val[0]
                prices.append(float(price_val))
            else:
                prices.append(0.0)

        # Technical indicators (13 indicators × 50 stocks) - capitalized names
        tech_cols = [
            'Macd', 'Rsi_30', 'Cci_30', 'Dx_30',
            'Close_20_sma', 'Close_50_sma', 'Close_200_sma',
            'Boll', 'Boll_ub', 'Boll_lb',
            'Atr', 'Wr_14', 'Kdjk'
        ]

        tech_indicators = []
        for col in tech_cols:
            for ticker in self.tickers:
                ticker_data = latest[latest['tic'] == ticker]
                if len(ticker_data) > 0 and col in ticker_data.columns:
                    val = ticker_data[col].iloc[0]
                    if pd.isna(val):
                        tech_indicators.append(0.0)
                    else:
                        tech_indicators.append(float(val))
                else:
                    tech_indicators.append(0.0)

        # Portfolio state
        portfolio_value = self.cash + sum(
            self.current_holdings[ticker] * prices[i]
            for i, ticker in enumerate(self.tickers)
        )

        cash_ratio = self.cash / portfolio_value if portfolio_value > 0 else 0
        holdings = [self.current_holdings[ticker] for ticker in self.tickers]

        # State: prices (50) + tech (650) + cash_ratio + portfolio_value + holdings (50) + weekday = 753
        state_list = []
        state_list.extend(prices)
        state_list.extend(tech_indicators)
        state_list.append(cash_ratio)
        state_list.append(portfolio_value)
        state_list.extend(holdings)
        state_list.append(float(datetime.now().weekday()))

        state = np.array(state_list, dtype=np.float32)

        return state

    def execute_trades(self, actions, current_prices):
        """
        Convert model actions to Alpaca orders

        Args:
            actions: Array of action values [-1, +1] for each stock
            current_prices: Current price for each stock
        """
        print(f"\n📊 Executing trades at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        for i, ticker in enumerate(self.tickers):
            action = actions[i]
            price = current_prices[i]
            current_shares = self.current_holdings[ticker]

            # Convert continuous action to target shares
            # action = -1 (sell all), 0 (hold), +1 (buy max)
            max_shares = min(100, int(self.cash / price)) if price > 0 else 0
            target_shares = int((action + 1) / 2 * max_shares)  # Scale [-1,1] to [0, max]

            # Calculate order size
            order_size = target_shares - current_shares

            if abs(order_size) < 1:
                continue  # Skip small orders

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
                    self.cash -= order_size * price * 1.001  # Include 0.1% fee
                    self.current_holdings[ticker] += order_size

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
                    self.cash += abs(order_size) * price * 0.999  # Subtract 0.1% fee
                    self.current_holdings[ticker] += order_size

            except Exception as e:
                print(f"  ⚠️  Failed to trade {ticker}: {e}")

    def get_portfolio_value(self):
        """Get current portfolio value from Alpaca"""
        try:
            account = self.api.get_account()
            return float(account.portfolio_value)
        except:
            return 0.0

    def run(self, interval_minutes=5, max_iterations=None):
        """
        Run the trading bot continuously

        Args:
            interval_minutes: Minutes between trading decisions
            max_iterations: Maximum number of trading cycles (None = infinite)
        """
        print(f"\n🚀 Starting paper trading bot")
        print(f"   Interval: {interval_minutes} minutes")
        print(f"   Market hours: 9:30 AM - 4:00 PM ET")
        print("   Press Ctrl+C to stop\n")

        iteration = 0

        try:
            while max_iterations is None or iteration < max_iterations:
                iteration += 1

                # Check if market is open
                clock = self.api.get_clock()
                if not clock.is_open:
                    print(f"⏸️  Market closed. Next open: {clock.next_open}")
                    time.sleep(300)  # Check again in 5 minutes
                    continue

                # Get market data
                print(f"\n{'='*60}")
                print(f"Iteration {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*60}")

                df = self.get_market_data(lookback_days=5)
                df = self.calculate_technical_indicators(df)

                # Prepare state
                state = self.prepare_state(df)

                # Get model prediction
                actions, _ = self.model.predict(state, deterministic=True)

                # Get current prices
                latest = df.groupby('tic').tail(1)
                current_prices = [
                    latest[latest['tic']==ticker]['Close'].values[0]
                    for ticker in self.tickers
                ]

                # Execute trades
                self.execute_trades(actions, current_prices)

                # Report status
                portfolio_value = self.get_portfolio_value()
                print(f"\n💰 Portfolio Value: ${portfolio_value:,.2f}")
                print(f"   Cash: ${self.cash:,.2f}")
                print(f"   P&L: ${portfolio_value - self.initial_capital:,.2f} ({(portfolio_value/self.initial_capital - 1)*100:.2f}%)")

                # Wait for next interval
                print(f"\n⏳ Waiting {interval_minutes} minutes until next cycle...")
                time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            print("\n\n🛑 Trading bot stopped by user")
            self._print_final_report()

    def _print_final_report(self):
        """Print final performance summary"""
        try:
            account = self.api.get_account()
            positions = self.api.list_positions()

            print(f"\n{'='*60}")
            print("FINAL TRADING REPORT")
            print(f"{'='*60}")
            print(f"Portfolio Value: ${float(account.portfolio_value):,.2f}")
            print(f"Cash: ${float(account.cash):,.2f}")
            print(f"P&L: ${float(account.portfolio_value) - self.initial_capital:,.2f}")
            print(f"Return: {(float(account.portfolio_value)/self.initial_capital - 1)*100:.2f}%")
            print(f"\nOpen Positions: {len(positions)}")
            for pos in positions:
                print(f"  {pos.symbol}: {pos.qty} shares @ ${float(pos.current_price):.2f}")
            print(f"{'='*60}\n")
        except Exception as e:
            print(f"Could not generate final report: {e}")


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "models/paper_trading_model_with_tech.zip"

    # 50 S&P 500 stocks (same as training)
    TICKERS = [
        'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'TSLA', 'GOOG', 'META', 'UNH', 'XOM',
        'LLY', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'ABBV', 'AVGO',
        'PFE', 'KO', 'BAC', 'TMO', 'COST', 'MRK', 'WMT', 'ACN', 'LIN', 'ABT',
        'CSCO', 'DIS', 'CRM', 'DHR', 'VZ', 'ADBE', 'TXN', 'NEE', 'NKE', 'ORCL',
        'PM', 'RTX', 'CMCSA', 'T', 'WFC', 'SPGI', 'AMD', 'LOW', 'HON', 'UPS'
    ]

    # Create and run bot
    bot = AlpacaTradingBot(
        model_path=MODEL_PATH,
        tickers=TICKERS,
        initial_capital=100000
    )

    # Run with 5-minute intervals
    bot.run(interval_minutes=5)
