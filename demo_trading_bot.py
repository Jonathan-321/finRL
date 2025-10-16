"""
Fully Simulated Trading Demo
Runs immediately with pre-downloaded historical data
No market hours restrictions - pure simulation
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time

class SimulatedTradingBot:
    """
    Simulated trading bot using historical data
    Replays past market data as if trading in real-time
    """

    def __init__(self, tickers, initial_capital=100000):
        self.tickers = tickers
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.holdings = {ticker: 0 for ticker in tickers}
        self.portfolio_history = []

        print(f"🚀 Initializing Simulated Trading Bot")
        print(f"   Portfolio: {len(tickers)} stocks")
        print(f"   Capital: ${initial_capital:,.2f}\n")

    def download_historical_data(self, start_date, end_date):
        """Download historical data for backtesting"""
        print(f"📊 Downloading historical data...")
        print(f"   Period: {start_date} to {end_date}")

        all_data = []
        successful = []

        for ticker in self.tickers:
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if len(df) > 0:
                    df['ticker'] = ticker
                    # Ensure Close is float
                    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
                    all_data.append(df.reset_index())
                    successful.append(ticker)
            except:
                pass

        if not all_data:
            raise ValueError("No data downloaded")

        combined = pd.concat(all_data, ignore_index=True)
        print(f"   ✅ Downloaded {len(combined)} data points for {len(successful)} stocks\n")

        return combined

    def make_simple_decision(self, current_prices, prev_prices):
        """
        Simple momentum strategy (for demo purposes)
        Buy stocks that went up, sell stocks that went down
        """
        actions = []

        for ticker in self.tickers:
            if ticker not in current_prices or ticker not in prev_prices:
                actions.append(0)  # Hold
                continue

            curr_price = current_prices[ticker]
            prev_price = prev_prices[ticker]

            # Simple momentum: buy if up, sell if down
            if curr_price > prev_price * 1.01:  # Up >1%
                actions.append(1)  # Buy signal
            elif curr_price < prev_price * 0.99:  # Down >1%
                actions.append(-1)  # Sell signal
            else:
                actions.append(0)  # Hold

        return actions

    def execute_simulated_trades(self, actions, prices, timestamp):
        """Execute trades in simulation"""
        trades = []

        for i, ticker in enumerate(self.tickers):
            if ticker not in prices:
                continue

            action = actions[i]
            price = prices[ticker]
            current_shares = self.holdings[ticker]

            if action > 0 and self.cash > price:  # Buy
                max_shares = int(self.cash / price / len(self.tickers))  # Limit per stock
                if max_shares > 0:
                    cost = max_shares * price * 1.001  # 0.1% fee
                    if cost <= self.cash:
                        self.cash -= cost
                        self.holdings[ticker] += max_shares
                        trades.append(f"🟢 BUY {max_shares} {ticker} @ ${price:.2f}")

            elif action < 0 and current_shares > 0:  # Sell
                proceeds = current_shares * price * 0.999  # 0.1% fee
                self.cash += proceeds
                self.holdings[ticker] = 0
                trades.append(f"🔴 SELL {current_shares} {ticker} @ ${price:.2f}")

        # Calculate portfolio value
        portfolio_value = self.cash + sum(
            self.holdings[ticker] * prices.get(ticker, 0)
            for ticker in self.tickers
        )

        self.portfolio_history.append({
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'num_positions': sum(1 for h in self.holdings.values() if h > 0)
        })

        return trades, portfolio_value

    def run_simulation(self, data, interval_minutes=5, speed_multiplier=1):
        """
        Run trading simulation on historical data

        Args:
            data: Historical OHLCV data
            interval_minutes: Trading frequency
            speed_multiplier: How fast to run (1=real-time, 10=10x speed)
        """
        print("🎮 Starting Simulated Trading")
        print("="*60)
        print(f"Speed: {speed_multiplier}x real-time")
        print(f"Interval: Every {interval_minutes} minutes")
        print("Press Ctrl+C to stop\n")

        # Group by date
        data['Date'] = pd.to_datetime(data['Date'])
        dates = sorted(data['Date'].unique())

        prev_prices = {}
        iteration = 0

        try:
            for i in range(1, len(dates)):
                iteration += 1
                current_date = dates[i]
                prev_date = dates[i-1]

                # Get prices for current and previous dates
                current_data = data[data['Date'] == current_date]
                prev_data = data[data['Date'] == prev_date]

                current_prices = dict(zip(current_data['ticker'], current_data['Close'].astype(float)))
                prev_prices = dict(zip(prev_data['ticker'], prev_data['Close'].astype(float)))

                # Make trading decision
                actions = self.make_simple_decision(current_prices, prev_prices)

                # Execute trades
                trades, portfolio_value = self.execute_simulated_trades(
                    actions, current_prices, current_date
                )

                # Report
                pnl = portfolio_value - self.initial_capital
                pnl_pct = (pnl / self.initial_capital) * 100

                print(f"\n{'='*60}")
                print(f"📅 {current_date.strftime('%Y-%m-%d')} | Iteration {iteration}")
                print(f"{'='*60}")

                if trades:
                    print("📊 Trades executed:")
                    for trade in trades[:5]:  # Show first 5
                        print(f"  {trade}")
                    if len(trades) > 5:
                        print(f"  ... and {len(trades)-5} more")
                else:
                    print("📊 No trades (holding positions)")

                print(f"\n💰 Portfolio Status:")
                print(f"  Value: ${portfolio_value:,.2f}")
                print(f"  Cash: ${self.cash:,.2f}")
                print(f"  P&L: ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
                print(f"  Positions: {self.portfolio_history[-1]['num_positions']}")

                # Simulate time delay
                time.sleep(interval_minutes * 60 / speed_multiplier / 1000)  # Scale down for demo

        except KeyboardInterrupt:
            print("\n\n🛑 Simulation stopped by user")

        self.print_final_report()

    def print_final_report(self):
        """Print final performance summary"""
        if not self.portfolio_history:
            print("No trading history to report")
            return

        final = self.portfolio_history[-1]
        final_value = final['portfolio_value']
        total_return = (final_value / self.initial_capital - 1) * 100

        print(f"\n{'='*60}")
        print("📊 FINAL TRADING REPORT")
        print(f"{'='*60}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Value: ${final_value:,.2f}")
        print(f"Total P&L: ${final_value - self.initial_capital:+,.2f}")
        print(f"Total Return: {total_return:+.2f}%")
        print(f"\nTrading Days: {len(self.portfolio_history)}")
        print(f"Final Positions: {final['num_positions']}")

        # Calculate best/worst days
        if len(self.portfolio_history) > 1:
            values = [h['portfolio_value'] for h in self.portfolio_history]
            returns = np.diff(values) / values[:-1] * 100
            print(f"\nBest Day: +{np.max(returns):.2f}%")
            print(f"Worst Day: {np.min(returns):.2f}%")

        print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Use smaller set of stocks for faster demo
    DEMO_TICKERS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
        'META', 'TSLA', 'JPM', 'V', 'WMT'
    ]

    print("="*60)
    print("  📈 SIMULATED TRADING BOT DEMO")
    print("="*60)
    print("\nThis is a pure simulation using historical data")
    print("No real money, no Alpaca, no market hours restrictions")
    print("Just a demo of how the bot would trade!\n")

    # Create bot
    bot = SimulatedTradingBot(
        tickers=DEMO_TICKERS,
        initial_capital=100000
    )

    # Download last 3 months of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    data = bot.download_historical_data(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )

    # Run simulation at 100x speed (so it finishes quickly)
    print("🎬 Running simulation at 100x speed...")
    print("   (3 months of trading compressed into ~2 minutes)\n")

    bot.run_simulation(
        data=data,
        interval_minutes=5,
        speed_multiplier=100  # Run fast for demo
    )
