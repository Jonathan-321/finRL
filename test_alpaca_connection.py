"""
Test Alpaca API connection
Verifies credentials and displays account info
"""

import os
from alpaca_trade_api import REST
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_connection():
    """Test Alpaca paper trading connection"""

    print("🔐 Testing Alpaca Paper Trading Connection...")
    print("="*60)

    try:
        # Initialize API
        api = REST(
            key_id=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        )

        # Get account info
        account = api.get_account()

        print("✅ CONNECTION SUCCESSFUL!\n")
        print(f"Account Number: {account.account_number}")
        print(f"Status: {account.status}")
        print(f"Currency: {account.currency}")
        print(f"\n💰 ACCOUNT BALANCES:")
        print(f"  Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"  Cash: ${float(account.cash):,.2f}")
        print(f"  Buying Power: ${float(account.buying_power):,.2f}")

        # Get market clock
        clock = api.get_clock()
        print(f"\n🕐 MARKET STATUS:")
        print(f"  Open: {'Yes' if clock.is_open else 'No'}")
        print(f"  Current Time: {clock.timestamp}")
        if not clock.is_open:
            print(f"  Next Open: {clock.next_open}")
            print(f"  Next Close: {clock.next_close}")

        # Get positions
        positions = api.list_positions()
        print(f"\n📊 CURRENT POSITIONS: {len(positions)}")
        if positions:
            for pos in positions[:10]:  # Show first 10
                pnl_pct = float(pos.unrealized_plpc) * 100
                print(f"  {pos.symbol}: {pos.qty} shares @ ${float(pos.current_price):.2f} (P&L: {pnl_pct:+.2f}%)")
        else:
            print("  (No open positions)")

        print("\n" + "="*60)
        print("✅ All systems ready for paper trading!")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n❌ CONNECTION FAILED!")
        print(f"Error: {e}")
        print("\nPlease check:")
        print("  1. .env file exists with correct credentials")
        print("  2. ALPACA_API_KEY and ALPACA_SECRET_KEY are set")
        print("  3. You're using paper trading credentials")
        return False


if __name__ == "__main__":
    test_connection()
