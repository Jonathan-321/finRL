"""
Quick local training for 5-minute intraday model
Train locally first to test, then deploy to Modal for production training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_intraday_system import train_advanced_intraday

if __name__ == "__main__":
    print("\n🚀 Starting local training of 5-minute intraday model...")
    print("   This will train a smaller version for quick testing")
    print("   For production: use modal_train_intraday_advanced.py")
    print()

    # Train the model
    model_path = train_advanced_intraday()

    print("\n✅ Local training complete!")
    print(f"   Model saved: {model_path}")
    print("\n📋 Next steps:")
    print("   1. Test paper trading: python3 paper_trading_intraday_5min.py")
    print("   2. For production training: modal run modal_train_intraday_advanced.py")
    print("   3. Live testing tomorrow at 9:30 AM ET")