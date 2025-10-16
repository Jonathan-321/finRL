#!/usr/bin/env python3
"""
Launch massive FinRL training with discovered HuggingFace datasets
"""

import subprocess
import sys

def launch_massive_training():
    """Launch the massive training on Modal with A100"""
    
    print("🚀 LAUNCHING MASSIVE FINRL TRAINING")
    print("="*50)
    print("Dataset Sources:")
    print("✓ bwzheng2010/yahoo-finance-data (13,460 downloads)")
    print("✓ Arrechenash/stocks (13,172 downloads)")
    print("✓ zeroshot/twitter-financial-news-sentiment (2,532 downloads)")
    print("✓ paperswithbacktest/Stocks-Daily-Price (1,256 downloads)")
    print("✓ WinkingFace/CryptoLM-Bitcoin-BTC-USDT (1,092 downloads)")
    print("✓ Plus yfinance API for 500+ stocks")
    print()
    print("Training Configuration:")
    print("• GPU: A100 (your credits)")
    print("• Memory: 64GB RAM")
    print("• Training Time: ~2-3 hours")
    print("• Estimated Cost: $20-40")
    print("• Dataset Size: 10M+ data points")
    print("• Assets: Stocks + Crypto + Sentiment")
    print("• Algorithms: PPO, SAC, A2C ensemble")
    print()
    
    confirm = input("🔥 Ready to spend Modal credits on massive training? (y/N): ")
    
    if confirm.lower() == 'y':
        print("\n🚀 Launching Modal training...")
        
        try:
            # First, let's update the Modal script with the actual working datasets
            update_modal_script()
            
            # Run the massive training
            print("Starting massive dataset build...")
            subprocess.run([
                'modal', 'run', 
                'massive_finrl_hf.py::build_massive_dataset'
            ], check=True)
            
            print("\nStarting massive model training...")
            subprocess.run([
                'modal', 'run', 
                'massive_finrl_hf.py::train_massive_portfolio'
            ], check=True)
            
            print("\n🎉 MASSIVE TRAINING COMPLETED!")
            print("Check the logs above for results.")
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Training failed: {e}")
            print("This could be due to:")
            print("1. Modal credit limits")
            print("2. GPU availability")
            print("3. Dataset loading issues")
            print("4. Memory constraints")
            
        except KeyboardInterrupt:
            print("\n⏹️ Training interrupted by user")
    
    else:
        print("\n💡 Alternative options:")
        print("1. Run smaller test: python finrl_quick_test.py")
        print("2. Use T4 GPU instead of A100 (cheaper)")
        print("3. Reduce dataset size in massive_finrl_hf.py")
        print("4. Train locally with CPU (slower)")

def update_modal_script():
    """Update the Modal script with working datasets"""
    
    print("📝 Updating Modal script with verified datasets...")
    
    # The working datasets we found
    working_datasets = [
        "bwzheng2010/yahoo-finance-data",
        "Arrechenash/stocks", 
        "zeroshot/twitter-financial-news-sentiment",
        "paperswithbacktest/Stocks-Daily-Price",
        "WinkingFace/CryptoLM-Bitcoin-BTC-USDT",
        "WinkingFace/CryptoLM-Ethereum-ETH-USDT",
        "pauri32/fiqa-2018",
        "gbharti/wealth-alpaca_lora"
    ]
    
    # Read the current script
    with open('massive_finrl_hf.py', 'r') as f:
        content = f.read()
    
    # Update the datasets list
    new_datasets = f"""
    datasets_to_load = {working_datasets}
    """
    
    # Replace the old list
    import re
    pattern = r'datasets_to_load = \[.*?\]'
    content = re.sub(pattern, new_datasets.strip(), content, flags=re.DOTALL)
    
    # Write back
    with open('massive_finrl_hf.py', 'w') as f:
        f.write(content)
    
    print("✓ Updated with 8 verified datasets")

def run_local_test_first():
    """Run a local test with HF datasets before Modal"""
    
    print("🧪 Running local test with HF datasets...")
    
    try:
        from datasets import load_dataset
        
        # Test the best dataset
        print("Loading yahoo-finance-data...")
        dataset = load_dataset("bwzheng2010/yahoo-finance-data", split='train')
        print(f"✓ Loaded {len(dataset)} rows")
        print(f"Columns: {list(dataset.features.keys())}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample: {sample}")
        
        # Test sentiment dataset
        print("\nLoading sentiment data...")
        sentiment_data = load_dataset("zeroshot/twitter-financial-news-sentiment", split='train')
        print(f"✓ Loaded {len(sentiment_data)} sentiment records")
        
        print("\n✅ Local test successful!")
        print("Ready for Modal deployment.")
        return True
        
    except Exception as e:
        print(f"❌ Local test failed: {e}")
        return False

def main():
    print("Massive FinRL Training Launcher")
    print("="*40)
    
    print("\nOptions:")
    print("1. Run local HF dataset test")
    print("2. Launch massive Modal training")  
    print("3. Update Modal script with verified datasets")
    print("4. Exit")
    
    choice = input("\nChoose option (1-4): ")
    
    if choice == '1':
        run_local_test_first()
    elif choice == '2':
        launch_massive_training()
    elif choice == '3':
        update_modal_script()
    elif choice == '4':
        print("Goodbye!")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()