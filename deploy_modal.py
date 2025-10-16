#!/usr/bin/env python3
"""
Deploy and run FinRL training on Modal with A100 GPU
"""

import subprocess
import sys

def check_modal_setup():
    """Check if Modal is set up correctly"""
    try:
        result = subprocess.run(['modal', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Modal CLI installed: {result.stdout.strip()}")
            return True
        else:
            print("✗ Modal CLI not found")
            return False
    except FileNotFoundError:
        print("✗ Modal CLI not found")
        return False

def setup_modal():
    """Set up Modal if not already done"""
    print("Setting up Modal...")
    print("1. Install Modal CLI:")
    print("   pip install modal")
    print("\n2. Authenticate with Modal:")
    print("   modal token new")
    print("\n3. Run the training:")
    print("   modal run modal_finrl_training.py::train_large_portfolio")

def deploy_finrl():
    """Deploy FinRL training to Modal"""
    if not check_modal_setup():
        setup_modal()
        return
    
    print("\n🚀 Deploying FinRL to Modal...")
    print("This will use your A100 GPU credits for serious RL training!")
    print("\nWhat you're about to run:")
    print("- 50 stocks from S&P 500")
    print("- 4+ years of market data")
    print("- 500,000 training timesteps")
    print("- Advanced technical indicators")
    print("- GPU-accelerated training")
    print("- Professional risk management")
    
    confirm = input("\nProceed with A100 training? (y/N): ")
    
    if confirm.lower() == 'y':
        print("\n🔥 Starting Modal deployment...")
        try:
            # Run the Modal function
            result = subprocess.run([
                'modal', 'run', 
                'modal_finrl_training.py::train_large_portfolio'
            ], check=True)
            
            print("\n🎉 Training completed successfully!")
            print("Check the output above for results.")
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Training failed: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure Modal token is set up: modal token new")
            print("2. Check Modal credits: modal stats")
            print("3. Verify A100 GPU availability")
            
        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")
            
    else:
        print("Deployment cancelled.")
        print("\nFor a smaller test run, you can:")
        print("1. Modify the ticker list in modal_finrl_training.py")
        print("2. Reduce train_timesteps to 50,000")
        print("3. Use smaller GPU: change gpu='A100' to gpu='T4'")

def quick_local_test():
    """Run a quick local test before Modal deployment"""
    print("🧪 Running quick local test...")
    
    try:
        result = subprocess.run([
            sys.executable, 'finrl_quick_test.py'
        ], check=True, capture_output=True, text=True)
        
        print("✓ Local test passed!")
        print("Sample output:")
        print(result.stdout.split('\n')[-10:])  # Last 10 lines
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Local test failed: {e}")
        print("Fix local issues before deploying to Modal")
        return False

def main():
    print("FinRL Modal Deployment Tool")
    print("="*40)
    
    print("\nOptions:")
    print("1. Quick local test")
    print("2. Deploy to Modal A100")
    print("3. Setup Modal CLI")
    print("4. Exit")
    
    choice = input("\nChoose option (1-4): ")
    
    if choice == '1':
        quick_local_test()
    elif choice == '2':
        deploy_finrl()
    elif choice == '3':
        setup_modal()
    elif choice == '4':
        print("Goodbye!")
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()