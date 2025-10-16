#!/usr/bin/env python3
"""
Test the massive datasets we found locally before Modal deployment
"""

from datasets import load_dataset
import pandas as pd
import numpy as np

def test_financial_datasets():
    """Test our discovered financial datasets"""
    
    print("🧪 Testing Massive Financial Datasets")
    print("="*50)
    
    # Test the top datasets we found - start with smaller ones
    datasets_to_test = [
        "zeroshot/twitter-financial-news-sentiment",  # Smaller dataset first
        "pauri32/fiqa-2018",
    ]
    
    loaded_data = {}
    
    for dataset_name in datasets_to_test:
        print(f"\n📊 Testing: {dataset_name}")
        
        try:
            # Use streaming mode or limit size to avoid timeout
            dataset = load_dataset(dataset_name, split='train', streaming=False)
            # Take only first 1000 samples for testing
            if len(dataset) > 1000:
                dataset = dataset.select(range(1000))
            
            print(f"  ✅ Success! {len(dataset):,} rows")
            print(f"  Columns: {list(dataset.features.keys())}")
            
            # Show sample
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"  Sample keys: {list(sample.keys())}")
                
                # Show first few values
                for key, value in sample.items():
                    if isinstance(value, str):
                        print(f"    {key}: {str(value)[:100]}...")
                    else:
                        print(f"    {key}: {value}")
                    if len(str(key)) > 20:  # Limit output
                        break
            
            loaded_data[dataset_name] = dataset
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    print(f"\n🎯 Successfully loaded {len(loaded_data)} datasets")
    
    # Analyze the data for FinRL usage
    analyze_for_finrl(loaded_data)
    
    return loaded_data

def analyze_for_finrl(datasets):
    """Analyze datasets for FinRL training potential"""
    
    print(f"\n🔍 FinRL Analysis")
    print("="*30)
    
    total_samples = 0
    
    for name, dataset in datasets.items():
        print(f"\n📈 {name}:")
        
        sample_count = len(dataset)
        total_samples += sample_count
        
        if 'yahoo-finance' in name:
            print(f"  • {sample_count:,} financial records")
            print(f"  • Type: OHLCV price data")
            print(f"  • Use: Primary training data for RL environment")
            
        elif 'sentiment' in name:
            print(f"  • {sample_count:,} sentiment records") 
            print(f"  • Type: Twitter financial sentiment")
            print(f"  • Use: Alternative data feature for RL")
            
        elif 'fiqa' in name:
            print(f"  • {sample_count:,} Q&A records")
            print(f"  • Type: Financial question-answering")
            print(f"  • Use: Market context understanding")
            
        elif 'wealth' in name:
            print(f"  • {sample_count:,} instruction records")
            print(f"  • Type: Financial instruction following")
            print(f"  • Use: Portfolio strategy reasoning")
    
    print(f"\n🚀 FINRL TRAINING POTENTIAL:")
    print(f"  Total samples: {total_samples:,}")
    print(f"  Estimated training data: {total_samples * 0.8:,.0f} samples")
    print(f"  Multi-modal: Price + Sentiment + Context")
    print(f"  Ready for: A100 GPU training with Modal")

def create_sample_combined_dataset(datasets):
    """Create a sample combined dataset for RL training"""
    
    print(f"\n🏗️  Creating Combined Sample Dataset")
    print("="*40)
    
    combined_samples = []
    
    # Sample from each dataset
    for name, dataset in datasets.items():
        sample_size = min(100, len(dataset))  # 100 samples each
        samples = dataset.select(range(sample_size))
        
        for sample in samples:
            combined_sample = {
                'source': name,
                'data': sample
            }
            combined_samples.append(combined_sample)
    
    print(f"✓ Combined {len(combined_samples)} samples from {len(datasets)} sources")
    
    # Show structure
    print(f"\nSample combined record:")
    if combined_samples:
        sample = combined_samples[0]
        print(f"  Source: {sample['source']}")
        print(f"  Data keys: {list(sample['data'].keys())}")
    
    return combined_samples

def estimate_modal_training():
    """Estimate Modal training requirements"""
    
    print(f"\n💰 Modal Training Estimation")
    print("="*35)
    
    estimates = {
        'datasets': 4,
        'total_samples': 60000,  # Conservative estimate
        'training_samples': 48000,  # 80% train split
        'features_per_sample': 50,  # Including technical indicators
        'gpu_type': 'A100',
        'estimated_training_time': '2-3 hours',
        'estimated_cost': '$25-50',
        'memory_required': '32-64GB',
        'algorithms': ['PPO', 'SAC', 'A2C']
    }
    
    for key, value in estimates.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n🎯 Expected Outcomes:")
    print(f"  • Multi-asset portfolio optimization")
    print(f"  • Sentiment-enhanced trading decisions") 
    print(f"  • Risk-adjusted performance metrics")
    print(f"  • Production-ready trading models")
    print(f"  • Benchmark comparison vs buy-and-hold")

if __name__ == "__main__":
    # Test the datasets
    datasets = test_financial_datasets()
    
    if datasets:
        # Create combined sample
        combined = create_sample_combined_dataset(datasets)
        
        # Estimate Modal requirements
        estimate_modal_training()
        
        print(f"\n🚀 READY FOR MODAL A100 TRAINING!")
        print(f"Run: modal run massive_finrl_hf.py::build_massive_dataset")
        print(f"Then: modal run massive_finrl_hf.py::train_massive_portfolio")
    else:
        print(f"\n❌ No datasets loaded - check HuggingFace access")