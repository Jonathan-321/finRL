#!/usr/bin/env python3
"""
Search and catalog available financial datasets on HuggingFace Hub
"""

import requests
import json
from datasets import load_dataset

def search_financial_datasets():
    """Search HuggingFace Hub for financial datasets"""
    
    print("🔍 Searching HuggingFace Hub for Financial Datasets...")
    print("="*60)
    
    # Search queries for financial data
    search_terms = [
        "stock market",
        "financial data", 
        "crypto currency",
        "forex trading",
        "stock prices",
        "financial news",
        "earnings data",
        "economic indicators",
        "trading data",
        "market data",
        "finance",
        "stocks",
        "cryptocurrency",
        "bitcoin",
        "ethereum"
    ]
    
    found_datasets = []
    
    for term in search_terms:
        print(f"\n📊 Searching for: '{term}'")
        
        try:
            # Use HF Hub API to search
            url = f"https://huggingface.co/api/datasets"
            params = {
                'search': term,
                'limit': 20,
                'sort': 'downloads',
                'direction': -1
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                datasets = response.json()
                
                for dataset in datasets:
                    dataset_info = {
                        'id': dataset['id'],
                        'downloads': dataset.get('downloads', 0),
                        'description': dataset.get('description', '')[:200] + '...',
                        'tags': dataset.get('tags', []),
                        'size': dataset.get('size', 'Unknown')
                    }
                    
                    # Filter for relevant financial datasets
                    if any(keyword in dataset_info['id'].lower() or 
                          keyword in dataset_info['description'].lower() 
                          for keyword in ['stock', 'financial', 'crypto', 'trading', 'market', 'price', 'economic']):
                        
                        found_datasets.append(dataset_info)
                        print(f"  ✓ {dataset_info['id']} ({dataset_info['downloads']} downloads)")
            
        except Exception as e:
            print(f"  ❌ Error searching '{term}': {e}")
    
    # Remove duplicates and sort by downloads
    unique_datasets = {}
    for dataset in found_datasets:
        if dataset['id'] not in unique_datasets:
            unique_datasets[dataset['id']] = dataset
    
    sorted_datasets = sorted(unique_datasets.values(), 
                           key=lambda x: x['downloads'], reverse=True)
    
    print(f"\n🎯 Found {len(sorted_datasets)} Unique Financial Datasets")
    print("="*60)
    
    # Top 20 most downloaded
    for i, dataset in enumerate(sorted_datasets[:20]):
        print(f"{i+1:2d}. {dataset['id']}")
        print(f"    Downloads: {dataset['downloads']:,}")
        print(f"    Size: {dataset['size']}")
        print(f"    Description: {dataset['description']}")
        print()
    
    return sorted_datasets

def test_promising_datasets():
    """Test loading some promising financial datasets"""
    
    print("🧪 Testing Promising Financial Datasets...")
    print("="*50)
    
    # Known financial datasets to test
    promising_datasets = [
        "zeroshot/twitter-financial-news-sentiment",
        "financial_phrasebank",
        "rahul2003/financial_news_dataset", 
        "pauri32/fiqa-2018",
        "AdiOO7/stock-market-tweets",
        "rajistics/financialnews",
        "sohom/fin-fact",
        "siebert/sentiment-roberta-large-english",
        "gbharti/wealth-alpaca_lora",
        "microsoft/DialoGPT-medium",
        "nvidia/megatron-bert-uncased-345m"
    ]
    
    working_datasets = []
    
    for dataset_name in promising_datasets:
        print(f"\n📊 Testing: {dataset_name}")
        
        try:
            # Try to load the dataset
            dataset = load_dataset(dataset_name, split='train')
            
            print(f"  ✅ Success! {len(dataset)} rows")
            print(f"  Columns: {list(dataset.features.keys())}")
            
            # Show sample
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"  Sample: {str(sample)[:200]}...")
            
            working_datasets.append({
                'name': dataset_name,
                'rows': len(dataset),
                'columns': list(dataset.features.keys()),
                'sample': dataset[0] if len(dataset) > 0 else None
            })
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    print(f"\n🎉 Working Datasets: {len(working_datasets)}")
    return working_datasets

def create_financial_data_plan():
    """Create a plan for building massive financial dataset"""
    
    print("\n🚀 MASSIVE FINANCIAL DATASET PLAN")
    print("="*50)
    
    plan = {
        "data_sources": {
            "huggingface_datasets": [
                "zeroshot/twitter-financial-news-sentiment",
                "financial_phrasebank", 
                "rahul2003/financial_news_dataset"
            ],
            "api_sources": [
                "yfinance (Yahoo Finance)",
                "alpha_vantage (Premium API)",
                "polygon.io (High-frequency data)",
                "coinbase_pro (Crypto)",
                "binance (Crypto spot/futures)"
            ],
            "alternative_data": [
                "Reddit sentiment (r/wallstreetbets)",
                "Twitter financial mentions",
                "Google Trends for tickers",
                "Economic calendar (FRED)",
                "Insider trading filings"
            ]
        },
        "target_universe": {
            "stocks": "S&P 500 + Russell 2000 + International (2500+ stocks)",
            "crypto": "Top 200 cryptocurrencies", 
            "forex": "50+ currency pairs",
            "commodities": "25+ futures contracts",
            "bonds": "Government and corporate bonds",
            "derivatives": "Options flow data"
        },
        "timeframe": "2000-2024 (24+ years)",
        "frequency": "Multiple (1min, 5min, 1hour, 1day)",
        "features": [
            "OHLCV price data",
            "50+ technical indicators", 
            "Sentiment scores",
            "Economic indicators",
            "Options flow",
            "Insider trading",
            "Earnings announcements",
            "News events",
            "Social media mentions"
        ],
        "estimated_size": "10+ TB of data",
        "training_samples": "100+ million data points"
    }
    
    for category, items in plan.items():
        print(f"\n{category.upper()}:")
        if isinstance(items, dict):
            for key, value in items.items():
                print(f"  {key}: {value}")
        elif isinstance(items, list):
            for item in items:
                print(f"  • {item}")
        else:
            print(f"  {items}")
    
    print(f"\n💡 RECOMMENDED APPROACH:")
    print(f"1. Start with HF sentiment datasets + yfinance for 500 stocks")
    print(f"2. Add crypto data from Binance/Coinbase APIs")
    print(f"3. Enhance with alternative data (Reddit, Twitter)")
    print(f"4. Scale to multi-asset universe on A100 GPU")
    print(f"5. Target 10M+ training samples for serious RL")
    
    return plan

if __name__ == "__main__":
    # Search for datasets
    datasets = search_financial_datasets()
    
    # Test promising ones
    working = test_promising_datasets()
    
    # Create comprehensive plan
    plan = create_financial_data_plan()
    
    print(f"\n🎯 READY FOR MASSIVE FINRL!")
    print(f"Found {len(working)} working datasets")
    print(f"Plan: {plan['estimated_size']} dataset with {plan['training_samples']}")