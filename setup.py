"""
FinRL Trading System Setup
Following Python packaging best practices
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="finrl-trading-system",
    version="1.0.0",
    author="Jonathan Muhire",
    description="Reinforcement Learning-based Algorithmic Trading System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jonathan-321/finRL",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "stable-baselines3>=2.0.0",
        "gymnasium>=0.29.1",
        "yfinance>=0.2.28",
        "pandas>=2.0.3",
        "numpy>=1.24.0,<2.0.0",
        "stockstats>=0.6.2",
        "alpaca-py>=0.13.3",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.1",
        ],
        "cloud": [
            "modal>=0.56.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "finrl-train=training.train_with_tech_indicators:main",
            "finrl-trade=trading.alpaca_paper_trading_production:main",
            "finrl-backtest=analysis.backtest_strategy:main",
        ],
    },
)