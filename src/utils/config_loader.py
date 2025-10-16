"""
Configuration loader following clean code principles.
Single responsibility: Load and validate configuration settings.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv


@dataclass
class TradingConfig:
    """Trading configuration with type hints for better code clarity."""
    stocks: list
    max_position_size: float
    max_sector_exposure: float
    min_cash_reserve: float
    max_drawdown: float
    position_stop_loss: float
    daily_var_limit: float
    volatility_threshold: int


@dataclass
class ModelConfig:
    """Model configuration with clear parameter definitions."""
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    gamma: float
    total_timesteps: int


class ConfigLoader:
    """
    Centralized configuration management following DRY principle.
    Loads configuration from YAML and environment variables.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration loader with optional custom path."""
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        self._load_env_variables()

    @staticmethod
    def _get_default_config_path() -> Path:
        """Get default configuration file path."""
        root_dir = Path(__file__).parent.parent.parent
        return root_dir / "config" / "config.yaml"

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Config file not found at {self.config_path}, using defaults")
            return self._get_default_config()

    @staticmethod
    def _load_env_variables():
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Return default configuration if file not found."""
        return {
            'trading': {
                'position_limits': {
                    'max_position_size': 0.10,
                    'max_sector_exposure': 0.30,
                    'min_cash_reserve': 0.10
                },
                'risk_management': {
                    'max_drawdown': 0.15,
                    'position_stop_loss': 0.10,
                    'daily_var_limit': 0.03,
                    'volatility_threshold': 40
                }
            },
            'model': {
                'ppo': {
                    'learning_rate': 0.00025,
                    'n_steps': 2048,
                    'batch_size': 64,
                    'n_epochs': 10,
                    'gamma': 0.99
                },
                'training': {
                    'total_timesteps_local': 100000,
                    'total_timesteps_cloud': 500000
                }
            }
        }

    def get_trading_config(self, production: bool = False) -> TradingConfig:
        """
        Get trading configuration.

        Args:
            production: Whether to use production settings (99 stocks) or test (50 stocks)

        Returns:
            TradingConfig object with validated settings
        """
        stocks = self.config['trading']['stocks_99_extended'] if production else self.config['trading']['stocks_50']
        limits = self.config['trading']['position_limits']
        risk = self.config['trading']['risk_management']

        return TradingConfig(
            stocks=stocks[:99] if production else stocks,  # Ensure we don't exceed 99
            max_position_size=limits['max_position_size'],
            max_sector_exposure=limits['max_sector_exposure'],
            min_cash_reserve=limits['min_cash_reserve'],
            max_drawdown=risk['max_drawdown'],
            position_stop_loss=risk['position_stop_loss'],
            daily_var_limit=risk['daily_var_limit'],
            volatility_threshold=risk['volatility_threshold']
        )

    def get_model_config(self, cloud: bool = False) -> ModelConfig:
        """
        Get model configuration.

        Args:
            cloud: Whether to use cloud training settings

        Returns:
            ModelConfig object with training parameters
        """
        ppo = self.config['model']['ppo']
        training = self.config['model']['training']

        return ModelConfig(
            learning_rate=ppo['learning_rate'],
            n_steps=ppo['n_steps'],
            batch_size=ppo['batch_size'],
            n_epochs=ppo['n_epochs'],
            gamma=ppo['gamma'],
            total_timesteps=training['total_timesteps_cloud'] if cloud else training['total_timesteps_local']
        )

    def get_alpaca_credentials(self) -> tuple:
        """
        Get Alpaca API credentials from environment variables.

        Returns:
            Tuple of (api_key, secret_key, base_url)
        """
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        base_url = self.config.get('alpaca', {}).get('paper_trading_url', 'https://paper-api.alpaca.markets')

        if not api_key or not secret_key:
            raise ValueError("Alpaca credentials not found in environment variables")

        return api_key, secret_key, base_url

    def get_indicators(self) -> list:
        """Get list of enabled technical indicators."""
        return self.config.get('indicators', {}).get('enabled', [])

    def get_date_range(self, dataset: str = 'train') -> tuple:
        """
        Get date range for training or testing.

        Args:
            dataset: 'train' or 'test'

        Returns:
            Tuple of (start_date, end_date)
        """
        data_config = self.config.get('data', {})

        if dataset == 'train':
            return data_config.get('train_start_date'), data_config.get('train_end_date')
        else:
            return data_config.get('test_start_date'), data_config.get('test_end_date')


# Singleton pattern for configuration
_config_instance = None


def get_config() -> ConfigLoader:
    """
    Get singleton configuration instance.

    Returns:
        ConfigLoader instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader()
    return _config_instance