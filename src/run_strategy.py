"""
Main script to run the pairs trading strategy with time simulation.
"""

import os
import sys
import yaml
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Local imports
from src.data.data_collector import DataCollector
from src.strategy.pairs_selection import PairsSelector
from src.models.ml_model import MLModel
from src.strategy.trading_strategy import TradingStrategy

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_dir = os.path.join(project_root, 'config')
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
        
    config_file = os.path.join(config_dir, config_path)
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
            
    # Create default config if it doesn't exist
    default_config = {
        'data': {
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'QCOM'],
            'start_date': '2022-01-01',
            'end_date': '2024-02-29'
        },
        'pairs_selection': {
            'formation_period': 126,
            'p_value_threshold': 0.05,
            'correlation_threshold': 0.7,
            'min_half_life': 5.0,
            'max_half_life': 20.0
        },
        'ml_model': {
            'model_type': 'ensemble',
            'lookback_period': 20,
            'test_size': 0.2,
            'random_state': 42,
            'entry_threshold': 2.0
        },
        'strategy': {
            'entry_threshold': 2.0,
            'exit_threshold': 0.5,
            'stop_loss': 3.0,
            'position_size': 1000.0,
            'max_positions': 5,
            'initial_capital': 1000000.0,
            'min_holding_period': 3,
            'max_holding_period': 20,
            'commission': 0.001,
            'slippage': 0.0005,
            'profit_target': 2.0,
            'risk_per_trade': 0.02,
            'atr_periods': 14,
            'position_size_atr_multiplier': 1.0,
            'max_position_size_pct': 0.1,
            'min_liquidity_ratio': 0.1
        },
        'logging': {
            'log_dir': 'logs',
            'log_level': 'INFO'
        }
    }
    
    # Save default config
    with open(config_file, 'w') as f:
        yaml.safe_dump(default_config, f)
    
    return default_config

def setup_logging(log_dir: str):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(
        log_dir, 
        f'strategy_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main():
    """Main function to run the pairs trading strategy."""
    try:
        # Load configuration
        config = load_config('strategy_config.yaml')
        
        # Setup logging
        log_dir = os.path.join(project_root, config.get('logging', {}).get('log_dir', 'logs'))
        setup_logging(log_dir)
        logger = logging.getLogger(__name__)
        
        logger.info("Starting pairs trading strategy...")
        
        # Initialize data collector with configuration parameters
        data_collector = DataCollector(
            symbols=config['data']['symbols'],
            start_date=config['data']['start_date'],
            end_date=config['data']['end_date']
        )
        
        # Download and preprocess data
        logger.info("Downloading historical data...")
        data = data_collector.download_data()
        
        # Initialize pairs selector
        logger.info("Selecting pairs...")
        pairs_selector = PairsSelector(
            price_data=data,
            formation_period=config['pairs_selection']['formation_period'],
            p_value_threshold=config['pairs_selection']['p_value_threshold'],
            correlation_threshold=config['pairs_selection']['correlation_threshold'],
            min_half_life=config['pairs_selection']['min_half_life'],
            max_half_life=config['pairs_selection']['max_half_life']
        )
        
        pairs = pairs_selector.find_cointegrated_pairs()
        
        if not pairs:
            logger.warning("No cointegrated pairs found.")
            return
            
        logger.info(f"Found {len(pairs)} cointegrated pairs")
        logger.info(f"Pairs found: {pairs}")
        
        # Convert pairs list to dictionary format for ML model
        pairs_dict = {}
        hedge_ratios = {}
        for pair in pairs:
            symbol1, symbol2, correlation, p_value = pair
            # Calculate spread using price data
            prices1 = data[symbol1]['Close']
            prices2 = data[symbol2]['Close']
            hedge_ratio = np.polyfit(prices2, prices1, 1)[0]
            spread = prices1 - hedge_ratio * prices2
            
            pair_name = f"{symbol1}-{symbol2}"
            pairs_dict[pair_name] = {
                'spread': spread,
                'stock1': data[symbol1],
                'stock2': data[symbol2],
                'hedge_ratio': hedge_ratio,
                'correlation': correlation,
                'p_value': p_value
            }
            hedge_ratios[pair_name] = hedge_ratio
        
        # Initialize ML model
        logger.info("Initializing ML model...")
        ml_model = MLModel(
            model_type=config['ml_model']['model_type'],
            lookback_period=config['ml_model']['lookback_period'],
            test_size=config['ml_model']['test_size'],
            random_state=config['ml_model']['random_state'],
            entry_threshold=config['ml_model']['entry_threshold']
        )
        
        # Train model on historical data
        logger.info("Training ML model...")
        ml_model.train(pairs_dict)
        
        # Initialize trading strategy
        logger.info("Initializing trading strategy...")
        strategy = TradingStrategy(**config['strategy'])
        
        # Run backtest
        logger.info("Running backtest...")
        
        # Generate trading signals
        signals = ml_model.generate_signals(pairs_dict)
        logger.info(f"Generated signals: {signals}")
        
        # Execute trades based on signals
        executed_trades = strategy.execute_trades(signals, pairs_dict, hedge_ratios)
        
        # Update positions and get performance metrics
        strategy.update_positions(pairs_dict)
        performance = strategy.get_performance_metrics()
        
        # Log results
        logger.info("Backtest completed.")
        logger.info(f"Number of trades executed: {len(executed_trades)}")
        logger.info("Performance metrics:")
        for metric, value in performance.items():
            logger.info(f"{metric}: {value}")
        
        return performance
        
    except Exception as e:
        logger.error(f"Error running strategy: {str(e)}")
        raise

if __name__ == '__main__':
    main()
