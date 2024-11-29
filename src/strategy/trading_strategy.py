"""
Implementation of the pairs trading strategy with ML-enhanced signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from src.models.ml_model import MLModel
import logging
from src.utils.rate_limiter import rate_limited

class TradingStrategy:
    """Implements the pairs trading strategy."""
    
    def __init__(self,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 stop_loss: float = 3.0,
                 position_size: float = 1000.0,
                 max_positions: int = 5,
                 initial_capital: float = 1000000.0,
                 min_holding_period: int = 3,
                 max_holding_period: int = 20,
                 commission: float = 0.001,
                 slippage: float = 0.0005,
                 profit_target: float = 2.0,
                 risk_per_trade: float = 0.02,
                 atr_periods: int = 14,
                 position_size_atr_multiplier: float = 1.0,
                 max_position_size_pct: float = 0.1,
                 min_liquidity_ratio: float = 0.1):
        """
        Initialize the TradingStrategy with enhanced parameters.
        
        Args:
            entry_threshold: Z-score threshold for trade entry
            exit_threshold: Z-score threshold for trade exit
            stop_loss: Maximum loss allowed per trade
            position_size: Base position size in base currency
            max_positions: Maximum number of concurrent positions
            initial_capital: Starting capital
            min_holding_period: Minimum holding period in days
            max_holding_period: Maximum holding period in days
            commission: Commission rate per trade
            slippage: Slippage rate per trade
            profit_target: Take profit at this z-score
            risk_per_trade: Maximum risk per trade as fraction of capital
            atr_periods: Periods for ATR calculation
            position_size_atr_multiplier: Multiplier for ATR-based position sizing
            max_position_size_pct: Maximum position size as fraction of capital
            min_liquidity_ratio: Minimum liquidity ratio for trade entry
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.base_position_size = position_size
        self.max_positions = max_positions
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.min_holding_period = min_holding_period
        self.max_holding_period = max_holding_period
        self.commission = commission
        self.slippage = slippage
        self.profit_target = profit_target
        self.risk_per_trade = risk_per_trade
        self.atr_periods = atr_periods
        self.position_size_atr_multiplier = position_size_atr_multiplier
        self.max_position_size_pct = max_position_size_pct
        self.min_liquidity_ratio = min_liquidity_ratio
        self.positions = {}
        self.trades = []
        self.logger = logging.getLogger(__name__)
    
    def calculate_zscore(self, spread: pd.Series, window: int = 20) -> float:
        """
        Calculate the z-score of the current spread.
        
        Args:
            spread: Time series of price spreads
            window: Rolling window size for z-score calculation
            
        Returns:
            Current z-score
        """
        if len(spread) < window:
            return 0.0
        
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        
        current_zscore = (spread.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
        return current_zscore
    
    @rate_limited
    def generate_signals(self, pair_data: Dict[str, pd.DataFrame], ml_model: MLModel) -> Dict[str, dict]:
        """
        Generate trading signals with enhanced criteria.
        
        Args:
            pair_data: Dictionary of price data for each pair
            ml_model: Trained ML model for spread prediction
            
        Returns:
            Dictionary of trading signals
        """
        signals = {}
        
        for pair_name, data in pair_data.items():
            try:
                # Get ML model signals
                ml_signals = ml_model.generate_signals({pair_name: data})
                if not ml_signals or pair_name not in ml_signals:
                    continue
                
                signal_data = ml_signals[pair_name]
                signal = signal_data['signal']
                
                # Only generate signals for strong enough predictions
                if signal != 0:
                    signals[pair_name] = signal_data
            
            except Exception as e:
                self.logger.warning(f"Error generating signal for {pair_name}: {str(e)}")
                continue
        
        return signals
    
    def _calculate_atr(self, price_data: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range."""
        try:
            high = price_data.rolling(2).max()
            low = price_data.rolling(2).min()
            tr = high - low
            atr = tr.rolling(window=period).mean()
            return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else price_data.std()
        except Exception as e:
            self.logger.warning(f"Error calculating ATR: {str(e)}")
            return price_data.std()
    
    def execute_trades(self, signals: Dict[str, dict], pair_data: Dict[str, dict], hedge_ratios: Dict[str, float]) -> List[dict]:
        """
        Execute trades based on generated signals.
        
        Args:
            signals: Dictionary of trading signals for each pair
            pair_data: Dictionary of price data for each pair
            hedge_ratios: Dictionary of hedge ratios for each pair
            
        Returns:
            List of executed trades
        """
        executed_trades = []
        
        try:
            self.logger.info(f"Processing signals for {len(signals)} pairs")
            for pair_name, signal_data in signals.items():
                if not signal_data or 'signal' not in signal_data:
                    self.logger.debug(f"No valid signal data for pair {pair_name}. Skipping trade execution.")
                    continue
                
                signal = signal_data['signal']
                confidence = signal_data.get('confidence', 0.5)
                zscore = signal_data.get('zscore', 0)
                
                # Skip if no clear signal - using a very low threshold
                if abs(signal) < 0.01:  # Reduced from 0.1
                    self.logger.debug(f"Signal for pair {pair_name} below minimum threshold. Skipping.")
                    continue
                
                self.logger.info(f"Processing trade for {pair_name} - Signal: {signal}, Confidence: {confidence:.2f}, Z-score: {zscore:.2f}")
                
                # Get current prices and calculate position sizes
                try:
                    stock1_data = pair_data[pair_name]['stock1']['Close']
                    stock2_data = pair_data[pair_name]['stock2']['Close']
                    price1 = stock1_data.iloc[-1]
                    price2 = stock2_data.iloc[-1]
                    
                    # Calculate position size based on confidence and risk
                    base_position = self.base_position_size * (1 + confidence)  # Increased position size with confidence
                    max_position = self.capital * self.max_position_size_pct
                    position_size = min(base_position, max_position)
                    
                    # Calculate ATR-based position sizing
                    atr1 = self._calculate_atr(stock1_data, self.atr_periods)
                    atr2 = self._calculate_atr(stock2_data, self.atr_periods)
                    
                    # Adjust position size based on volatility - more aggressive
                    vol_adjustment = 1.5 / (atr1 / price1 + atr2 / price2)  # Increased from 1.0
                    position_size *= min(1.5, vol_adjustment)  # Increased from 1.0
                    
                    # Calculate number of shares based on hedge ratio
                    hedge_ratio = hedge_ratios[pair_name]
                    shares1 = int(position_size / price1)
                    shares2 = int(shares1 * hedge_ratio)
                    
                    # Calculate transaction costs
                    transaction_cost1 = shares1 * price1 * (self.commission + self.slippage)
                    transaction_cost2 = shares2 * price2 * (self.commission + self.slippage)
                    total_cost = transaction_cost1 + transaction_cost2
                    
                    # Check if we have enough capital - more lenient check
                    total_position_value = shares1 * price1 + shares2 * price2
                    if total_position_value + total_cost > self.capital * (self.max_position_size_pct * 1.2):  # Increased limit
                        self.logger.warning(f"Position size exceeds maximum allowed for pair {pair_name}. Adjusting size.")
                        # Adjust position size instead of skipping
                        adjustment_factor = (self.capital * self.max_position_size_pct) / (total_position_value + total_cost)
                        shares1 = int(shares1 * adjustment_factor)
                        shares2 = int(shares2 * adjustment_factor)
                    
                    # Execute the trade
                    trade = {
                        'pair': pair_name,
                        'signal': signal,
                        'confidence': confidence,
                        'zscore': zscore,
                        'shares1': shares1,
                        'shares2': shares2,
                        'price1': price1,
                        'price2': price2,
                        'hedge_ratio': hedge_ratio,
                        'timestamp': pd.Timestamp.now(),
                        'total_cost': total_cost,
                        'position_value': total_position_value
                    }
                    
                    # Update positions and capital
                    self.positions[pair_name] = trade
                    self.capital -= (total_position_value + total_cost)
                    
                    executed_trades.append(trade)
                    self.logger.info(f"Executed trade for {pair_name}: {trade}")
                    
                except Exception as e:
                    self.logger.error(f"Error executing trade for {pair_name}: {str(e)}")
                    continue
            
            self.logger.info(f"Successfully executed {len(executed_trades)} trades")
            
        except Exception as e:
            self.logger.error(f"Error in execute_trades: {str(e)}")
        
        return executed_trades
    
    def update_positions(self, pair_data: Dict[str, pd.DataFrame]) -> None:
        """
        Update PnL for all open positions.
        
        Args:
            pair_data: Dictionary mapping pair names to their data
        """
        for pair_name, position in list(self.positions.items()):
            try:
                # Get current prices
                stock1_data = pair_data[pair_name]['stock1']['Close']
                stock2_data = pair_data[pair_name]['stock2']['Close']
                current_price1 = stock1_data.iloc[-1]
                current_price2 = stock2_data.iloc[-1]
                
                # Calculate PnL
                entry_price1 = position['price1']
                entry_price2 = position['price2']
                shares1 = position['shares1']
                shares2 = position['shares2']
                
                pnl1 = (current_price1 - entry_price1) * shares1
                pnl2 = (current_price2 - entry_price2) * shares2
                total_pnl = pnl1 + pnl2
                
                # Update position info
                position['current_prices'] = {'stock1': current_price1, 'stock2': current_price2}
                position['pnl'] = total_pnl
                position['return'] = total_pnl / position['position_value']
                
                # Check exit conditions
                holding_period = (pd.Timestamp.now() - position['timestamp']).days
                zscore = pair_data[pair_name].get('zscore', 0)
                
                should_exit = (
                    abs(zscore) <= self.exit_threshold or
                    abs(zscore) >= self.stop_loss or
                    holding_period >= self.max_holding_period or
                    (holding_period >= self.min_holding_period and position['return'] >= self.profit_target)
                )
                
                if should_exit:
                    # Close position
                    exit_costs = position['position_value'] * (self.commission + self.slippage)
                    self.capital += position['position_value'] + total_pnl - exit_costs
                    self.trades.append({
                        **position,
                        'exit_time': pd.Timestamp.now(),
                        'exit_prices': {'stock1': current_price1, 'stock2': current_price2},
                        'final_pnl': total_pnl,
                        'exit_costs': exit_costs
                    })
                    del self.positions[pair_name]
                    self.logger.info(f"Closed position for pair {pair_name} with PnL: {total_pnl}")
                
            except Exception as e:
                self.logger.error(f"Error updating position for pair {pair_name}: {str(e)}")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate strategy performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        metrics = {
            'total_trades': len(self.trades),
            'open_positions': len(self.positions),
            'current_capital': self.capital,
            'total_return': (self.capital - self.initial_capital) / self.initial_capital
        }
        
        if self.trades:
            pnls = [trade['final_pnl'] for trade in self.trades if 'final_pnl' in trade]
            metrics.update({
                'total_pnl': sum(pnls),
                'avg_trade_pnl': np.mean(pnls),
                'win_rate': len([p for p in pnls if p > 0]) / len(pnls),
                'max_drawdown': self._calculate_max_drawdown(),
                'sharpe_ratio': self._calculate_sharpe_ratio(pnls)
            })
        
        return metrics
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from trade history."""
        try:
            equity_curve = [self.initial_capital]
            for trade in self.trades:
                equity_curve.append(equity_curve[-1] + trade.get('final_pnl', 0))
            
            equity_curve = np.array(equity_curve)
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (peak - equity_curve) / peak
            return float(np.max(drawdown))
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {str(e)}")
            return 0.0
    
    def _calculate_sharpe_ratio(self, pnls: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from trade PnLs."""
        try:
            if not pnls:
                return 0.0
            
            returns = np.array(pnls) / self.initial_capital
            excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
            if len(excess_returns) < 2:
                return 0.0
            
            return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns, ddof=1)
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize strategy
    strategy = TradingStrategy(
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_loss=3.0,
        position_size=1000.0,
        max_positions=5,
        initial_capital=1000000.0,
        min_holding_period=3,
        max_holding_period=20,
        commission=0.001,
        slippage=0.0005,
        profit_target=2.0,
        risk_per_trade=0.02,
        atr_periods=14,
        position_size_atr_multiplier=1.0,
        max_position_size_pct=0.1,
        min_liquidity_ratio=0.1
    )
    
    # Load ML model
    predictor = None  # Load your trained predictor here
    
    # Load pair data
    pair_data = {}  # Load your pair data here
    hedge_ratios = {}  # Load your hedge ratios here
    
    # Generate signals
    signals = strategy.generate_signals(pair_data, predictor)
    
    # Execute trades
    strategy.execute_trades(signals, pair_data, hedge_ratios)
    
    # Update positions
    strategy.update_positions(pair_data)
    
    # Get performance metrics
    metrics = strategy.get_performance_metrics()
    logging.info(f"Performance metrics: {metrics}")

if __name__ == '__main__':
    main()
