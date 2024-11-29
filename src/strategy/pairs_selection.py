"""
Pairs selection module for identifying cointegrated stock pairs.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from statsmodels.tsa.stattools import coint, adfuller
from scipy import stats
import logging

class PairsSelector:
    """Handles the selection of trading pairs based on statistical analysis."""
    
    def __init__(self, 
                 price_data: Dict[str, pd.DataFrame],
                 formation_period: int = 126,  # Six months of trading data
                 p_value_threshold: float = 0.10,  # More lenient p-value
                 correlation_threshold: float = 0.5,  # Lower correlation threshold
                 min_half_life: float = 1.0,  # Minimum half-life for mean reversion
                 max_half_life: float = 25.0,  # Maximum half-life for mean reversion
                 config: Dict = {}):  # Configuration for sector-based selection
        """
        Initialize the PairsSelector.
        
        Args:
            price_data: Dict mapping symbols to their price DataFrames
            formation_period: Number of days to use for pair formation
            p_value_threshold: Maximum p-value for cointegration test
            correlation_threshold: Minimum absolute correlation coefficient
            min_half_life: Minimum half-life for mean reversion
            max_half_life: Maximum half-life for mean reversion
            config: Configuration for sector-based selection
        """
        self.price_data = price_data
        self.formation_period = formation_period
        self.p_value_threshold = p_value_threshold
        self.correlation_threshold = correlation_threshold
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def calculate_correlation(self, returns1: pd.Series, returns2: pd.Series) -> float:
        """
        Calculate correlation between two return series.
        
        Args:
            returns1: Return series for first stock
            returns2: Return series for second stock
            
        Returns:
            Correlation coefficient
        """
        return returns1.corr(returns2)
    
    def test_cointegration(self, 
                          prices1: pd.Series, 
                          prices2: pd.Series) -> Tuple[float, float]:
        """
        Test for cointegration between two price series.
        
        Args:
            prices1: Price series for first stock
            prices2: Price series for second stock
            
        Returns:
            Tuple of (test statistic, p-value)
        """
        # Ensure we have matching data
        common_index = prices1.index.intersection(prices2.index)
        prices1 = prices1[common_index]
        prices2 = prices2[common_index]
        
        # Run Engle-Granger cointegration test
        score, pvalue, _ = coint(prices1, prices2)
        return score, pvalue
    
    def calculate_hedge_ratio(self, symbol1: str, symbol2: str) -> float:
        """
        Calculate the hedge ratio between two stocks.
        
        Args:
            symbol1: First stock symbol
            symbol2: Second stock symbol
            
        Returns:
            Hedge ratio
        """
        prices1 = self.price_data[symbol1]['Close']
        prices2 = self.price_data[symbol2]['Close']
        
        # Use last formation_period days
        prices1 = prices1[-self.formation_period:]
        prices2 = prices2[-self.formation_period:]
        
        # Calculate hedge ratio using linear regression
        slope, _, _, _, _ = stats.linregress(prices2, prices1)
        return slope
    
    def find_cointegrated_pairs(self, attempt: int = 0) -> List[Tuple[str, str, float, float]]:
        """
        Find cointegrated pairs with enhanced sector-based selection.
        
        Args:
            attempt: Number of attempts made to find pairs with relaxed constraints
            
        Returns:
            List of tuples containing cointegrated pairs and their metrics
        """
        # Maximum number of attempts to relax constraints
        MAX_ATTEMPTS = 3  # Reduced from 5 to prevent excessive relaxation
        
        # Base thresholds with more lenient values
        base_correlation = 0.60  # Reduced from 0.75
        base_p_value = 0.10    # Increased from 0.05
        min_data_points = 50   # Reduced from 60
        
        # Current thresholds with more aggressive relaxation
        current_correlation = max(base_correlation * (0.90 ** attempt), 0.45)  # Allow down to 0.45
        current_p_value = min(base_p_value * (1.2 ** attempt), 0.15)  # Allow up to 0.15
        
        self.logger.info(f"Attempt {attempt + 1}: correlation >= {current_correlation:.3f}, p-value <= {current_p_value:.3f}")
        
        # Get all symbols with sufficient data
        symbols = []
        for symbol, data in self.price_data.items():
            if len(data['Close'].dropna()) >= min_data_points:
                symbols.append(symbol)
            else:
                self.logger.warning(f"Skipping {symbol} due to insufficient data points")
        
        if len(symbols) < 2:
            self.logger.error("Not enough symbols with sufficient data")
            return []
            
        selected_pairs = []
        sector_pairs_count = {}
        
        # Initialize sector pair counts
        for sector in set(self.get_sector(symbol) for symbol in symbols):
            if sector:  # Only count valid sectors
                sector_pairs_count[sector] = 0
        
        # Test all possible pairs
        for i, symbol1 in enumerate(symbols):
            for symbol2 in symbols[i+1:]:
                # Get sectors
                sector1 = self.get_sector(symbol1)
                sector2 = self.get_sector(symbol2)
                
                # Skip if sectors are invalid or different
                if not sector1 or not sector2 or sector1 != sector2:
                    continue
                    
                # Skip if sector already has max pairs
                if sector_pairs_count[sector1] >= 3:  # Max 3 pairs per sector
                    continue
                
                try:
                    # Get price data and ensure alignment
                    prices1 = self.price_data[symbol1]['Close'][-self.formation_period:]
                    prices2 = self.price_data[symbol2]['Close'][-self.formation_period:]
                    
                    # Ensure data alignment and handle missing values
                    aligned_data = pd.concat([prices1, prices2], axis=1, join='inner')
                    if len(aligned_data) < min_data_points:
                        self.logger.debug(f"Insufficient aligned data points for {symbol1}-{symbol2}")
                        continue
                        
                    prices1 = aligned_data.iloc[:, 0]
                    prices2 = aligned_data.iloc[:, 1]
                    
                    # Calculate returns and handle extreme values
                    returns1 = prices1.pct_change().clip(-0.5, 0.5).dropna()
                    returns2 = prices2.pct_change().clip(-0.5, 0.5).dropna()
                    
                    if len(returns1) < min_data_points - 1:
                        continue
                    
                    # Check correlation
                    correlation = self.calculate_correlation(returns1, returns2)
                    if abs(correlation) < current_correlation:
                        continue
                    
                    # Test cointegration with error handling
                    try:
                        _, p_value = self.test_cointegration(prices1, prices2)
                        if p_value > current_p_value:
                            continue
                    except Exception as e:
                        self.logger.debug(f"Cointegration test failed for {symbol1}-{symbol2}: {str(e)}")
                        continue
                    
                    # Calculate hedge ratio with error handling
                    try:
                        hedge_ratio = self.calculate_hedge_ratio(symbol1, symbol2)
                        spread = prices1 - hedge_ratio * prices2
                    except Exception as e:
                        self.logger.debug(f"Hedge ratio calculation failed for {symbol1}-{symbol2}: {str(e)}")
                        continue
                    
                    # Calculate half-life with bounds checking
                    try:
                        half_life = self.calculate_half_life(spread)
                        if not (self.min_half_life <= half_life <= self.max_half_life):
                            self.logger.debug(f"Pair {symbol1}-{symbol2} skipped due to half-life {half_life:.2f}")
                            continue
                    except Exception as e:
                        self.logger.debug(f"Half-life calculation failed for {symbol1}-{symbol2}: {str(e)}")
                        continue
                    
                    # Calculate additional metrics with error handling
                    vol_ratio = prices1.std() / prices2.std()
                    price_ratio = prices1.iloc[-1] / prices2.iloc[-1]
                    
                    # Skip pairs with extreme ratios
                    if vol_ratio > 5 or vol_ratio < 0.2 or price_ratio > 10 or price_ratio < 0.1:
                        self.logger.debug(f"Pair {symbol1}-{symbol2} skipped due to extreme ratios")
                        continue
                    
                    # Calculate pair score
                    score = self.calculate_pair_score(correlation, p_value, half_life, vol_ratio, price_ratio)
                    
                    selected_pairs.append({
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'correlation': correlation,
                        'p_value': p_value,
                        'half_life': half_life,
                        'hedge_ratio': hedge_ratio,
                        'score': score,
                        'vol_ratio': vol_ratio,
                        'price_ratio': price_ratio
                    })
                    sector_pairs_count[sector1] += 1
                    
                except Exception as e:
                    self.logger.warning(f"Error processing pair {symbol1}-{symbol2}: {str(e)}")
                    continue
        
        if not selected_pairs and attempt < 3:
            self.logger.warning(f"No pairs found on attempt {attempt + 1}. Relaxing constraints...")
            return self.find_cointegrated_pairs(attempt + 1)
        
        if not selected_pairs:
            self.logger.warning("No pairs found after all attempts. Consider adjusting base thresholds.")
            return []
        
        # Sort pairs by score
        selected_pairs.sort(key=lambda x: x['score'], reverse=True)
        
        # Convert to required format
        return [(
            pair['symbol1'],
            pair['symbol2'],
            pair['correlation'],
            pair['p_value']
        ) for pair in selected_pairs]
    
    def calculate_pair_score(self, correlation, p_value, half_life, vol_ratio, price_ratio):
        """
        Calculate a composite score for pair ranking.
        
        Args:
            correlation: Price correlation coefficient
            p_value: Cointegration test p-value
            half_life: Mean reversion half-life
            vol_ratio: Volatility ratio between assets
            price_ratio: Price ratio between assets
        
        Returns:
            float: Composite score (higher is better)
        """
        # Normalize half-life score (prefer shorter half-lives)
        half_life_score = 1.0 / (1.0 + half_life/10.0)
        
        # Normalize p-value (prefer lower p-values)
        p_value_score = 1.0 - p_value
        
        # Volatility ratio score (prefer similar volatilities)
        vol_score = 1.0 - abs(1.0 - vol_ratio)
        
        # Price ratio score (prefer similar price levels)
        price_score = 1.0 - abs(1.0 - price_ratio)
        
        # Weights for different components
        weights = {
            'correlation': 0.3,
            'p_value': 0.2,
            'half_life': 0.2,
            'volatility': 0.15,
            'price': 0.15
        }
        
        # Calculate weighted score
        score = (
            weights['correlation'] * correlation +
            weights['p_value'] * p_value_score +
            weights['half_life'] * half_life_score +
            weights['volatility'] * vol_score +
            weights['price'] * price_score
        )
        
        return score
    
    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate the half-life of mean reversion for a spread series.
        
        Args:
            spread: Spread series
        
        Returns:
            float: Half-life of mean reversion
        """
        # Calculate the lagged spread
        lagged_spread = spread.shift(1)
        
        # Calculate the regression coefficient
        coefficient = lagged_spread.corr(spread)
        
        # Calculate the half-life
        half_life = -np.log(2) / np.log(coefficient)
        
        return half_life
    
    def save_pairs(self, pairs: List[Tuple[str, str, float, float]], output_path: str):
        """
        Save selected pairs to a CSV file.
        
        Args:
            pairs: List of selected pairs
            output_path: Path to save the CSV file
        """
        pairs_df = pd.DataFrame(pairs, columns=['Symbol1', 'Symbol2', 'Correlation', 'Pvalue'])
        
        # Calculate hedge ratios
        pairs_df['Hedge_Ratio'] = pairs_df.apply(
            lambda row: self.calculate_hedge_ratio(row['Symbol1'], row['Symbol2']), 
            axis=1
        )
        
        pairs_df.to_csv(output_path, index=False)
        self.logger.info(f"Saved {len(pairs)} pairs to {output_path}")

    def get_sector(self, symbol):
        sector_mapping = self.config.get('sector_mapping', {})
        for sector, symbols in sector_mapping.items():
            if symbol in symbols:
                return sector
        return 'other'

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Example usage (assuming price data is loaded)
    price_data = {}  # Load your price data here
    config = {
        'sector_mapping': {
            'tech': ['AAPL', 'GOOG', 'MSFT'],
            'finance': ['JPM', 'GS', 'BAC']
        },
        'max_pairs_per_sector': 2
    }
    selector = PairsSelector(price_data, config=config)
    
    # Find cointegrated pairs
    selected_pairs = selector.find_cointegrated_pairs()
    
    # Save results
    selector.save_pairs(selected_pairs, 'data/pairs/selected_pairs.csv')

if __name__ == '__main__':
    main()
