"""
Backtesting and Coverage Calibration Module

Implements rolling backtest framework to assess prediction accuracy
and calibrate models to achieve target coverage (≥98% for 95% CI).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from models.monte_carlo import MonteCarloSimulator
from models.returns import calculate_log_returns


class Backtester:
    """
    Backtesting framework for Monte Carlo predictions.
    
    Uses expanding/rolling windows to test predictions against actual outcomes.
    """
    
    def __init__(
        self,
        simulator: MonteCarloSimulator,
        calibration_target: float = 0.98
    ):
        """
        Initialize backtester.
        
        Args:
            simulator: MonteCarloSimulator instance
            calibration_target: Target coverage (e.g., 0.98 for 98%)
        """
        self.simulator = simulator
        self.calibration_target = calibration_target
    
    def rolling_backtest(
        self,
        prices: pd.Series,
        forecast_horizon_days: int,
        min_train_days: int = 252,
        step_size: int = 63,  # Quarterly updates
        n_simulations: int = 10000,
        confidence_level: float = 0.95,
        volatility_multiplier: float = 1.0
    ) -> Dict:
        """
        Perform rolling backtest with expanding window.
        
        For each test point:
        1. Use data up to that point for training
        2. Generate prediction interval
        3. Check if actual future price falls within interval
        4. Calculate coverage
        
        Args:
            prices: Historical price series
            forecast_horizon_days: Days ahead to forecast
            min_train_days: Minimum training period
            step_size: Days between backtest points
            n_simulations: Number of simulations per backtest
            confidence_level: Confidence level for intervals
            volatility_multiplier: Volatility adjustment factor
        
        Returns:
            Dictionary with backtest results
        """
        results = []
        
        # Convert to DataFrame for easier indexing
        price_df = prices.reset_index() if isinstance(prices, pd.Series) else prices.copy()
        price_df = price_df.set_index(price_df.columns[0])
        price_series = price_df.iloc[:, 0]
        
        # Generate test points
        max_test_idx = len(price_series) - forecast_horizon_days - 1
        test_indices = range(min_train_days, max_test_idx, step_size)
        
        if len(test_indices) == 0:
            return {
                'coverage': np.nan,
                'n_tests': 0,
                'results': []
            }
        
        for test_idx in test_indices:
            # Training data (up to test point)
            train_prices = price_series.iloc[:test_idx + 1]
            
            # Actual future price
            actual_future_idx = test_idx + forecast_horizon_days
            if actual_future_idx >= len(price_series):
                continue
            
            actual_price = float(price_series.iloc[actual_future_idx])
            initial_price = float(train_prices.iloc[-1])
            
            # Run simulation
            try:
                sim_result = self.simulator.simulate(
                    prices=train_prices,
                    forecast_horizon_days=forecast_horizon_days,
                    n_simulations=n_simulations,
                    confidence_levels=[confidence_level],
                    volatility_multiplier=volatility_multiplier
                )
                
                # Get prediction interval
                interval = sim_result['confidence_intervals'][confidence_level]
                lower = interval['lower']
                upper = interval['upper']
                
                # Check coverage
                covered = lower <= actual_price <= upper
                
                results.append({
                    'test_date': price_series.index[test_idx],
                    'initial_price': initial_price,
                    'predicted_lower': lower,
                    'predicted_upper': upper,
                    'actual_price': actual_price,
                    'covered': covered,
                    'error': actual_price - initial_price
                })
                
            except Exception as e:
                # Skip this test point if simulation fails
                continue
        
        if len(results) == 0:
            return {
                'coverage': np.nan,
                'n_tests': 0,
                'results': []
            }
        
        # Calculate coverage
        results_df = pd.DataFrame(results)
        coverage = results_df['covered'].mean()
        
        return {
            'coverage': float(coverage),
            'n_tests': len(results),
            'results': results_df.to_dict('records'),
            'results_df': results_df
        }
    
    def calibrate_volatility_multiplier(
        self,
        prices: pd.Series,
        forecast_horizon_days: int,
        target_coverage: float = 0.98,
        confidence_level: float = 0.95,
        initial_multiplier: float = 1.0,
        max_iterations: int = 10
    ) -> Tuple[float, Dict]:
        """
        Calibrate volatility multiplier to achieve target coverage.
        
        Uses binary search to find multiplier that gives desired coverage.
        
        Args:
            prices: Historical price series
            forecast_horizon_days: Forecast horizon
            target_coverage: Target coverage (e.g., 0.98)
            confidence_level: Confidence level for intervals
            initial_multiplier: Starting multiplier
            max_iterations: Maximum calibration iterations
        
        Returns:
            Tuple of (calibrated_multiplier, calibration_results)
        """
        multipliers = [0.8, 1.0, 1.2, 1.5, 2.0]
        coverages = []
        
        for mult in multipliers:
            backtest_result = self.rolling_backtest(
                prices=prices,
                forecast_horizon_days=forecast_horizon_days,
                n_simulations=5000,  # Fewer sims for speed
                confidence_level=confidence_level,
                volatility_multiplier=mult
            )
            coverage = backtest_result['coverage']
            coverages.append(coverage)
            
            if not np.isnan(coverage) and coverage >= target_coverage:
                # Found adequate multiplier
                return mult, {
                    'multiplier': mult,
                    'coverage': coverage,
                    'calibrated': True
                }
        
        # Find best multiplier (closest to target)
        valid_results = [(m, c) for m, c in zip(multipliers, coverages) if not np.isnan(c)]
        
        if not valid_results:
            return initial_multiplier, {
                'multiplier': initial_multiplier,
                'coverage': np.nan,
                'calibrated': False
            }
        
        # Select multiplier with coverage closest to (but >=) target
        best_mult = initial_multiplier
        best_coverage = 0.0
        
        for mult, cov in valid_results:
            if cov >= target_coverage and (best_coverage < target_coverage or cov < best_coverage):
                best_mult = mult
                best_coverage = cov
            elif best_coverage < target_coverage and cov > best_coverage:
                best_mult = mult
                best_coverage = cov
        
        return best_mult, {
            'multiplier': best_mult,
            'coverage': best_coverage,
            'calibrated': best_coverage >= target_coverage
        }
