"""
Monte Carlo Simulation Engine

Implements Geometric Brownian Motion with enhancements:
- Student-t distributed shocks (fat tails)
- Regime-aware drift adjustments
- Volatility scaling during stress periods
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy import stats
from models.returns import calculate_log_returns
from models.volatility import VolatilityModel


class MonteCarloSimulator:
    """
    Monte Carlo simulation engine for stock price prediction.
    
    Base model: Geometric Brownian Motion (GBM)
    dS = μS dt + σS dW
    
    Where:
    - S: Stock price
    - μ: Drift (expected return)
    - σ: Volatility
    - dW: Wiener process (Brownian motion)
    
    Enhanced with:
    - Student-t shocks for fat tails
    - Regime-aware drift adjustments
    - Volatility scaling for stress periods
    """
    
    def __init__(
        self,
        volatility_model: VolatilityModel,
        random_seed: Optional[int] = 42
    ):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            volatility_model: Volatility model instance
            random_seed: Random seed for reproducibility
        """
        self.volatility_model = volatility_model
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def classify_regime(
        self,
        returns: pd.Series,
        lookback: int = 60
    ) -> Tuple[str, float]:
        """
        Classify market regime: Bull, Neutral, or Bear.
        
        Args:
            returns: Recent log returns
            lookback: Number of periods to analyze
        
        Returns:
            Tuple of (regime_name, drift_multiplier)
        """
        recent_returns = returns.tail(lookback).dropna()
        
        if len(recent_returns) < 10:
            return 'Neutral', 1.0
        
        mean_return = recent_returns.mean()
        volatility = recent_returns.std()
        
        # Z-score of mean return
        z_score = mean_return / volatility if volatility > 0 else 0
        
        # Classify based on z-score
        if z_score > 0.5:
            regime = 'Bull'
            multiplier = 1.2  # Slightly higher drift in bull markets
        elif z_score < -0.5:
            regime = 'Bear'
            multiplier = 0.5  # Reduced drift in bear markets
        else:
            regime = 'Neutral'
            multiplier = 1.0
        
        return regime, multiplier
    
    def estimate_parameters(
        self,
        prices: pd.Series,
        volatility_multiplier: float = 1.0
    ) -> Tuple[float, float, str, float]:
        """
        Estimate GBM parameters from historical data.
        
        Args:
            prices: Historical price series
            volatility_multiplier: Multiplier for volatility (stress testing)
        
        Returns:
            Tuple of (drift, volatility, regime, regime_multiplier)
        """
        # Calculate returns
        log_returns = calculate_log_returns(prices)
        
        # Estimate volatility
        volatility_series = self.volatility_model.estimate(log_returns)
        latest_volatility = float(volatility_series.iloc[-1])
        
        # Apply volatility multiplier
        volatility = latest_volatility * volatility_multiplier
        
        # Estimate drift (mean of log returns)
        clean_returns = log_returns.dropna()
        drift = float(clean_returns.mean())
        
        # Classify regime and adjust drift
        regime, regime_mult = self.classify_regime(clean_returns)
        drift = drift * regime_mult
        
        return drift, volatility, regime, regime_mult
    
    def simulate_paths(
        self,
        initial_price: float,
        drift: float,
        volatility: float,
        time_steps: int,
        n_simulations: int,
        degrees_of_freedom: float = 5.0,
        stress_volatility_mult: float = 1.0
    ) -> np.ndarray:
        """
        Simulate stock price paths using GBM with Student-t shocks.
        
        Discrete form: S_{t+1} = S_t * exp((μ - σ²/2) * dt + σ * sqrt(dt) * ε_t)
        
        Where ε_t follows Student-t distribution for fat tails.
        
        Args:
            initial_price: Starting stock price
            drift: Annualized drift (log return per period)
            volatility: Annualized volatility (standard deviation)
            time_steps: Number of time steps (days)
            n_simulations: Number of simulation paths
            degrees_of_freedom: Degrees of freedom for Student-t (lower = fatter tails)
            stress_volatility_mult: Multiplier for volatility (stress testing)
        
        Returns:
            Array of shape (n_simulations, time_steps + 1) with price paths
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        # Apply stress volatility multiplier
        volatility = volatility * stress_volatility_mult
        
        # Initialize paths
        paths = np.zeros((n_simulations, time_steps + 1))
        paths[:, 0] = initial_price
        
        # Time step (daily)
        dt = 1.0 / 252.0  # Assuming 252 trading days per year
        
        # Drift adjustment (Ito's lemma correction)
        adjusted_drift = (drift - 0.5 * volatility ** 2) * dt
        volatility_scaling = volatility * np.sqrt(dt)
        
        # Generate Student-t random variables
        # Normalize to have unit variance
        t_rvs = stats.t.rvs(df=degrees_of_freedom, size=(n_simulations, time_steps))
        t_normalized = t_rvs / np.sqrt(degrees_of_freedom / (degrees_of_freedom - 2))
        
        # Simulate paths
        for t in range(time_steps):
            shocks = t_normalized[:, t]
            paths[:, t + 1] = paths[:, t] * np.exp(adjusted_drift + volatility_scaling * shocks)
        
        return paths
    
    def simulate(
        self,
        prices: pd.Series,
        forecast_horizon_days: int,
        n_simulations: int = 25000,
        confidence_levels: list = [0.90, 0.95, 0.99],
        degrees_of_freedom: float = 5.0,
        volatility_multiplier: float = 1.0,
        stress_volatility_mult: float = 1.0,
        stress_drift_adjustment: float = 0.0
    ) -> dict:
        """
        Run complete Monte Carlo simulation.
        
        Args:
            prices: Historical price series
            forecast_horizon_days: Number of days to forecast
            n_simulations: Number of simulation paths
            confidence_levels: List of confidence levels for intervals
            degrees_of_freedom: Student-t degrees of freedom
            volatility_multiplier: Multiplier for base volatility
            stress_volatility_mult: Additional stress volatility multiplier
            stress_drift_adjustment: Adjustment to drift (e.g., -0.10 for -10%)
        
        Returns:
            Dictionary with simulation results
        """
        # Get initial price
        initial_price = float(prices.iloc[-1])
        
        # Estimate parameters
        drift, volatility, regime, regime_mult = self.estimate_parameters(
            prices,
            volatility_multiplier=volatility_multiplier
        )
        
        # Apply stress drift adjustment
        drift = drift + stress_drift_adjustment
        
        # Simulate paths
        paths = self.simulate_paths(
            initial_price=initial_price,
            drift=drift,
            volatility=volatility,
            time_steps=forecast_horizon_days,
            n_simulations=n_simulations,
            degrees_of_freedom=degrees_of_freedom,
            stress_volatility_mult=stress_volatility_mult
        )
        
        # Extract final prices
        final_prices = paths[:, -1]
        
        # Calculate statistics
        mean_price = float(np.mean(final_prices))
        median_price = float(np.median(final_prices))
        
        # Calculate confidence intervals
        intervals = {}
        for cl in confidence_levels:
            lower_tail = (1 - cl) / 2
            upper_tail = 1 - lower_tail
            intervals[cl] = {
                'lower': float(np.percentile(final_prices, lower_tail * 100)),
                'upper': float(np.percentile(final_prices, upper_tail * 100))
            }
        
        return {
            'initial_price': initial_price,
            'final_prices': final_prices,
            'paths': paths,
            'mean_price': mean_price,
            'median_price': median_price,
            'confidence_intervals': intervals,
            'parameters': {
                'drift': drift,
                'volatility': volatility,
                'regime': regime,
                'regime_multiplier': regime_mult,
                'degrees_of_freedom': degrees_of_freedom
            },
            'n_simulations': n_simulations,
            'forecast_horizon_days': forecast_horizon_days
        }
