"""
Returns Calculation Module

Handles computation of log returns with support for full-sample and rolling windows.
Uses logarithmic returns for mathematical consistency in continuous-time models.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate logarithmic returns from price series.
    
    Log returns are used because:
    1. They are time-additive (multi-period return = sum of single-period returns)
    2. They are symmetric (multiplying by -1 gives the inverse)
    3. They are bounded below by -∞ but not bounded above
    4. They align with continuous-time finance theory (GBM, etc.)
    
    Formula: r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
    
    Args:
        prices: Series of closing prices (must be positive)
    
    Returns:
        Series of log returns (first value will be NaN)
    """
    if (prices <= 0).any():
        raise ValueError("Prices must be strictly positive for log returns")
    
    log_prices = np.log(prices)
    log_returns = log_prices.diff()
    
    return log_returns


def calculate_rolling_returns(
    prices: pd.Series,
    window: int,
    min_periods: Optional[int] = None
) -> pd.Series:
    """
    Calculate rolling window log returns.
    
    For each period, computes log return over the previous 'window' periods.
    Useful for adaptive models that adjust to recent market conditions.
    
    Args:
        prices: Series of closing prices
        window: Rolling window size (number of periods)
        min_periods: Minimum periods required (default: window)
    
    Returns:
        Series of rolling log returns
    """
    if min_periods is None:
        min_periods = window
    
    log_prices = np.log(prices)
    
    # Rolling difference over window
    rolling_log_returns = log_prices.diff(window)
    
    return rolling_log_returns


def calculate_statistics(returns: pd.Series) -> dict:
    """
    Calculate descriptive statistics for return series.
    
    Args:
        returns: Series of log returns (may contain NaN)
    
    Returns:
        Dictionary with statistical measures
    """
    clean_returns = returns.dropna()
    
    if len(clean_returns) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'skewness': np.nan,
            'kurtosis': np.nan,
            'count': 0
        }
    
    mean = clean_returns.mean()
    std = clean_returns.std()
    skew = clean_returns.skew()
    kurt = clean_returns.kurtosis()
    
    return {
        'mean': float(mean),
        'std': float(std),
        'skewness': float(skew),
        'kurtosis': float(kurt),
        'count': len(clean_returns)
    }


def annualize_return(log_return: float, periods_per_year: int = 252) -> float:
    """
    Annualize a log return.
    
    Since log returns are additive, annualization is simply multiplication.
    
    Args:
        log_return: Daily log return
        periods_per_year: Trading days per year (default 252)
    
    Returns:
        Annualized log return
    """
    return log_return * periods_per_year


def annualize_volatility(volatility: float, periods_per_year: int = 252) -> float:
    """
    Annualize volatility using square-root-of-time rule.
    
    Args:
        volatility: Daily volatility (standard deviation)
        periods_per_year: Trading days per year (default 252)
    
    Returns:
        Annualized volatility
    """
    return volatility * np.sqrt(periods_per_year)
