"""
Risk Metrics Calculation Module

Computes various risk and performance metrics:
- Value at Risk (VaR)
- Expected Shortfall (Conditional VaR)
- Probability of loss
- Maximum drawdown
- Sharpe ratio
- Sortino ratio
"""

import numpy as np
from typing import Dict, Optional


def calculate_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR).
    
    VaR is the maximum expected loss at a given confidence level.
    For example, VaR(95%) = -5th percentile of returns.
    
    Args:
        returns: Array of returns (can be log returns or simple returns)
        confidence_level: Confidence level (e.g., 0.95 for 95%)
    
    Returns:
        VaR as a positive number (loss)
    """
    if len(returns) == 0:
        return np.nan
    
    percentile = (1 - confidence_level) * 100
    var = -np.percentile(returns, percentile)
    
    return float(var)


def calculate_expected_shortfall(
    returns: np.ndarray,
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Expected Shortfall (Conditional VaR, CVaR).
    
    Expected Shortfall is the expected loss given that the loss exceeds VaR.
    More conservative than VaR as it considers tail risk.
    
    Formula: ES = E[L | L > VaR]
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
    
    Returns:
        Expected Shortfall as a positive number (loss)
    """
    if len(returns) == 0:
        return np.nan
    
    var = calculate_var(returns, confidence_level)
    
    # Calculate mean of losses exceeding VaR
    tail_losses = returns[returns <= -var]
    
    if len(tail_losses) == 0:
        return var
    
    es = -np.mean(tail_losses)
    
    return float(es)


def calculate_probability_of_loss(final_prices: np.ndarray, initial_price: float) -> float:
    """
    Calculate probability of experiencing a loss.
    
    Args:
        final_prices: Array of final prices from simulations
        initial_price: Initial/current price
    
    Returns:
        Probability of loss (0 to 1)
    """
    if len(final_prices) == 0:
        return np.nan
    
    losses = final_prices < initial_price
    prob_loss = np.mean(losses)
    
    return float(prob_loss)


def calculate_maximum_drawdown(prices: np.ndarray) -> float:
    """
    Calculate maximum drawdown from a price series.
    
    Drawdown: (Peak - Trough) / Peak
    
    Args:
        prices: Array of prices (can be a single path or aggregated)
    
    Returns:
        Maximum drawdown as a percentage (0 to 1)
    """
    if len(prices) == 0:
        return np.nan
    
    # Calculate running maximum (peak)
    running_max = np.maximum.accumulate(prices)
    
    # Calculate drawdown
    drawdown = (running_max - prices) / running_max
    
    max_drawdown = np.max(drawdown)
    
    return float(max_drawdown)


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Sharpe Ratio = (Mean Return - Risk-Free Rate) / Volatility
    
    Args:
        returns: Array of returns (daily)
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Trading periods per year (default 252)
    
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0:
        return np.nan
    
    mean_return = np.mean(returns)
    volatility = np.std(returns)
    
    if volatility == 0:
        return np.nan
    
    # Annualize
    annual_return = mean_return * periods_per_year
    annual_vol = volatility * np.sqrt(periods_per_year)
    annual_rf = risk_free_rate
    
    sharpe = (annual_return - annual_rf) / annual_vol
    
    return float(sharpe)


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sortino ratio.
    
    Sortino Ratio = (Mean Return - Risk-Free Rate) / Downside Deviation
    
    Unlike Sharpe, Sortino only penalizes downside volatility.
    
    Args:
        returns: Array of returns (daily)
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Trading periods per year (default 252)
    
    Returns:
        Annualized Sortino ratio
    """
    if len(returns) == 0:
        return np.nan
    
    mean_return = np.mean(returns)
    
    # Calculate downside deviation (only negative returns)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf if mean_return > risk_free_rate else np.nan
    
    downside_std = np.std(downside_returns)
    
    if downside_std == 0:
        return np.nan
    
    # Annualize
    annual_return = mean_return * periods_per_year
    annual_downside_std = downside_std * np.sqrt(periods_per_year)
    annual_rf = risk_free_rate
    
    sortino = (annual_return - annual_rf) / annual_downside_std
    
    return float(sortino)


def calculate_all_metrics(
    final_prices: np.ndarray,
    initial_price: float,
    paths: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.0
) -> Dict[str, float]:
    """
    Calculate all risk metrics in one call.
    
    Args:
        final_prices: Array of final prices from simulations
        initial_price: Initial/current price
        paths: Optional array of price paths (for drawdown calculation)
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Dictionary with all metrics
    """
    # Calculate returns (log returns)
    returns = np.log(final_prices / initial_price)
    
    metrics = {
        'var_95': calculate_var(returns, 0.95),
        'var_99': calculate_var(returns, 0.99),
        'expected_shortfall_95': calculate_expected_shortfall(returns, 0.95),
        'expected_shortfall_99': calculate_expected_shortfall(returns, 0.99),
        'probability_of_loss': calculate_probability_of_loss(final_prices, initial_price),
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate)
    }
    
    # Calculate maximum drawdown if paths provided
    if paths is not None and len(paths) > 0:
        # Average drawdown across all paths
        drawdowns = []
        for path in paths[:1000]:  # Sample for efficiency
            drawdown = calculate_maximum_drawdown(path)
            if not np.isnan(drawdown):
                drawdowns.append(drawdown)
        
        if drawdowns:
            metrics['max_drawdown'] = float(np.mean(drawdowns))
        else:
            metrics['max_drawdown'] = np.nan
    else:
        metrics['max_drawdown'] = np.nan
    
    return metrics
