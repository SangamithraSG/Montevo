"""
Volatility Modeling Module

Implements multiple volatility estimation methods:
- Standard Deviation (baseline)
- EWMA (Exponentially Weighted Moving Average)
- GARCH(1,1) (optional advanced model)
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal
from scipy.optimize import minimize


class VolatilityModel:
    """
    Base class for volatility models.
    """
    
    def estimate(self, returns: pd.Series) -> pd.Series:
        """
        Estimate volatility time series from returns.
        
        Args:
            returns: Series of log returns
        
        Returns:
            Series of volatility estimates
        """
        raise NotImplementedError
    
    def get_latest(self, returns: pd.Series) -> float:
        """
        Get the most recent volatility estimate.
        
        Args:
            returns: Series of log returns
        
        Returns:
            Latest volatility value
        """
        volatility_series = self.estimate(returns)
        return float(volatility_series.iloc[-1])


class StandardDeviationVolatility(VolatilityModel):
    """
    Baseline volatility model using rolling standard deviation.
    
    Simple and robust, serves as baseline for comparison.
    """
    
    def __init__(self, window: int = 252):
        """
        Initialize standard deviation volatility model.
        
        Args:
            window: Rolling window size (default 252 = 1 year)
        """
        self.window = window
    
    def estimate(self, returns: pd.Series) -> pd.Series:
        """
        Estimate volatility as rolling standard deviation.
        
        Args:
            returns: Series of log returns
        
        Returns:
            Series of rolling volatility estimates
        """
        clean_returns = returns.dropna()
        rolling_std = clean_returns.rolling(window=self.window, min_periods=min(60, self.window)).std()
        
        # Forward fill to handle initial NaN values
        rolling_std = rolling_std.bfill().fillna(clean_returns.std())
        
        return rolling_std


class EWMAVolatility(VolatilityModel):
    """
    Exponentially Weighted Moving Average volatility model.
    
    Gives more weight to recent observations, making it more responsive
    to recent market conditions. Uses λ = 0.94 as per RiskMetrics standard.
    """
    
    def __init__(self, lambda_param: float = 0.94):
        """
        Initialize EWMA volatility model.
        
        Args:
            lambda_param: Decay factor (0 < λ < 1). Higher values give more weight to history.
                          Default 0.94 follows RiskMetrics methodology.
        """
        if not 0 < lambda_param < 1:
            raise ValueError("Lambda must be between 0 and 1")
        self.lambda_param = lambda_param
    
    def estimate(self, returns: pd.Series) -> pd.Series:
        """
        Estimate volatility using EWMA methodology.
        
        EWMA variance: σ²_t = (1-λ)r²_{t-1} + λσ²_{t-1}
        Volatility: σ_t = sqrt(σ²_t)
        
        Args:
            returns: Series of log returns
        
        Returns:
            Series of EWMA volatility estimates
        """
        clean_returns = returns.dropna().copy()
        
        if len(clean_returns) == 0:
            return pd.Series([], dtype=float)
        
        # Initialize with unconditional variance
        variance = clean_returns.var()
        
        # Calculate squared returns
        squared_returns = clean_returns ** 2
        
        # Apply EWMA recursively
        ewma_variance = squared_returns.ewm(alpha=1 - self.lambda_param, adjust=False).mean()
        
        # Handle initial values
        if ewma_variance.iloc[0] == 0 or np.isnan(ewma_variance.iloc[0]):
            ewma_variance.iloc[0] = variance
        
        # Calculate volatility as square root of variance
        volatility = np.sqrt(ewma_variance)
        
        return volatility


class GARCHVolatility(VolatilityModel):
    """
    GARCH(1,1) volatility model.
    
    Generalized Autoregressive Conditional Heteroskedasticity model.
    More sophisticated than EWMA, allows for mean reversion in variance.
    
    Model: σ²_t = ω + αε²_{t-1} + βσ²_{t-1}
    where: ω > 0, α ≥ 0, β ≥ 0, α + β < 1 (stationarity condition)
    """
    
    def __init__(self):
        """Initialize GARCH(1,1) model."""
        self.omega = None
        self.alpha = None
        self.beta = None
    
    def _garch_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """
        Calculate negative log-likelihood for GARCH(1,1).
        
        Args:
            params: [omega, alpha, beta]
            returns: Array of returns
        
        Returns:
            Negative log-likelihood (for minimization)
        """
        omega, alpha, beta = params
        
        # Ensure stationarity and positivity constraints
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        
        n = len(returns)
        variance = np.zeros(n)
        variance[0] = np.var(returns)
        
        # Iteratively calculate variance
        for t in range(1, n):
            variance[t] = omega + alpha * (returns[t-1] ** 2) + beta * variance[t-1]
        
        # Avoid numerical issues
        variance = np.maximum(variance, 1e-8)
        
        # Calculate log-likelihood (assuming normal errors)
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * variance) + (returns ** 2) / variance)
        
        return -log_likelihood  # Return negative for minimization
    
    def estimate(self, returns: pd.Series) -> pd.Series:
        """
        Estimate GARCH(1,1) parameters and generate volatility series.
        
        Args:
            returns: Series of log returns
        
        Returns:
            Series of GARCH volatility estimates
        """
        clean_returns = returns.dropna().values
        
        if len(clean_returns) < 100:
            # Fallback to EWMA if insufficient data
            ewma = EWMAVolatility()
            return ewma.estimate(returns)
        
        # Initial parameter estimates
        unconditional_var = np.var(clean_returns)
        initial_omega = unconditional_var * 0.1
        initial_alpha = 0.1
        initial_beta = 0.85
        
        initial_params = np.array([initial_omega, initial_alpha, initial_beta])
        
        # Bounds: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
        bounds = [(1e-8, unconditional_var), (0, 0.5), (0, 0.99)]
        
        try:
            # Optimize parameters
            result = minimize(
                self._garch_likelihood,
                initial_params,
                args=(clean_returns,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500}
            )
            
            if not result.success:
                # Fallback to EWMA if optimization fails
                ewma = EWMAVolatility()
                return ewma.estimate(returns)
            
            self.omega, self.alpha, self.beta = result.x
            
            # Generate volatility series
            n = len(clean_returns)
            variance = np.zeros(n)
            variance[0] = unconditional_var
            
            for t in range(1, n):
                variance[t] = self.omega + self.alpha * (clean_returns[t-1] ** 2) + self.beta * variance[t-1]
            
            variance = np.maximum(variance, 1e-8)
            volatility = np.sqrt(variance)
            
            # Create aligned Series
            volatility_series = pd.Series(volatility, index=returns.dropna().index)
            
            return volatility_series
            
        except Exception:
            # Fallback to EWMA on any error
            ewma = EWMAVolatility()
            return ewma.estimate(returns)
    
    def get_parameters(self) -> dict:
        """
        Get estimated GARCH parameters.
        
        Returns:
            Dictionary with omega, alpha, beta, and persistence
        """
        if self.omega is None:
            return {}
        
        return {
            'omega': float(self.omega),
            'alpha': float(self.alpha),
            'beta': float(self.beta),
            'persistence': float(self.alpha + self.beta)
        }


def get_volatility_model(model_type: Literal['SD', 'EWMA', 'GARCH']) -> VolatilityModel:
    """
    Factory function to get volatility model instance.
    
    Args:
        model_type: 'SD', 'EWMA', or 'GARCH'
    
    Returns:
        VolatilityModel instance
    """
    if model_type == 'SD':
        return StandardDeviationVolatility()
    elif model_type == 'EWMA':
        return EWMAVolatility()
    elif model_type == 'GARCH':
        return GARCHVolatility()
    else:
        raise ValueError(f"Unknown volatility model type: {model_type}")
