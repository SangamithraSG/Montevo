"""
Stress Testing Module

Implements predefined stress scenarios to test portfolio resilience
under extreme market conditions.
"""

import numpy as np
from typing import Dict, List
from models.monte_carlo import MonteCarloSimulator
from utils.metrics import calculate_all_metrics


class StressTester:
    """
    Stress testing framework for Monte Carlo simulations.
    """
    
    def __init__(self, simulator: MonteCarloSimulator):
        """
        Initialize stress tester.
        
        Args:
            simulator: MonteCarloSimulator instance
        """
        self.simulator = simulator
    
    def run_stress_scenarios(
        self,
        prices,
        forecast_horizon_days: int,
        n_simulations: int = 25000,
        base_volatility_mult: float = 1.0
    ) -> Dict:
        """
        Run all predefined stress scenarios.
        
        Scenarios:
        1. Baseline (normal conditions)
        2. Moderate stress (volatility × 1.5)
        3. Severe stress (volatility × 3, crash)
        4. Negative drift (-10% annualized)
        5. Combined crash (volatility × 3, drift -15%)
        
        Args:
            prices: Historical price series
            forecast_horizon_days: Forecast horizon
            n_simulations: Number of simulations
            base_volatility_mult: Base volatility multiplier
        
        Returns:
            Dictionary with stress test results
        """
        scenarios = {
            'Baseline': {
                'volatility_mult': 1.0 * base_volatility_mult,
                'drift_adjustment': 0.0
            },
            'Moderate Stress (σ × 1.5)': {
                'volatility_mult': 1.5 * base_volatility_mult,
                'drift_adjustment': 0.0
            },
            'Severe Stress (σ × 3)': {
                'volatility_mult': 3.0 * base_volatility_mult,
                'drift_adjustment': 0.0
            },
            'Negative Drift (-10%)': {
                'volatility_mult': 1.0 * base_volatility_mult,
                'drift_adjustment': -0.10 / 252  # Daily adjustment
            },
            'Combined Crash (σ × 3, μ = -15%)': {
                'volatility_mult': 3.0 * base_volatility_mult,
                'drift_adjustment': -0.15 / 252  # Daily adjustment
            }
        }
        
        results = {}
        
        for scenario_name, params in scenarios.items():
            try:
                # Run simulation
                sim_result = self.simulator.simulate(
                    prices=prices,
                    forecast_horizon_days=forecast_horizon_days,
                    n_simulations=n_simulations,
                    volatility_multiplier=params['volatility_mult'],
                    stress_drift_adjustment=params['drift_adjustment']
                )
                
                # Calculate metrics
                metrics = calculate_all_metrics(
                    final_prices=sim_result['final_prices'],
                    initial_price=sim_result['initial_price'],
                    paths=sim_result['paths']
                )
                
                # Combine results
                results[scenario_name] = {
                    'simulation': sim_result,
                    'metrics': metrics,
                    'parameters': params
                }
                
            except Exception as e:
                results[scenario_name] = {
                    'error': str(e)
                }
        
        return results
    
    def compare_scenarios(self, stress_results: Dict) -> Dict:
        """
        Compare metrics across stress scenarios.
        
        Args:
            stress_results: Results from run_stress_scenarios
        
        Returns:
            Dictionary with comparison metrics
        """
        comparison = {}
        
        baseline = stress_results.get('Baseline')
        if baseline is None or 'metrics' not in baseline:
            return comparison
        
        baseline_metrics = baseline['metrics']
        
        for scenario_name, result in stress_results.items():
            if scenario_name == 'Baseline' or 'metrics' not in result:
                continue
            
            metrics = result['metrics']
            comparison[scenario_name] = {
                'var_95_change': metrics.get('var_95', 0) - baseline_metrics.get('var_95', 0),
                'var_99_change': metrics.get('var_99', 0) - baseline_metrics.get('var_99', 0),
                'es_95_change': metrics.get('expected_shortfall_95', 0) - baseline_metrics.get('expected_shortfall_95', 0),
                'prob_loss_change': metrics.get('probability_of_loss', 0) - baseline_metrics.get('probability_of_loss', 0)
            }
        
        return comparison
