<div align="center">

# Montevo 🚀
## Monte Carlo Based Stock Price Interval Prediction and Portfolio Risk Simulation

This application provides probabilistic stock price predictions using Monte Carlo simulation methods with strong statistical foundations. It predicts price intervals (confidence intervals) rather than point estimates, and includes comprehensive risk metrics and stress testing capabilities.

</div>

---

## 🎯 Project Overview

### ⚡ Purpose

- **Predict probabilistic price ranges** (confidence intervals) for any stock
- **Simulate portfolio risk** using Monte Carlo methods
- **Focus on statistical coverage**, not point prediction
- **Target ≥98% empirical coverage** through calibration & backtesting

### ✨ Key Features

- ✅ Unlimited historical data fetch (not capped at 7 years)
- ✅ Support for NSE, NYSE, NASDAQ tickers
- ✅ Multiple volatility models (SD, EWMA, GARCH)
- ✅ Monte Carlo simulation with Student-t distributed shocks
- ✅ Regime-aware drift adjustments (Bull/Neutral/Bear)
- ✅ Comprehensive risk metrics (VaR, Expected Shortfall, Sharpe, Sortino)
- ✅ Rolling backtesting with coverage validation
- ✅ Stress testing under extreme scenarios
- ✅ Interactive Streamlit frontend

---

## 📋 Requirements

### Python Version
- Python 3.8 or higher

### Dependencies
See `requirements.txt` for complete list:
- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- yfinance >= 0.2.28
- plotly >= 5.17.0
- scipy >= 1.11.0

---

## 🚀 Installation

1. **Clone or download this repository**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Access the application:**
   - The app will open in your default web browser
   - Default URL: http://localhost:8501

---

## 📊 Output Metrics

### Price Metrics

- **Mean final price**: Average across all simulations
- **Median final price**: 50th percentile
- **Confidence intervals**: 90%, 95%, 99%
  - Lower and upper bounds
  - Range (difference)

### Risk Metrics

1. **Value at Risk (VaR)**
   - VaR 95%: Maximum expected loss at 95% confidence
   - VaR 99%: Maximum expected loss at 99% confidence
   - Expressed as dollar amount or percentage

2. **Expected Shortfall (Conditional VaR)**
   - Expected loss given that loss exceeds VaR
   - More conservative than VaR
   - Considers tail risk

3. **Probability of Loss**
   - Probability that final price < initial price
   - Percentage (0 to 100%)

4. **Maximum Drawdown**
   - Largest peak-to-trough decline
   - Percentage of peak value

5. **Sharpe Ratio**
   - Risk-adjusted return measure
   - Formula: (Return - Risk-Free Rate) / Volatility
   - Higher is better

6. **Sortino Ratio**
   - Like Sharpe, but only penalizes downside volatility
   - Formula: (Return - Risk-Free Rate) / Downside Deviation
   - More appropriate for asymmetric returns












