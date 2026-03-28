"""
Rheya's Stock Risk Prediction Prototype
Monte Carlo-based stock price interval prediction and portfolio risk simulation system.

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────
#   HELPER FUNCTIONS
# ────────────────────────────────────────────────

def get_currency_symbol(ticker: str) -> str:
    return "₹"  # Always INR regardless of market


# ────────────────────────────────────────────────
#   BUSINESS LOGIC - MAIN ANALYSIS FUNCTION
# ────────────────────────────────────────────────

def run_analysis(
    ticker: str,
    start_date,
    end_date,
    forecast_horizon: int,
    vol_model_type: str,
    confidence_levels: list,
    n_simulations: int,
    run_stress_test: bool,
    run_backtest: bool
):
    """Execute the complete analysis pipeline."""
    
    currency = get_currency_symbol(ticker)

    # Import custom modules inside function (better for Streamlit reload/cloud)
    from data.loader import DataLoader
    from models.returns import calculate_log_returns, calculate_statistics
    from models.volatility import get_volatility_model
    from models.monte_carlo import MonteCarloSimulator
    from models.stress import StressTester
    from backtesting.coverage import Backtester
    from utils.metrics import calculate_all_metrics

    # Initialize data loader
    loader = DataLoader(min_trading_days=1000)
    
    # Load data
    with st.spinner(f"Loading data for {ticker}..."):
        try:
            start_str = start_date.strftime("%Y-%m-%d") if start_date else None
            end_str = end_date.strftime("%Y-%m-%d") if end_date else None
            
            df, metadata = loader.load_clean_data(ticker, start_str, end_str)
            
            st.success(f"✅ Loaded {metadata['trading_days']} trading days from {metadata['start_date'].date()} to {metadata['end_date'].date()}")
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return
    
    # Display data summary
    with st.expander("📊 Data Summary", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Trading Days", metadata['trading_days'])
        with col2:
            st.metric("Start Date", metadata['start_date'].strftime("%Y-%m-%d"))
        with col3:
            st.metric("End Date", metadata['end_date'].strftime("%Y-%m-%d"))
        with col4:
            st.metric("Current Price", f"{currency}{df['Close'].iloc[-1]:.2f}")
        
        st.dataframe(df.tail(10), use_container_width=True)
    
    # Calculate returns and statistics
    prices = df.set_index('Date')['Close']
    log_returns = calculate_log_returns(prices)
    returns_stats = calculate_statistics(log_returns)
    
    # Initialize volatility model
    vol_model = get_volatility_model(vol_model_type)
    
    # Initialize Monte Carlo simulator
    simulator = MonteCarloSimulator(volatility_model=vol_model, random_seed=42)
    
    # Run simulation
    with st.spinner(f"Running {n_simulations:,} Monte Carlo simulations..."):
        try:
            sim_result = simulator.simulate(
                prices=prices,
                forecast_horizon_days=forecast_horizon,
                n_simulations=n_simulations,
                confidence_levels=confidence_levels
            )
        except Exception as e:
            st.error(f"❌ Simulation error: {str(e)}")
            return

    # ── DISPLAY RESULTS ──────────────────────────────────────────────────────

    st.header("📈 Prediction Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Initial Price", f"{currency}{sim_result['initial_price']:.2f}")
    with col2:
        st.metric("Mean Predicted Price", f"{currency}{sim_result['mean_price']:.2f}")
    with col3:
        st.metric("Median Predicted Price", f"{currency}{sim_result['median_price']:.2f}")
    with col4:
        regime = sim_result['parameters']['regime']
        st.metric("Market Regime", regime)
    
    # Confidence intervals
    st.subheader("🎯 Confidence Intervals")
    intervals_data = []
    for cl in sorted(confidence_levels):
        interval = sim_result['confidence_intervals'][cl]
        intervals_data.append({
            'Confidence Level': f"{int(cl*100)}%",
            'Lower Bound': f"{currency}{interval['lower']:.2f}",
            'Upper Bound': f"{currency}{interval['upper']:.2f}",
            'Range': f"{currency}{interval['upper'] - interval['lower']:.2f}"
        })
    
    intervals_df = pd.DataFrame(intervals_data)
    st.dataframe(intervals_df, use_container_width=True, hide_index=True)
    
    # ── VISUALIZATIONS ───────────────────────────────────────────────────────

    st.subheader("📊 Visualizations")
    
    # Histogram of final prices
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=sim_result['final_prices'],
        nbinsx=100,
        name="Price Distribution",
        marker_color='lightblue'
    ))
    
    for cl in sorted(confidence_levels):
        interval = sim_result['confidence_intervals'][cl]
        fig_hist.add_vline(
            x=interval['lower'], line_dash="dash", line_color="red",
            annotation_text=f"{int(cl*100)}% Lower"
        )
        fig_hist.add_vline(
            x=interval['upper'], line_dash="dash", line_color="red",
            annotation_text=f"{int(cl*100)}% Upper"
        )
    
    fig_hist.add_vline(
        x=sim_result['initial_price'],
        line_dash="solid", line_color="black",
        annotation_text="Current Price"
    )
    
    fig_hist.update_layout(
        title="Distribution of Predicted Final Prices",
        xaxis_title=f"Price ({currency})",
        yaxis_title="Frequency",
        height=400
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Sample paths
    st.subheader("🛤️ Sample Simulation Paths")
    n_sample_paths = min(100, n_simulations)
    sample_paths = sim_result['paths'][:n_sample_paths]
    
    fig_paths = go.Figure()
    for i in range(min(50, n_sample_paths)):
        fig_paths.add_trace(go.Scatter(
            x=list(range(len(sample_paths[i]))),
            y=sample_paths[i],
            mode='lines',
            line=dict(width=1, color='lightblue'),
            showlegend=False
        ))
    
    mean_path = np.mean(sample_paths, axis=0)
    fig_paths.add_trace(go.Scatter(
        x=list(range(len(mean_path))),
        y=mean_path,
        mode='lines',
        line=dict(width=2, color='red'),
        name='Mean Path'
    ))
    
    fig_paths.update_layout(
        title=f"Sample of {n_sample_paths} Simulation Paths",
        xaxis_title="Days",
        yaxis_title=f"Price ({currency})",
        height=400
    )
    st.plotly_chart(fig_paths, use_container_width=True)
    
    # ── RISK METRICS ─────────────────────────────────────────────────────────

    st.header("⚠️ Risk Metrics")
    
    metrics = calculate_all_metrics(
        final_prices=sim_result['final_prices'],
        initial_price=sim_result['initial_price'],
        paths=sim_result['paths']
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Value at Risk (VaR)")
        st.metric("VaR 95%", f"{currency}{metrics['var_95']:.2f}")
        st.metric("VaR 99%", f"{currency}{metrics['var_99']:.2f}")
        
        st.subheader("Expected Shortfall")
        st.metric("ES 95%", f"{currency}{metrics['expected_shortfall_95']:.2f}")
        st.metric("ES 99%", f"{currency}{metrics['expected_shortfall_99']:.2f}")
    
    with col2:
        st.subheader("Performance Metrics")
        st.metric("Probability of Loss", f"{metrics['probability_of_loss']*100:.2f}%")
        st.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
        st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.3f}")
    
    # Model parameters
    with st.expander("⚙️ Model Parameters", expanded=False):
        params = sim_result['parameters']
        st.write(f"**Drift (μ):** {params['drift']:.6f}")
        st.write(f"**Volatility (σ):** {params['volatility']:.6f}")
        st.write(f"**Market Regime:** {params['regime']}")
        st.write(f"**Regime Multiplier:** {params['regime_multiplier']:.2f}")
        st.write(f"**Student-t Degrees of Freedom:** {params['degrees_of_freedom']:.1f}")
        st.write(f"**Volatility Model:** {vol_model_type}")
    
    # ── STRESS TESTING (optional) ────────────────────────────────────────────
    if run_stress_test:
        st.header("💥 Stress Testing")
        
        with st.spinner("Running stress scenarios..."):
            stress_tester = StressTester(simulator)
            stress_results = stress_tester.run_stress_scenarios(
                prices=prices,
                forecast_horizon_days=forecast_horizon,
                n_simulations=n_simulations
            )
        
        # (rest of stress test display code - add your original implementation here)
        st.info("Stress test results would be displayed here...")

    # ── BACKTESTING (optional) ───────────────────────────────────────────────
    if run_backtest:
        st.header("🔍 Backtesting & Coverage Analysis")
        st.warning("⚠️ Backtesting may take several minutes. Please be patient.")
        
        with st.spinner("Running backtest (this may take a while)..."):
            backtester = Backtester(simulator, calibration_target=0.98)
            backtest_result = backtester.rolling_backtest(
                prices=prices,
                forecast_horizon_days=forecast_horizon,
                n_simulations=10000  # reduced for speed
            )
        # (rest of backtest display code - add your original implementation here)
        st.info("Backtest results would be displayed here...")


# ────────────────────────────────────────────────
#   STREAMLIT MAIN APPLICATION
# ────────────────────────────────────────────────

def main():
    """Main application entry point."""
    
    st.set_page_config(
        page_title="Stock Risk Prediction System",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .warning-box {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 1rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-header">📈 Stock Risk Prediction System (INR)</div>', unsafe_allow_html=True)
    
    st.markdown("""
    **Monte Carlo-based probabilistic price prediction and portfolio risk simulation**  
    All prices and risk metrics are displayed in **Indian Rupees (₹)**  
    Note: Predictions are probabilistic and should not be interpreted as investment advice.
    """)

    # ── SIDEBAR CONFIGURATION ────────────────────────────────────────────────
    st.sidebar.header("Configuration")
    
    ticker = st.sidebar.text_input(
        "Stock Ticker",
        value="AAPL",
        help="Enter stock ticker (e.g., AAPL, MSFT, RELIANCE.NS, TCS.NS)"
    )
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime(2015, 1, 1),
            help="Leave default for maximum history"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.today(),
            help="End date for historical data"
        )
    
    horizon_options = {
        "1 Day": 1,
        "1 Month": 21,
        "3 Months": 63,
        "6 Months": 126,
        "1 Year": 252
    }
    horizon_label = st.sidebar.selectbox(
        "Forecast Horizon",
        options=list(horizon_options.keys()),
        index=4
    )
    forecast_horizon = horizon_options[horizon_label]
    
    vol_model_type = st.sidebar.selectbox(
        "Volatility Model",
        options=["SD", "EWMA", "GARCH"],
        index=1,
        help="SD: Standard Deviation, EWMA: Exponential Weighted Moving Average, GARCH: GARCH(1,1)"
    )
    
    confidence_levels = st.sidebar.multiselect(
        "Confidence Levels",
        options=[0.90, 0.95, 0.99],
        default=[0.90, 0.95, 0.99]
    )
    
    if not confidence_levels:
        st.sidebar.warning("Please select at least one confidence level")
        return
    
    st.sidebar.subheader("Simulation Parameters")
    n_simulations = st.sidebar.slider(
        "Number of Simulations",
        min_value=5000,
        max_value=100000,
        value=25000,
        step=5000
    )
    
    run_stress_test = st.sidebar.checkbox("Run Stress Tests", value=False)
    run_backtest = st.sidebar.checkbox("Run Backtest", value=False)
    
    # ── RUN BUTTON ───────────────────────────────────────────────────────────
    if st.sidebar.button("🚀 Run Analysis", type="primary"):
        if not confidence_levels:
            st.error("Please select at least one confidence level")
            return
            
        run_analysis(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            forecast_horizon=forecast_horizon,
            vol_model_type=vol_model_type,
            confidence_levels=confidence_levels,
            n_simulations=n_simulations,
            run_stress_test=run_stress_test,
            run_backtest=run_backtest
        )


if __name__ == "__main__":
    main()