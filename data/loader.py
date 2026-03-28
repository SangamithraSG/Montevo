"""
Data Loading and Cleaning Module

Handles fetching stock data from yfinance and applying data quality checks.
Supports NSE, NYSE, and NASDAQ tickers with unlimited historical data.
"""

import pandas as pd
import yfinance as yf
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Handles stock data fetching and cleaning with strict quality checks.
    """
    
    def __init__(self, min_trading_days: int = 1000):
        """
        Initialize the data loader.
        
        Args:
            min_trading_days: Minimum number of trading days required
        """
        self.min_trading_days = min_trading_days
    
    def fetch_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch stock data from yfinance with maximum available history.
        
        Args:
            ticker: Stock ticker symbol (supports NSE, NYSE, NASDAQ)
            start_date: Start date (YYYY-MM-DD) or None for max history
            end_date: End date (YYYY-MM-DD) or None for today
        
        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume
        """
        try:
            # Determine ticker format
            ticker_upper = ticker.upper()
            
            # If ticker already has exchange suffix, use it directly
            if '.' in ticker_upper:
                yf_ticker = ticker_upper
            else:
                # Try without suffix first (most common for US stocks)
                # If it fails, user can manually add .NS for NSE stocks
                yf_ticker = ticker_upper
            
            stock = yf.Ticker(yf_ticker)
            
            # Fetch maximum available data
            if start_date is None and end_date is None:
                hist = stock.history(period="max")
            else:
                hist = stock.history(start=start_date, end=end_date)
            
            if hist.empty:
                raise ValueError(f"No data found for ticker: {ticker}")
            
            # Reset index to make Date a column
            hist = hist.reset_index()
            
            # Select required columns
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in hist.columns for col in required_cols):
                raise ValueError(f"Missing required columns. Available: {hist.columns.tolist()}")
            
            df = hist[required_cols].copy()
            
            return df
            
        except Exception as e:
            raise ValueError(f"Error fetching data for {ticker}: {str(e)}")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply strict data quality checks and cleaning.
        
        Rules:
        - Remove rows with Close <= 0
        - Remove rows with Volume <= 0
        - Remove rows with High < Low
        - Sort by Date
        - Forward-fill only missing values (not entire rows)
        
        Args:
            df: Raw stock data DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Convert Date to datetime if string
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by Date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Store original length
        original_len = len(df)
        
        # Remove invalid prices
        df = df[df['Close'] > 0].copy()
        
        # Remove invalid volume
        df = df[df['Volume'] > 0].copy()
        
        # Remove invalid high-low relationships
        df = df[df['High'] >= df['Low']].copy()
        
        # Check for missing values in price columns
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            if df[col].isna().any():
                df[col] = df[col].ffill()
        
        # Ensure no NaN values remain in critical columns
        df = df.dropna(subset=['Date', 'Close', 'Volume'])
        
        # Remove duplicate dates (keep first occurrence)
        df = df.drop_duplicates(subset='Date', keep='first').reset_index(drop=True)
        
        # Final validation: check minimum trading days
        if len(df) < self.min_trading_days:
            raise ValueError(
                f"Insufficient data after cleaning: {len(df)} days "
                f"(minimum required: {self.min_trading_days}). "
                f"Original data had {original_len} rows."
            )
        
        return df
    
    def load_clean_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Fetch and clean stock data in one step.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD) or None
            end_date: End date (YYYY-MM-DD) or None
        
        Returns:
            Tuple of (cleaned DataFrame, metadata dict)
        """
        # Fetch data
        raw_df = self.fetch_data(ticker, start_date, end_date)
        
        # Clean data
        cleaned_df = self.clean_data(raw_df)
        
        # Generate metadata
        metadata = {
            'ticker': ticker,
            'start_date': cleaned_df['Date'].min(),
            'end_date': cleaned_df['Date'].max(),
            'trading_days': len(cleaned_df),
            'original_rows': len(raw_df),
            'removed_rows': len(raw_df) - len(cleaned_df)
        }
        
        return cleaned_df, metadata
