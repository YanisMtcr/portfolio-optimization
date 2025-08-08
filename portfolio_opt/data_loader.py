import yfinance as yf
import pandas as pd
import numpy as np

class TickerData:
    """
    A class to download and process financial data for multiple tickers using Yahoo Finance.
    
    This class provides functionality to download historical price data for a list of tickers
    and calculate their returns. It uses the yfinance library to fetch data from Yahoo Finance.
    
    Attributes:
        tickers (list): List of stock ticker symbols (e.g., ['AAPL', 'MSFT', 'GOOGL'])
        start_date (str): Start date for data download in 'YYYY-MM-DD' format
        end_date (str): End date for data download (default: "2025-04-30")
        time_interval (str): Data frequency interval (default: "1mo" for monthly data for this project)
        returns_series (pd.DataFrame): Calculated returns for all tickers
    
    Example:
        >>> ticker_data = TickerData(['AAPL', 'MSFT', 'GOOGL'], '2020-01-01')
        >>> returns = ticker_data.returns_series
        >>> print(returns.head())
    """
    
    def __init__(self, tickers, start_date):
        """
        Initialize the TickerData object.
        
        Args:
            tickers (list): List of stock ticker symbols to download data for
            start_date (str): Start date for data download in 'YYYY-MM-DD' format
            
        Example:
            >>> data = TickerData(['AAPL', 'MSFT'], '2020-01-01')
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = "2025-04-30" 
        self.time_interval = "1mo"
        self.returns_series = self.get_returns()

    def get_data(self):
        """
        Download historical price data for the specified tickers.
        
        Downloads OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance
        for all tickers in the specified date range and interval.
        
        Returns:
            pd.DataFrame: Multi-level DataFrame with OHLCV data for all tickers
            
        Example:
            >>> data = self.get_data()
            >>> print(data.head())
        """
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, interval=self.time_interval)
        return data
    
    def get_returns(self):
        """
        Calculate percentage returns for all tickers based on closing prices.
        
        Computes the percentage change in closing prices for each ticker,
        drops any NaN values, and cleans up the DataFrame index and column names.
        
        Returns:
            pd.DataFrame: DataFrame with percentage returns for all tickers,
                         with dates as index and ticker symbols as columns
                         
        Example:
            >>> returns = self.get_returns()
            >>> print(returns.head())
            >>> print(f"Returns shape: {returns.shape}")
        """
        data = self.get_data()
        returns = data["Close"].pct_change().dropna()
        returns.index.name = None
        returns.columns.name = None
        return returns