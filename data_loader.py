import yfinance as yf
import pandas as pd
import os
import datetime
import requests
from bs4 import BeautifulSoup
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_directory_exists(directory):
    """Ensure that the specified directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def get_stock_data(ticker='AAPL', start='2020-01-01', end=None, interval='1d', save=True):
    """
    Download historical stock price data with error handling and retries.
    
    Args:
        ticker (str): Stock ticker symbol
        start (str): Start date in 'YYYY-MM-DD' format
        end (str): End date in 'YYYY-MM-DD' format, defaults to today
        interval (str): Data interval ('1d', '1wk', '1mo')
        save (bool): Whether to save the data to CSV
        
    Returns:
        pd.DataFrame: Historical stock data
    """
    if end is None:
        end = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # Try up to 3 times to download data
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading data for {ticker} from {start} to {end}")
            df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
            
            if df.empty:
                logger.warning(f"No data found for {ticker}")
                if attempt < max_retries - 1:
                    time.sleep(2)  # Wait before retrying
                    continue
                return None
            
            # Reset index to make Date a column
            df.reset_index(inplace=True)
            
            # Ensure data is sorted by date
            df = df.sort_values('Date')
            
            # Add ticker column for multi-symbol datasets
            df['Ticker'] = ticker
            
            if save:
                ensure_directory_exists('data/raw')
                filename = f"data/raw/{ticker}_data.csv"
                df.to_csv(filename, index=False)
                logger.info(f"Saved data to {filename}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error downloading {ticker} data (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retrying
            else:
                logger.error(f"Failed to download data for {ticker} after {max_retries} attempts")
                return None

def get_market_indices(start='2020-01-01', end=None, save=True):
    """
    Download major market indices for comparison.
    
    Args:
        start (str): Start date in 'YYYY-MM-DD' format
        end (str): End date in 'YYYY-MM-DD' format, defaults to today
        save (bool): Whether to save the data to CSV
        
    Returns:
        dict: Dictionary of DataFrames with index data
    """
    indices = {
        'S&P 500': '^GSPC',
        'Dow Jones': '^DJI',
        'NASDAQ': '^IXIC',
        'Russell 2000': '^RUT',
        'VIX': '^VIX'
    }
    
    results = {}
    for name, symbol in indices.items():
        try:
            df = get_stock_data(symbol, start, end, save=False)
            if df is not None:
                results[name] = df
                
                if save:
                    ensure_directory_exists('data/raw')
                    filename = f"data/raw/{symbol.replace('^', '')}_data.csv"
                    df.to_csv(filename, index=False)
                    logger.info(f"Saved {name} data to {filename}")
        except Exception as e:
            logger.error(f"Error downloading {name} data: {str(e)}")
    
    return results

def get_forex_data(pair='EURUSD=X', start='2020-01-01', end=None, save=True):
    """
    Download forex data for currency pairs.
    
    Args:
        pair (str): Forex pair symbol
        start (str): Start date in 'YYYY-MM-DD' format
        end (str): End date in 'YYYY-MM-DD' format, defaults to today
        save (bool): Whether to save the data to CSV
        
    Returns:
        pd.DataFrame: Forex historical data
    """
    return get_stock_data(pair, start, end, save=save)

def get_crypto_data(symbol='BTC-USD', start='2020-01-01', end=None, save=True):
    """
    Download cryptocurrency historical data.
    
    Args:
        symbol (str): Cryptocurrency symbol
        start (str): Start date in 'YYYY-MM-DD' format
        end (str): End date in 'YYYY-MM-DD' format, defaults to today
        save (bool): Whether to save the data to CSV
        
    Returns:
        pd.DataFrame: Cryptocurrency historical data
    """
    return get_stock_data(symbol, start, end, save=save)

def get_multiple_stocks(tickers=['AAPL', 'MSFT', 'GOOGL'], start='2020-01-01', end=None, save=True):
    """
    Download data for multiple stocks and combine into a single DataFrame.
    
    Args:
        tickers (list): List of ticker symbols
        start (str): Start date in 'YYYY-MM-DD' format
        end (str): End date in 'YYYY-MM-DD' format, defaults to today
        save (bool): Whether to save individual stock data to CSV
        
    Returns:
        pd.DataFrame: Combined DataFrame with all stock data
    """
    all_data = []
    
    for ticker in tickers:
        df = get_stock_data(ticker, start, end, save=save)
        if df is not None:
            all_data.append(df)
    
    if not all_data:
        logger.warning("No data was successfully retrieved")
        return None
    
    # Combine all stock data
    combined_df = pd.concat(all_data)
    
    # Save combined data if requested
    if save:
        ensure_directory_exists('data/raw')
        tickers_str = '_'.join(tickers) if len(tickers) <= 3 else f"{tickers[0]}_{tickers[1]}_and_{len(tickers)-2}_more"
        filename = f"data/raw/combined_{tickers_str}_data.csv"
        combined_df.to_csv(filename, index=False)
        logger.info(f"Saved combined data to {filename}")
    
    return combined_df

def fetch_stock_news(ticker='AAPL', days_back=30):
    """
    Simulates fetching news for a given ticker.
    
    In a production system, this would connect to a proper financial news API.
    For demonstration purposes, we're generating realistic but fictional news.
    
    Args:
        ticker (str): Stock ticker symbol
        days_back (int): Number of days of historical news to fetch
        
    Returns:
        pd.DataFrame: News data with dates, headlines, and sources
    """
    # Generate realistic dates
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days_back)
    
    # Generate random dates within the range (more news on trading days)
    dates = []
    current = start_date
    while current <= end_date:
        # More news on weekdays
        if current.weekday() < 5:  # Monday to Friday
            # Generate 1-3 news items for weekdays
            for _ in range(1, 4):
                if np.random.random() > 0.3:  # 70% chance for news on a weekday
                    dates.append(current)
        else:
            # Generate 0-1 news items for weekends
            if np.random.random() > 0.7:  # 30% chance for news on weekends
                dates.append(current)
        
        current += datetime.timedelta(days=1)
    
    # News templates categorized by sentiment
    positive_templates = [
        "{ticker} reports strong quarterly earnings, exceeding expectations",
        "{ticker} stock surges following positive analyst reports",
        "New product launch boosts {ticker} market position",
        "{ticker} announces expansion into emerging markets",
        "Investors bullish on {ticker} following strategic acquisition",
        "{ticker} increases dividend, signaling strong financial health",
        "Analysts upgrade {ticker} rating, citing growth potential",
        "{ticker} exceeds revenue forecasts, shares climb",
        "{ticker} announces share buyback program",
        "{ticker} forms strategic partnership with industry leader"
    ]
    
    neutral_templates = [
        "{ticker} quarterly results meet expectations",
        "{ticker} maintains market position amid industry changes",
        "Analysts have mixed views on {ticker}'s latest announcement",
        "{ticker} restructures management team",
        "{ticker} holds annual shareholder meeting",
        "Industry report shows stable outlook for {ticker}",
        "{ticker} to present at upcoming investor conference",
        "{ticker} maintains dividend levels",
        "New {ticker} CEO outlines strategic vision",
        "{ticker} completes previously announced acquisition"
    ]
    
    negative_templates = [
        "{ticker} misses earnings expectations, shares decline",
        "Analysts downgrade {ticker} amid growth concerns",
        "{ticker} faces increased competition in key markets",
        "Regulatory challenges ahead for {ticker}, analysts warn",
        "{ticker} announces restructuring and job cuts",
        "Supply chain issues impact {ticker} production capacity",
        "{ticker} lowers guidance for upcoming quarter",
        "Investor concerns grow over {ticker}'s debt levels",
        "{ticker} faces lawsuit over business practices",
        "{ticker} product recall affects quarterly outlook"
    ]
    
    # News sources
    sources = [
        "Financial Times", "Wall Street Journal", "CNBC", 
        "Bloomberg", "Reuters", "MarketWatch", "Investor's Business Daily",
        "Barron's", "The Street", "Motley Fool", "Seeking Alpha"
    ]
    
    # Generate news data
    news_data = []
    
    for date in dates:
        # Decide sentiment
        sentiment_choice = np.random.choice(['positive', 'neutral', 'negative'], p=[0.4, 0.4, 0.2])
        
        if sentiment_choice == 'positive':
            headline = np.random.choice(positive_templates).format(ticker=ticker)
        elif sentiment_choice == 'neutral':
            headline = np.random.choice(neutral_templates).format(ticker=ticker)
        else:
            headline = np.random.choice(negative_templates).format(ticker=ticker)
        
        # Add some randomness to the time
        hours = np.random.randint(8, 20)  # Business hours
        minutes = np.random.randint(0, 60)
        date_with_time = date.replace(hour=hours, minute=minutes)
        
        news_data.append({
            'date': date_with_time,
            'headline': headline,
            'source': np.random.choice(sources),
            'sentiment_category': sentiment_choice  # This would normally be derived from analysis
        })
    
    # Convert to DataFrame and sort by date
    news_df = pd.DataFrame(news_data)
    news_df = news_df.sort_values('date')
    
    # Save to CSV
    ensure_directory_exists('data/raw')
    filename = f"data/raw/{ticker}_news.csv"
    news_df.to_csv(filename, index=False)
    logger.info(f"Generated and saved news data to {filename}")
    
    return news_df

if __name__ == "__main__":
    # Import numpy here to avoid global import when module is imported elsewhere
    import numpy as np
    
    # Example usage of the enhanced data loader
    ensure_directory_exists('data/raw')
    
    # Get stock data for multiple tech companies
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    df = get_multiple_stocks(tech_stocks, start='2022-01-01')
    print(f"Downloaded data for {len(tech_stocks)} tech stocks")
    
    # Get market indices for comparison
    indices = get_market_indices(start='2022-01-01')
    print(f"Downloaded data for {len(indices)} market indices")
    
    # Get some cryptocurrency data
    crypto_df = get_crypto_data('BTC-USD', start='2022-01-01')
    print("Downloaded Bitcoin data")
    
    # Generate news data
    news_df = fetch_stock_news('AAPL', days_back=60)
    print(f"Generated {len(news_df)} news items for AAPL")