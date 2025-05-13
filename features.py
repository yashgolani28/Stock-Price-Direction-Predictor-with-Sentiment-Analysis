import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator, AwesomeOscillatorIndicator
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator, IchimokuIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator, ChaikinMoneyFlowIndicator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging
import os
from pathlib import Path

# Set up logging with proper configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('feature_engineering.log')
    ]
)
logger = logging.getLogger(__name__)

class FeatureEngineering:
    """Class to handle all feature engineering operations for stock data."""
    
    def __init__(self):
        """Initialize the feature engineering class."""
        self.analyzer = SentimentIntensityAnalyzer()
        
    @staticmethod
    def ensure_directory_exists(directory):
        """Ensure that the specified directory exists.
        
        Args:
            directory (str): Path to the directory
        """
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True)
            logger.info(f"Created directory: {directory}")
            
    def preprocess_stock_data(self, df):
        """Preprocess stock data for model input.
        
        Args:
            df (pandas.DataFrame): Raw stock data
            
        Returns:
            pandas.DataFrame: Preprocessed stock data
        """
        try:
            # Make a copy to avoid modifying the original dataframe
            df = df.copy()
            
            # Convert columns to numeric
            numeric_columns = ['Close', 'High', 'Low', 'Open', 'Volume']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    logger.warning(f"Column '{col}' not found in DataFrame")
            
            # Handle missing values
            df.fillna(method='ffill', inplace=True)
            df.dropna(subset=['Close', 'High', 'Low', 'Volume'], inplace=True)
            
            # Ensure all required columns exist
            required_columns = ['Close', 'High', 'Low', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Convert Date to datetime if it exists and set as index (optional)
            if 'Date' in df.columns:
                try:
                    df['Date'] = pd.to_datetime(df['Date'])
                except Exception as e:
                    logger.warning(f"Could not convert 'Date' column to datetime: {str(e)}")
            
            logger.info("Preprocessed stock data successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing stock data: {str(e)}")
            raise
            
    def compute_price_change(self, df):
        """Calculate price change as percentage change.
        
        Args:
            df (pandas.DataFrame): Stock data
            
        Returns:
            pandas.DataFrame: Stock data with price change
        """
        try:
            # Make a copy to avoid modifying the original dataframe
            df = df.copy()
            
            # Calculate price changes
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_1d'] = df['Close'].pct_change(periods=1)
            df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
            df['Price_Change_10d'] = df['Close'].pct_change(periods=10)
            
            # Calculate returns
            df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
            
            # Target variable for prediction (next day's price change)
            df['Next_Day_Return'] = df['Price_Change'].shift(-1)
            
            logger.info("Computed price changes successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error computing price changes: {str(e)}")
            raise
            
    def compute_sentiment_score(self, df):
        """Calculate sentiment score based on news sentiment.
        
        Args:
            df (pandas.DataFrame): Stock data with news sentiment
            
        Returns:
            pandas.DataFrame: Stock data with sentiment scores
        """
        try:
            # Make a copy to avoid modifying the original dataframe
            df = df.copy()
            
            if 'News_Sentiment' in df.columns:
                logger.info("Calculating sentiment scores...")
                
                # Convert to string to ensure compatibility with VADER
                df['News_Sentiment'] = df['News_Sentiment'].astype(str)
                
                # Apply sentiment analysis
                df['sentiment_score'] = df['News_Sentiment'].apply(
                    lambda x: self.analyzer.polarity_scores(x)['compound'] if x else 0
                )
                
                # Add sentiment categories
                df['sentiment_positive'] = df['News_Sentiment'].apply(
                    lambda x: self.analyzer.polarity_scores(x)['pos'] if x else 0
                )
                df['sentiment_negative'] = df['News_Sentiment'].apply(
                    lambda x: self.analyzer.polarity_scores(x)['neg'] if x else 0
                )
                df['sentiment_neutral'] = df['News_Sentiment'].apply(
                    lambda x: self.analyzer.polarity_scores(x)['neu'] if x else 0
                )
                
                # Create sentiment signal (1 for positive, -1 for negative, 0 for neutral)
                df['sentiment_signal'] = np.where(df['sentiment_score'] > 0.05, 1, 
                                            np.where(df['sentiment_score'] < -0.05, -1, 0))
                
                logger.info("Sentiment scores calculated successfully")
            else:
                logger.warning("Column 'News_Sentiment' not found. Skipping sentiment score calculation.")
                
            return df
            
        except Exception as e:
            logger.error(f"Error computing sentiment scores: {str(e)}")
            raise
            
    def compute_momentum_indicators(self, df):
        """Calculate momentum indicators.
        
        Args:
            df (pandas.DataFrame): Stock data
            
        Returns:
            pandas.DataFrame: Stock data with momentum indicators
        """
        try:
            # Make a copy to avoid modifying the original dataframe
            df = df.copy()
            
            # RSI (Relative Strength Index)
            for window in [7, 14, 21]:
                df[f'RSI_{window}'] = RSIIndicator(
                    close=df['Close'], window=window, fillna=True
                ).rsi()
            
            # Stochastic Oscillator
            stoch = StochasticOscillator(
                high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3, fillna=True
            )
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
            
            # Rate of Change
            for window in [5, 10, 20]:
                df[f'ROC_{window}'] = ROCIndicator(
                    close=df['Close'], window=window, fillna=True
                ).roc()
            
            # Williams %R
            df['Williams_R'] = WilliamsRIndicator(
                high=df['High'], low=df['Low'], close=df['Close'], lbp=14, fillna=True
            ).williams_r()
            
            # Awesome Oscillator
            df['Awesome_Osc'] = AwesomeOscillatorIndicator(
                high=df['High'], low=df['Low'], window1=5, window2=34, fillna=True
            ).awesome_oscillator()
            
            # Create signals
            df['RSI_Overbought'] = np.where(df['RSI_14'] > 70, 1, 0)
            df['RSI_Oversold'] = np.where(df['RSI_14'] < 30, 1, 0)
            df['Stoch_Overbought'] = np.where(df['Stoch_K'] > 80, 1, 0)
            df['Stoch_Oversold'] = np.where(df['Stoch_K'] < 20, 1, 0)
            
            logger.info("Computed momentum indicators successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error computing momentum indicators: {str(e)}")
            raise
            
    def compute_trend_indicators(self, df):
        """Calculate trend indicators.
        
        Args:
            df (pandas.DataFrame): Stock data
            
        Returns:
            pandas.DataFrame: Stock data with trend indicators
        """
        try:
            # Make a copy to avoid modifying the original dataframe
            df = df.copy()
            
            # Simple Moving Averages (SMA)
            for window in [10, 20, 50, 100, 200]:
                df[f'SMA_{window}'] = SMAIndicator(
                    close=df['Close'], window=window, fillna=True
                ).sma_indicator()
            
            # Exponential Moving Averages (EMA)
            for window in [10, 20, 50, 100, 200]:
                df[f'EMA_{window}'] = EMAIndicator(
                    close=df['Close'], window=window, fillna=True
                ).ema_indicator()
            
            # MACD (Moving Average Convergence Divergence)
            macd = MACD(
                close=df['Close'], window_slow=26, window_fast=12, window_sign=9, fillna=True
            )
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Hist'] = macd.macd_diff()
            
            # ADX (Average Directional Index)
            adx = ADXIndicator(
                high=df['High'], low=df['Low'], close=df['Close'], window=14, fillna=True
            )
            df['ADX'] = adx.adx()
            df['ADX_Pos'] = adx.adx_pos()
            df['ADX_Neg'] = adx.adx_neg()
            
            # CCI (Commodity Channel Index)
            df['CCI'] = CCIIndicator(
                high=df['High'], low=df['Low'], close=df['Close'], window=20, constant=0.015, fillna=True
            ).cci()
            
            # Ichimoku Cloud (try with error handling)
            try:
                ichimoku = IchimokuIndicator(
                    high=df['High'], low=df['Low'], window1=9, window2=26, window3=52, fillna=True
                )
                df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
                df['Ichimoku_Conv'] = ichimoku.ichimoku_conversion_line()
                df['Ichimoku_A'] = ichimoku.ichimoku_a()
                df['Ichimoku_B'] = ichimoku.ichimoku_b()
            except Exception as ichimoku_error:
                logger.warning(f"Could not compute Ichimoku indicator: {str(ichimoku_error)}")
            
            # Create crossover signals
            df['SMA_10_20_Cross'] = np.where(df['SMA_10'] > df['SMA_20'], 1, 0)
            df['EMA_10_20_Cross'] = np.where(df['EMA_10'] > df['EMA_20'], 1, 0)
            df['Golden_Cross'] = np.where(df['SMA_50'] > df['SMA_200'], 1, 0)
            df['Death_Cross'] = np.where(df['SMA_50'] < df['SMA_200'], 1, 0)
            df['MACD_Signal_Cross'] = np.where(df['MACD'] > df['MACD_Signal'], 1, 0)
            
            # Price relative to moving averages
            df['Close_vs_SMA_50'] = (df['Close'] / df['SMA_50'] - 1) * 100
            df['Close_vs_SMA_200'] = (df['Close'] / df['SMA_200'] - 1) * 100
            
            logger.info("Computed trend indicators successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error computing trend indicators: {str(e)}")
            raise
            
    def compute_volatility_indicators(self, df):
        """Calculate volatility indicators.
        
        Args:
            df (pandas.DataFrame): Stock data
            
        Returns:
            pandas.DataFrame: Stock data with volatility indicators
        """
        try:
            # Make a copy to avoid modifying the original dataframe
            df = df.copy()
            
            # Bollinger Bands
            for window in [20, 50]:
                bollinger = BollingerBands(
                    close=df['Close'], window=window, window_dev=2, fillna=True
                )
                df[f'BB_High_{window}'] = bollinger.bollinger_hband()
                df[f'BB_Low_{window}'] = bollinger.bollinger_lband()
                df[f'BB_Mid_{window}'] = bollinger.bollinger_mavg()
                df[f'BB_Width_{window}'] = (df[f'BB_High_{window}'] - df[f'BB_Low_{window}']) / df[f'BB_Mid_{window}']
                df[f'BB_Pct_{window}'] = (df['Close'] - df[f'BB_Low_{window}']) / (df[f'BB_High_{window}'] - df[f'BB_Low_{window}'])
            
            # Average True Range (ATR)
            for window in [14, 20]:
                df[f'ATR_{window}'] = AverageTrueRange(
                    high=df['High'], low=df['Low'], close=df['Close'], window=window, fillna=True
                ).average_true_range()
            
            # Historical Volatility
            for window in [10, 20, 30]:
                df[f'Volatility_{window}'] = df['Close'].pct_change().rolling(window=window).std() * np.sqrt(252)
            
            # Normalized Volatility
            df['Normalized_Vol_10'] = df['Volatility_10'] / df['Volatility_10'].rolling(window=100).mean()
            df['Normalized_Vol_20'] = df['Volatility_20'] / df['Volatility_20'].rolling(window=100).mean()
            
            # True Range
            df['True_Range'] = np.maximum(
                df['High'] - df['Low'],
                np.maximum(
                    abs(df['High'] - df['Close'].shift(1)),
                    abs(df['Low'] - df['Close'].shift(1))
                )
            )
            
            # Bollinger Band signals
            df['BB_Upper_Touch'] = np.where(df['Close'] > df['BB_High_20'], 1, 0)
            df['BB_Lower_Touch'] = np.where(df['Close'] < df['BB_Low_20'], 1, 0)
            
            logger.info("Computed volatility indicators successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error computing volatility indicators: {str(e)}")
            raise
            
    def compute_volume_indicators(self, df):
        """Calculate volume-based indicators.
        
        Args:
            df (pandas.DataFrame): Stock data
            
        Returns:
            pandas.DataFrame: Stock data with volume indicators
        """
        try:
            # Make a copy to avoid modifying the original dataframe
            df = df.copy()
            
            # On-Balance Volume (OBV)
            df['OBV'] = OnBalanceVolumeIndicator(
                close=df['Close'], volume=df['Volume'], fillna=True
            ).on_balance_volume()
            
            # Accumulation/Distribution Index
            df['ADI'] = AccDistIndexIndicator(
                high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], fillna=True
            ).acc_dist_index()
            
            # Chaikin Money Flow
            for window in [10, 20]:
                df[f'CMF_{window}'] = ChaikinMoneyFlowIndicator(
                    high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], 
                    window=window, fillna=True
                ).chaikin_money_flow()
            
            # Volume Moving Averages
            for window in [10, 20, 50]:
                df[f'Volume_MA_{window}'] = df['Volume'].rolling(window=window).mean()
            
            # Volume Ratios
            df['Volume_Ratio_10'] = df['Volume'] / df['Volume_MA_10']
            df['Volume_Ratio_20'] = df['Volume'] / df['Volume_MA_20']
            
            # Up/Down Volume
            df['Up_Volume'] = np.where(df['Close'] > df['Close'].shift(1), df['Volume'], 0)
            df['Down_Volume'] = np.where(df['Close'] < df['Close'].shift(1), df['Volume'], 0)
            
            # Price-Volume Trend
            df['PVT'] = (df['Close'].pct_change() * df['Volume']).cumsum()
            
            # Volume signals
            df['High_Volume'] = np.where(df['Volume_Ratio_10'] > 1.5, 1, 0)
            df['Low_Volume'] = np.where(df['Volume_Ratio_10'] < 0.5, 1, 0)
            
            logger.info("Computed volume indicators successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error computing volume indicators: {str(e)}")
            raise
            
    def create_additional_features(self, df):
        """Create additional derived features.
        
        Args:
            df (pandas.DataFrame): Stock data with technical indicators
            
        Returns:
            pandas.DataFrame: Stock data with additional features
        """
        try:
            # Make a copy to avoid modifying the original dataframe
            df = df.copy()
            
            # Create trend strength indicators
            df['Trend_Strength'] = abs(df['ADX'])
            df['Strong_Trend'] = np.where(df['ADX'] > 25, 1, 0)
            
            # Create price patterns
            df['Higher_High'] = np.where(
                (df['High'] > df['High'].shift(1)) & (df['High'].shift(1) > df['High'].shift(2)), 1, 0
            )
            df['Lower_Low'] = np.where(
                (df['Low'] < df['Low'].shift(1)) & (df['Low'].shift(1) < df['Low'].shift(2)), 1, 0
            )
            
            # Create moving average convergence/divergence signals
            df['MA_Convergence'] = abs(df['SMA_20'] - df['EMA_20']) / df['Close'] * 100
            
            # Combine indicators for consolidated signals
            df['Bull_Signal'] = np.where(
                (df['RSI_14'] > 50) & 
                (df['MACD'] > df['MACD_Signal']) & 
                (df['Close'] > df['SMA_50']), 
                1, 0
            )
            df['Bear_Signal'] = np.where(
                (df['RSI_14'] < 50) & 
                (df['MACD'] < df['MACD_Signal']) & 
                (df['Close'] < df['SMA_50']), 
                1, 0
            )
            
            # Handle outliers
            for col in df.select_dtypes(include=[np.number]).columns:
                if col not in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    df[col] = np.clip(
                        df[col], 
                        df[col].quantile(0.01), 
                        df[col].quantile(0.99)
                    )
            
            logger.info("Created additional features successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error creating additional features: {str(e)}")
            raise
    
    def feature_engineering(self, df):
        """Apply all feature engineering steps to the stock data.
        
        Args:
            df (pandas.DataFrame): Raw stock data
            
        Returns:
            pandas.DataFrame: Stock data with engineered features
        """
        try:
            logger.info("Starting feature engineering process...")
            
            # Apply each step sequentially
            df = self.preprocess_stock_data(df)
            df = self.compute_momentum_indicators(df)
            df = self.compute_trend_indicators(df)
            df = self.compute_volatility_indicators(df)
            df = self.compute_volume_indicators(df)
            df = self.compute_price_change(df)
            
            # Check if 'News_Sentiment' column exists before performing sentiment analysis
            if 'News_Sentiment' in df.columns:
                df = self.compute_sentiment_score(df)
                logger.info("Applied sentiment analysis")
            else:
                logger.warning("Column 'News_Sentiment' not found. Skipping sentiment analysis.")
                
            # Create additional features
            df = self.create_additional_features(df)
            
            # Drop rows with NaN values created during feature engineering
            initial_rows = len(df)
            df.dropna(inplace=True)
            final_rows = len(df)
            
            logger.info(f"Dropped {initial_rows - final_rows} rows with NaN values")
            logger.info(f"Feature engineering completed with {len(df.columns)} features")
            logger.info(f"DataFrame shape: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in feature engineering process: {str(e)}")
            raise

def main():
    """Main function to run the feature engineering process."""
    try:
        # Initialize the feature engineering class
        feature_eng = FeatureEngineering()
        
        # Define input and output paths
        input_path = "data/raw/combined_stock_data.csv"
        output_dir = "data/processed"
        output_path = f"{output_dir}/combined_features.csv"
        
        # Create the output directory if it doesn't exist
        feature_eng.ensure_directory_exists(output_dir)
        
        # Load the raw stock data
        logger.info(f"Loading data from {input_path}")
        try:
            df = pd.read_csv(input_path)
            logger.info(f"Loaded data with shape: {df.shape}")
            logger.info(f"Columns in loaded DataFrame: {df.columns.tolist()}")
        except FileNotFoundError:
            logger.error(f"File not found: {input_path}")
            logger.info("Using fallback file path...")
            # Fallback to the hardcoded path from the original code
            df = pd.read_csv(r"C:\Users\acer\OneDrive\Documents\College\Project\Stock Price Direction Predictor with Sentiment Analysis\data\raw\combined_AAPL_MSFT_and_3_more_data.csv")
            logger.info(f"Loaded data with shape: {df.shape}")
            
        # Apply feature engineering
        df = feature_eng.feature_engineering(df)
        
        # Save the processed data
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()