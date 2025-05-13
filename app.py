import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import re
from io import StringIO
import base64

# Page configuration
st.set_page_config(
    page_title="Stock Prediction App",
    page_icon="📈",
    layout="wide"
)

# Apply custom CSS for better styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
    }
    .stProgress .st-bo {
        background-color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

# Define functions for data loading
@st.cache_data(ttl=3600)
def load_stock_data(ticker, start_date, end_date):
    """Load stock data with caching for better performance"""
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return None

@st.cache_data(ttl=86400)
def fetch_news_data(ticker, days=30):
    """Fetch recent news headlines for a given ticker"""
    try:
        # This is a placeholder. In a real app, you would connect to a news API
        # For demonstration, we'll create some simulated news data
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days)
        
        # Generate some sample news headlines based on the ticker
        news_templates = [
            "{} reports strong quarterly earnings",
            "{} stock rises after analyst upgrade",
            "Investors remain cautious about {} future growth",
            "{} announces new product line",
            "Market analysts predict bright future for {}",
            "{} faces regulatory scrutiny",
            "Competitors putting pressure on {} market share",
            "{} expands into new markets",
            "Economic downturn affects {} sales",
            "{} CEO gives optimistic outlook"
        ]
        
        # Generate random dates within the range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = np.random.choice(date_range, min(len(date_range), 20), replace=False)
        dates.sort()
        
        # Generate headlines
        headlines = [np.random.choice(news_templates).format(ticker) for _ in range(len(dates))]
        
        news_df = pd.DataFrame({
            'date': dates,
            'headline': headlines
        })
        
        return news_df
    except Exception as e:
        st.error(f"Error fetching news for {ticker}: {e}")
        return pd.DataFrame(columns=['date', 'headline'])

def analyze_sentiment(text):
    """Return compound sentiment score using VADER"""
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    return score['compound']

def apply_sentiment_analysis(news_df):
    """Apply sentiment analysis to a dataframe of news headlines"""
    if not news_df.empty:
        news_df['sentiment_score'] = news_df['headline'].apply(analyze_sentiment)
    return news_df

def compute_technical_indicators(df):
    """Enhanced version of technical indicator calculation"""
    # First check if we're dealing with multi-index columns (which can happen with yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        # If multi-index, flatten the columns
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    
    # Ensure we're working with Series and not DataFrames
    # If df['Close'] returns a DataFrame, we need to squeeze it to a Series
    close_series = df['Close'].squeeze() if hasattr(df['Close'], 'squeeze') else df['Close']
    high_series = df['High'].squeeze() if hasattr(df['High'], 'squeeze') else df['High']
    low_series = df['Low'].squeeze() if hasattr(df['Low'], 'squeeze') else df['Low']
    
    # Simple Moving Averages
    df['SMA_10'] = SMAIndicator(close=close_series, window=10).sma_indicator()
    df['SMA_20'] = SMAIndicator(close=close_series, window=20).sma_indicator()
    df['SMA_50'] = SMAIndicator(close=close_series, window=50).sma_indicator()
    
    # Exponential Moving Averages
    df['EMA_10'] = EMAIndicator(close=close_series, window=10).ema_indicator()
    df['EMA_20'] = EMAIndicator(close=close_series, window=20).ema_indicator()
    
    # RSI (Relative Strength Index)
    df['RSI'] = RSIIndicator(close=close_series, window=14).rsi()
    
    # MACD (Moving Average Convergence Divergence)
    macd = MACD(close=close_series)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = BollingerBands(close=close_series, window=20, window_dev=2)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    df['BB_Mid'] = bollinger.bollinger_mavg()
    df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid']
    
    # Stochastic Oscillator
    stoch = StochasticOscillator(high=high_series, low=low_series, close=close_series)
    df['Stoch_K'] = stoch.stoch()
    df['Stoch_D'] = stoch.stoch_signal()
    
    # Price changes and volatility
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_2d'] = df['Close'].pct_change(periods=2)
    df['Price_Change_5d'] = df['Close'].pct_change(periods=5)
    
    # Volatility measures
    df['Volatility_10'] = df['Close'].rolling(window=10).std()
    df['Volatility_20'] = df['Close'].rolling(window=20).std()
    
    # Trading volume features
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Volume_MA_10'] = df['Volume'].rolling(window=10).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_10']
    
    # Price position within recent range
    df['Price_Range_10'] = (df['Close'] - df['Low'].rolling(10).min()) / (df['High'].rolling(10).max() - df['Low'].rolling(10).min())
    
    return df

def create_target_labels(df, prediction_days=1):
    """Create prediction targets for future price movements"""
    # Binary classification label: 1 if price goes up in the next n days, else 0
    df[f'Target_{prediction_days}d'] = (df['Close'].shift(-prediction_days) > df['Close']).astype(int)
    
    # Percentage change for the next n days (for regression models)
    df[f'Target_Return_{prediction_days}d'] = df['Close'].pct_change(periods=-prediction_days)
    
    return df

def merge_price_and_sentiment(stock_df, sentiment_df):
    """Merge stock price data with news sentiment"""
    if sentiment_df.empty:
        stock_df['sentiment_score'] = 0
        stock_df['sentiment_ma5'] = 0
        return stock_df
    
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    
    # Aggregate sentiment by date
    daily_sentiment = sentiment_df.groupby('date')['sentiment_score'].mean().reset_index()
    
    # Merge with stock data
    merged = pd.merge(stock_df, daily_sentiment, left_on='Date', right_on='date', how='left')
    merged.drop(columns=['date'], inplace=True)
    
    # Fill missing sentiment values and add a moving average of sentiment
    merged['sentiment_score'].fillna(method='ffill', inplace=True)
    merged['sentiment_score'].fillna(0, inplace=True)
    merged['sentiment_ma5'] = merged['sentiment_score'].rolling(window=5).mean().fillna(0)
    
    return merged

def prepare_model_data(df, target_days=1, train_size=0.8):
    """Prepare data for model training with feature scaling"""
    # Drop rows with NaN values
    df = df.dropna()
    
    # Select features for training
    feature_columns = [
        'SMA_10', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_20',
        'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Width', 'Stoch_K', 'Stoch_D',
        'Price_Change', 'Price_Change_2d', 'Price_Change_5d',
        'Volatility_10', 'Volatility_20',
        'Volume_Change', 'Volume_Ratio', 'Price_Range_10',
        'sentiment_score', 'sentiment_ma5'
    ]
    
    # Make sure all feature columns exist in the dataframe
    existing_features = [col for col in feature_columns if col in df.columns]
    
    X = df[existing_features]
    y = df[f'Target_{target_days}d']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=existing_features)
    
    # Train-test split (time-based)
    split_idx = int(len(X_scaled) * train_size)
    X_train, X_test = X_scaled_df.iloc[:split_idx], X_scaled_df.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler, existing_features

def train_xgboost_model(X_train, y_train, use_grid_search=False):
    """Train an XGBoost classifier with optional hyperparameter tuning"""
    if use_grid_search and len(X_train) > 500:  # Only do grid search if we have enough data
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [50, 100, 200],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(
            estimator=xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
            param_grid=param_grid,
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        return best_model, grid_search.best_params_
    else:
        # Use default parameters
        model = xgb.XGBClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        return model, None

def evaluate_model_performance(model, X_test, y_test):
    """Evaluate model performance with multiple metrics"""
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    return {
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'roc_curve': {
            'fpr': fpr,
            'tpr': tpr,
            'auc': roc_auc
        },
        'y_pred': y_pred,
        'y_proba': y_proba
    }

def plot_stock_with_indicators(df, ticker):
    """Create an interactive plot of stock price with key indicators"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.2, 0.15, 0.15],
        subplot_titles=("Price & Indicators", "Volume", "RSI", "MACD")
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # Add Moving Averages
    if 'SMA_10' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['SMA_10'], name="SMA 10", line=dict(color='blue', width=1.5)),
            row=1, col=1
        )
    
    if 'SMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['SMA_50'], name="SMA 50", line=dict(color='orange', width=1.5)),
            row=1, col=1
        )
    
    # Add Bollinger Bands
    if 'BB_High' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['BB_High'], name="BB Upper", line=dict(color='rgba(250,0,0,0.5)', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['BB_Low'], name="BB Lower", line=dict(color='rgba(250,0,0,0.5)', width=1)),
            row=1, col=1
        )
    
    # Volume chart
    fig.add_trace(
        go.Bar(x=df['Date'], y=df['Volume'], name="Volume", marker_color='rgba(0,150,255,0.5)'),
        row=2, col=1
    )
    
    # RSI
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['RSI'], name="RSI", line=dict(color='purple', width=1.5)),
            row=3, col=1
        )
        
        # Add RSI overbought/oversold lines
        fig.add_shape(
            type="line", line_color="red", line_width=1, opacity=0.5, line_dash="dash",
            x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1], y0=70, y1=70,
            xref="x3", yref="y3"
        )
        fig.add_shape(
            type="line", line_color="green", line_width=1, opacity=0.5, line_dash="dash",
            x0=df['Date'].iloc[0], x1=df['Date'].iloc[-1], y0=30, y1=30,
            xref="x3", yref="y3"
        )
    
    # MACD
    if all(col in df.columns for col in ['MACD', 'MACD_Signal']):
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['MACD'], name="MACD", line=dict(color='blue', width=1.5)),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['MACD_Signal'], name="Signal", line=dict(color='red', width=1.5)),
            row=4, col=1
        )
        
        # MACD Histogram
        colors = ['green' if val > 0 else 'red' for val in df['MACD_Hist']]
        fig.add_trace(
            go.Bar(x=df['Date'], y=df['MACD_Hist'], name="Hist", marker_color=colors),
            row=4, col=1
        )
    
    # Update layout for better visualization
    fig.update_layout(
        title=f"{ticker} Stock Analysis",
        xaxis_rangeslider_visible=False,
        height=900,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    # Y-axis titles
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    fig.update_yaxes(title_text="MACD", row=4, col=1)
    
    return fig

def plot_sentiment_analysis(df):
    """Plot sentiment analysis over time"""
    if 'sentiment_score' not in df.columns or df['sentiment_score'].isna().all():
        return None
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Plot stock price
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Close'], name="Price", line=dict(color='blue', width=2)),
        secondary_y=False
    )
    
    # Plot sentiment score
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['sentiment_score'], name="Sentiment", line=dict(color='red', width=1.5)),
        secondary_y=True
    )
    
    # Add 5-day moving average of sentiment if available
    if 'sentiment_ma5' in df.columns:
        fig.add_trace(
            go.Scatter(x=df['Date'], y=df['sentiment_ma5'], name="Sentiment MA5", 
                      line=dict(color='orange', width=2, dash='dash')),
            secondary_y=True
        )
    
    # Update layout
    fig.update_layout(
        title="Stock Price vs. News Sentiment",
        xaxis_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=80, b=50),
        height=500
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Stock Price ($)", secondary_y=False)
    fig.update_yaxes(title_text="Sentiment Score", secondary_y=True)
    
    return fig

def plot_feature_importance(model, feature_names):
    """Plot feature importance from the XGBoost model"""
    # Get feature importance
    importance = model.feature_importances_
    
    # Create a DataFrame for visualization
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('Feature Importance')
    plt.tight_layout()
    
    return fig

def plot_confusion_matrix(conf_matrix):
    """Create a heatmap of the confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    return fig

def plot_roc_curve(fpr, tpr, roc_auc):
    """Plot ROC curve and AUC"""
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    return fig

def plot_prediction_results(df, results, prediction_days):
    """Plot actual vs predicted results"""
    # Create a merged dataframe with actual values and predictions
    results_df = pd.DataFrame({
        'Date': df['Date'].iloc[-len(results['y_pred']):].values,
        'Close': df['Close'].iloc[-len(results['y_pred']):].values,
        'Actual': results['y_test'],
        'Predicted': results['y_pred'],
        'Probability': results['y_proba']
    })
    
    # Create color maps for actual and predicted values
    actual_colors = ['red' if val == 0 else 'green' for val in results_df['Actual']]
    pred_colors = ['red' if val == 0 else 'green' for val in results_df['Predicted']]
    
    # Create the plot
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                         vertical_spacing=0.1, row_heights=[0.7, 0.3],
                         subplot_titles=("Stock Price", "Prediction Probability"))
    
    # Plot the stock price
    fig.add_trace(
        go.Scatter(x=results_df['Date'], y=results_df['Close'], name="Price", line=dict(color='blue', width=2)),
        row=1, col=1
    )
    
    # Add markers for actual movement
    fig.add_trace(
        go.Scatter(
            x=results_df['Date'],
            y=results_df['Close'],
            mode='markers',
            marker=dict(size=10, color=actual_colors, symbol='circle', line=dict(width=1, color='black')),
            name="Actual Movement",
            hovertext=[f"Actual: {'Up' if val == 1 else 'Down'}" for val in results_df['Actual']]
        ),
        row=1, col=1
    )
    
    # Add markers for predicted movement (slightly offset)
    fig.add_trace(
        go.Scatter(
            x=results_df['Date'],
            y=[price * 1.01 for price in results_df['Close']],  # Offset for visibility
            mode='markers',
            marker=dict(size=10, color=pred_colors, symbol='triangle-down', line=dict(width=1, color='black')),
            name="Predicted Movement",
            hovertext=[f"Predicted: {'Up' if val == 1 else 'Down'} (Conf: {prob:.2f})" 
                      for val, prob in zip(results_df['Predicted'], results_df['Probability'])]
        ),
        row=1, col=1
    )
    
    # Add prediction probability
    fig.add_trace(
        go.Bar(
            x=results_df['Date'],
            y=results_df['Probability'],
            marker_color=['red' if prob < 0.5 else 'green' for prob in results_df['Probability']],
            name="Up Probability"
        ),
        row=2, col=1
    )
    
    # Add 0.5 threshold line
    fig.add_shape(
        type="line", line_color="black", line_width=1, opacity=0.5, line_dash="dash",
        x0=results_df['Date'].iloc[0], x1=results_df['Date'].iloc[-1], y0=0.5, y1=0.5,
        xref="x2", yref="y2"
    )
    
    # Update layout
    fig.update_layout(
        title=f"{prediction_days}-Day Stock Movement Prediction Results",
        xaxis2_title="Date",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        height=700
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text="Stock Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Probability of Rise", row=2, col=1)
    
    return fig

def make_predictions(model, df, feature_columns, scaler, latest_date, prediction_days=1):
    """Make predictions for the immediate future"""
    # Prepare the most recent data point
    latest_data = df[feature_columns].iloc[-1:].values
    
    # Scale the data
    scaled_data = scaler.transform(latest_data)
    
    # Make prediction
    prob_up = model.predict_proba(scaled_data)[0, 1]
    prediction = 1 if prob_up >= 0.5 else 0
    
    # Calculate target date
    target_date = latest_date + datetime.timedelta(days=prediction_days)
    while target_date.weekday() > 4:  # Skip weekends
        target_date += datetime.timedelta(days=1)
    
    return {
        'prediction': prediction,
        'probability': prob_up,
        'latest_price': df['Close'].iloc[-1],
        'target_date': target_date.strftime('%Y-%m-%d'),
        'prediction_days': prediction_days
    }

def get_table_download_link(df, filename="data.csv", text="Download CSV"):
    """Generate a link to download the dataframe as a CSV file"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Main Streamlit app
def main():
    st.title("📈 Stock Price Prediction App")
    st.markdown("### Predict stock price movements using technical indicators and sentiment analysis")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Stock selection
    ticker_options = {
        "Apple": "AAPL", 
        "Microsoft": "MSFT", 
        "Google": "GOOGL", 
        "Amazon": "AMZN", 
        "Tesla": "TSLA",
        "NVIDIA": "NVDA",
        "Meta": "META",
        "Netflix": "NFLX"
    }
    
    # Allow custom ticker input
    custom_ticker = st.sidebar.text_input("Enter custom ticker symbol", "")
    if custom_ticker:
        selected_ticker = custom_ticker.upper()
    else:
        selected_ticker = st.sidebar.selectbox(
            "Select stock ticker",
            options=list(ticker_options.values()),
            index=0
        )
    
    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        lookback_years = st.number_input("Years of history", min_value=1, max_value=10, value=2)
    
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365 * lookback_years)
    
    with col2:
        prediction_days = st.number_input("Prediction horizon (days)", min_value=1, max_value=30, value=1)
    
    # Model parameters
    st.sidebar.header("Model Settings")
    use_grid_search = st.sidebar.checkbox("Use grid search (slower but better)", value=False)
    train_size = st.sidebar.slider("Training data size", min_value=0.6, max_value=0.9, value=0.8, step=0.05)
    
    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        include_sentiment = st.checkbox("Include news sentiment", value=True)
        show_model_details = st.checkbox("Show detailed model metrics", value=False)

    # Data loading button
    if st.sidebar.button("Load Data & Train Model"):
        with st.spinner(f"Loading data for {selected_ticker}..."):
            # Load stock data
            df = load_stock_data(selected_ticker, start_date, end_date)
            
            if df is not None and not df.empty:
                # Process the data
                df = compute_technical_indicators(df)
                df = create_target_labels(df, prediction_days)
                
                if include_sentiment:
                    # Fetch and analyze news sentiment
                    news_df = fetch_news_data(selected_ticker, days=365)
                    news_df = apply_sentiment_analysis(news_df)
                    df = merge_price_and_sentiment(df, news_df)
                else:
                    # Add zero sentiment columns to maintain compatibility
                    df['sentiment_score'] = 0
                    df['sentiment_ma5'] = 0
                
                # Display the processed data
                st.success(f"Successfully loaded data for {selected_ticker}")
                
                # Main visualization tab
                tab1, tab2, tab3, tab4 = st.tabs(["Stock Analysis", "Model Results", "Prediction", "Raw Data"])
                
                with tab1:
                    st.subheader("Stock Price with Technical Indicators")
                    fig = plot_stock_with_indicators(df, selected_ticker)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if include_sentiment:
                        st.subheader("News Sentiment Analysis")
                        sentiment_fig = plot_sentiment_analysis(df)
                        if sentiment_fig:
                            st.plotly_chart(sentiment_fig, use_container_width=True)
                        
                        with st.expander("View News Headlines"):
                            st.dataframe(news_df[['date', 'headline', 'sentiment_score']])
                
                with tab2:
                    st.subheader("Model Training and Evaluation")
                    
                    # Prepare data for modeling
                    X_train, X_test, y_train, y_test, scaler, features = prepare_model_data(
                        df, target_days=prediction_days, train_size=train_size
                    )
                    
                    if len(X_train) < 30:
                        st.error("Not enough data for training. Please select a longer time period.")
                    else:
                        # Train the model
                        with st.spinner("Training model..."):
                            model, best_params = train_xgboost_model(X_train, y_train, use_grid_search)
                            
                            if best_params and show_model_details:
                                st.write("Best parameters found:")
                                st.json(best_params)
                            
                            # Evaluate the model
                            results = evaluate_model_performance(model, X_test, y_test)
                            
                            # Store test set results for later plotting
                            results['y_test'] = y_test
                            
                            # Display metrics
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Accuracy", f"{results['accuracy']:.3f}")
                            col2.metric("Precision (Up)", f"{results['classification_report']['1']['precision']:.3f}")
                            col3.metric("Recall (Up)", f"{results['classification_report']['1']['recall']:.3f}")
                            
                            # Create evaluation plots
                            st.subheader("Model Evaluation")
                            
                            eval_col1, eval_col2 = st.columns(2)
                            with eval_col1:
                                conf_matrix_fig = plot_confusion_matrix(results['confusion_matrix'])
                                st.pyplot(conf_matrix_fig)
                                
                            with eval_col2:
                                roc_fig = plot_roc_curve(
                                    results['roc_curve']['fpr'], 
                                    results['roc_curve']['tpr'], 
                                    results['roc_curve']['auc']
                                )
                                st.pyplot(roc_fig)
                            
                            # Feature importance plot
                            st.subheader("Feature Importance")
                            importance_fig = plot_feature_importance(model, features)
                            st.pyplot(importance_fig)
                            
                            # Test predictions visualization
                            st.subheader("Test Set Predictions")
                            prediction_fig = plot_prediction_results(df, results, prediction_days)
                            st.plotly_chart(prediction_fig, use_container_width=True)
                            
                            # Display detailed classification report if requested
                            if show_model_details:
                                with st.expander("Detailed Classification Report"):
                                    report_df = pd.DataFrame(results['classification_report']).drop('accuracy', axis=1)
                                    st.dataframe(report_df.style.format("{:.3f}"))
                
                with tab3:
                    st.subheader("Future Price Movement Prediction")
                    
                    # Make prediction for the next period
                    latest_prediction = make_predictions(
                        model, df, features, scaler, 
                        df['Date'].iloc[-1], prediction_days
                    )
                    
                    # Display prediction result
                    direction = "Up ↑" if latest_prediction['prediction'] == 1 else "Down ↓"
                    prob_pct = latest_prediction['probability'] * 100
                    
                    st.markdown(f"""
                    ### Prediction for {latest_prediction['target_date']}:
                    """)
                    
                    # Create prediction cards
                    pred_col1, pred_col2 = st.columns(2)
                    
                    with pred_col1:
                        st.metric(
                            "Predicted Direction", 
                            direction,
                            delta=f"{abs(prob_pct - 50):.1f}% {'confidence' if prob_pct >= 50 else 'doubt'}"
                        )
                    
                    with pred_col2:
                        st.metric(
                            "Current Price", 
                            f"${latest_prediction['latest_price']:.2f}"
                        )
                    
                    # Probability gauge
                    st.markdown(f"""
                    #### Probability of price going UP: {prob_pct:.1f}%
                    """)
                    
                    # Create a gauge chart for the probability
                    gauge_fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob_pct,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Confidence Level"},
                        gauge = {
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "red"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    gauge_fig.update_layout(height=300)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    # Disclaimer
                    st.info("""
                    **Disclaimer:** These predictions are based on historical patterns and technical indicators.
                    They should not be considered as financial advice. Always do your own research before making
                    investment decisions.
                    """)
                    
                    # Prediction factors
                    with st.expander("Key Factors Influencing This Prediction"):
                        # Get the top 5 most important features
                        importance = model.feature_importances_
                        sorted_idx = importance.argsort()[::-1]
                        top_features = [(features[i], importance[i]) for i in sorted_idx[:5]]
                        
                        for feature, imp in top_features:
                            st.markdown(f"- **{feature}**: {imp:.3f} importance")
                
                with tab4:
                    st.subheader("Raw Data")
                    
                    # Display data preprocessing steps
                    with st.expander("Data Preprocessing Steps"):
                        st.write("""
                        1. Download historical stock data from Yahoo Finance
                        2. Calculate technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
                        3. Create target variables for future price movements
                        4. Fetch and analyze news sentiment (if enabled)
                        5. Merge price and sentiment data
                        6. Scale features for machine learning
                        """)
                    
                    # Display raw data tables
                    with st.expander("View Processed Data"):
                        # Show a sample of the processed data
                        st.dataframe(df.tail(100))
                        
                        # Download link
                        st.markdown(get_table_download_link(df, f"{selected_ticker}_processed_data.csv"), unsafe_allow_html=True)
                
            else:
                st.error(f"Failed to load data for {selected_ticker}. Please check the ticker symbol.")

# Run the app
if __name__ == "__main__":
    main()