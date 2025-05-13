import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze_sentiment(text):
    """Return compound sentiment score using VADER."""
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    return score['compound']

def apply_sentiment_analysis(news_df):
    """Apply sentiment analysis to a dataframe of news headlines."""
    news_df['sentiment_score'] = news_df['headline'].apply(analyze_sentiment)
    return news_df

if __name__ == "__main__":
    # Simulated news data
    news_data = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
        'headline': [
            "Apple stock rises after strong earnings report",
            "Market plunges amid inflation concerns",
            "Tech stocks see mixed results",
            "New iPhone launch boosts Apple sales",
            "Apple faces supply chain disruptions in Asia"
        ]
    })

    scored_news = apply_sentiment_analysis(news_data)
    print(scored_news)
    scored_news.to_csv("data/raw/news_sentiment.csv", index=False)
