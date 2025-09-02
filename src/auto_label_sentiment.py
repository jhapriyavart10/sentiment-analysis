"""
auto_label_sentiment.py
- Automatically label comments using VADER sentiment analysis
"""
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Ensure VADER is downloaded
nltk.download('vader_lexicon')

def vader_label(text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)
    compound = score['compound']
    if compound >= 0.3:
        return 'Positive'
    elif compound <= -0.3:
        return 'Negative'
    else:
        return 'Neutral'

if __name__ == "__main__":
    df = pd.read_csv('data/comments_clean.csv')
    # Use clean_comment if available, else comment
    text_col = 'clean_comment' if 'clean_comment' in df.columns else 'comment'
    sia = SentimentIntensityAnalyzer()
    df['sentiment'] = df[text_col].astype(str).apply(lambda x: vader_label(x))
    df.to_csv('data/comments_clean.csv', index=False)
    print(df[['comment', 'clean_comment', 'sentiment']].head())
    print('Sentiment labeling complete.')
