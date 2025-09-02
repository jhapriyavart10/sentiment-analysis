"""
evaluation.py
- Metrics: accuracy, precision, recall, F1-score
- Confusion matrix, classification report
- Visualizations: word clouds, bar plots
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import classification_report, confusion_matrix

def plot_sentiment_distribution(df, sentiment_col='sentiment'):
    plt.figure(figsize=(6,4))
    sns.countplot(x=sentiment_col, data=df, palette='Set2')
    plt.title('Sentiment Distribution')
    plt.show()

def plot_wordcloud(df, sentiment, text_col='clean_comment'):
    text = ' '.join(df[df['sentiment']==sentiment][text_col])
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud for {sentiment} Comments')
    plt.show()

def print_classification_report(y_true, y_pred):
    print(classification_report(y_true, y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))

# Example usage in notebook or script
if __name__ == "__main__":
    df = pd.read_csv('data/comments_clean.csv')
    # Automatically detect sentiment column
    sentiment_col = None
    for col in ['sentiment', 'predicted_sentiment']:
        if col in df.columns:
            sentiment_col = col
            break
    if sentiment_col is None:
        print("No sentiment column found in CSV. Available columns:", df.columns.tolist())
    else:
        plot_sentiment_distribution(df, sentiment_col=sentiment_col)
        plot_wordcloud(df, 'Positive', text_col='clean_comment')
        plot_wordcloud(df, 'Negative', text_col='clean_comment')
