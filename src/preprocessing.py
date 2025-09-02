"""
preprocessing.py
- Clean text: remove emojis, URLs, punctuation, stopwords
- Lowercase, tokenize, lemmatize (NLTK)
"""
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize

# Ensure all required NLTK data is downloaded
for pkg in ['stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{pkg}')
    except LookupError:
        print(f"Downloading NLTK corpus: {pkg}")
        nltk.download(pkg)

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Remove mentions
    text = re.sub(r'#[A-Za-z0-9_]+', '', text)  # Remove hashtags
    text = text.encode('ascii', 'ignore').decode('ascii')  # Remove emojis
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = wordpunct_tokenize(text)  # Use wordpunct_tokenize instead of word_tokenize
    tokens = [w for w in tokens if w not in STOPWORDS and w.isalpha()]
    tokens = [LEMMATIZER.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

def preprocess_df(df, text_col='comment'):
    df['clean_comment'] = df[text_col].astype(str).apply(clean_text)
    return df

if __name__ == "__main__":
    df = pd.read_csv('data/comments.csv')
    print('Columns in CSV:', df.columns.tolist())
    df = preprocess_df(df)
    df.to_csv('data/comments_clean.csv', index=False)
    print(df.head())
    print('Columns in CSV:', df.columns.tolist())
    df = preprocess_df(df)
    df.to_csv('data/comments_clean.csv', index=False)
    print(df.head())
