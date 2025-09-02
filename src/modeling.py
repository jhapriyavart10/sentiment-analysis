"""
modeling.py
- TF-IDF vectorization
- Train/test split
- Train Logistic Regression, Naive Bayes, Random Forest
- (Optional) DistilBERT fine-tuning stub
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def load_data(path):
    df = pd.read_csv(path)
    return df['clean_comment'], df['sentiment']

def vectorize_text(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer

def train_models(X_train_vec, y_train):
    models = {
        'LogisticRegression': LogisticRegression(max_iter=200),
        'NaiveBayes': MultinomialNB(),
        'RandomForest': RandomForestClassifier(n_estimators=100)
    }
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        joblib.dump(model, f'../models/{name}.joblib')
    return models

def evaluate_model(model, X_test_vec, y_test):
    y_pred = model.predict(X_test_vec)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    # Ensure models directory exists
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Load and clean data
    df = pd.read_csv('data/comments_clean.csv')
    # Drop rows with missing clean_comment or sentiment
    df = df.dropna(subset=['clean_comment', 'sentiment'])
    X, y = df['clean_comment'], df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)
    joblib.dump(vectorizer, os.path.join(models_dir, 'tfidf_vectorizer.joblib'))
    # Update train_models to use absolute path
    def train_models_fixed(X_train_vec, y_train):
        models = {
            'LogisticRegression': LogisticRegression(max_iter=200),
            'NaiveBayes': MultinomialNB(),
            'RandomForest': RandomForestClassifier(n_estimators=100)
        }
        for name, model in models.items():
            model.fit(X_train_vec, y_train)
            joblib.dump(model, os.path.join(models_dir, f'{name}.joblib'))
        return models
    models = train_models_fixed(X_train_vec, y_train)
    for name, model in models.items():
        print(f'\n{name} Results:')
        evaluate_model(model, X_test_vec, y_test)
    # DistilBERT stub: see notebook for implementation
