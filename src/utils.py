"""
utils.py
- Helper functions for loading models, vectorizers, and making predictions
"""
import joblib
import os


def load_model(model_name, model_dir=None):
    # Always load from the project root 'models' directory
    if model_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, 'models')
    path = os.path.join(model_dir, f'{model_name}.joblib')
    return joblib.load(path)


def load_vectorizer(vectorizer_path=None):
    # Always load from the project root 'models' directory
    if vectorizer_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        vectorizer_path = os.path.join(base_dir, 'models', 'tfidf_vectorizer.joblib')
    return joblib.load(vectorizer_path)

def predict_sentiment(text, model, vectorizer, preprocess_func):
    clean = preprocess_func(text)
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec).max()
    return pred, proba
