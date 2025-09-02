import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from src.utils import load_model, load_vectorizer, predict_sentiment
from src.preprocessing import clean_text
from src.summarization import textrank_summary, hf_abstractive_summary
from src.topic_modeling import lda_topics

st.title("YouTube Comment Sentiment Analysis")

# Load models and vectorizer
MODEL_NAME = st.selectbox("Choose Model", ["LogisticRegression", "NaiveBayes", "RandomForest"])
model = load_model(MODEL_NAME)
vectorizer = load_vectorizer()

st.write("Enter a comment to analyze its sentiment:")
user_input = st.text_area("Comment", "")

if st.button("Predict"):
    if user_input.strip():
        pred, proba = predict_sentiment(user_input, model, vectorizer, clean_text)
        st.write(f"**Predicted Sentiment:** {pred}")
        st.write(f"**Confidence:** {proba:.2f}")
    else:
        st.warning("Please enter a comment.")

st.write("---")
st.write("Or upload a CSV file of comments for batch prediction:")
file = st.file_uploader("Upload CSV", type=["csv"])
if file:
    df = pd.read_csv(file)
    if 'comment' in df.columns:
        df['predicted_sentiment'] = df['comment'].apply(lambda x: predict_sentiment(x, model, vectorizer, clean_text)[0])
        st.write(df[['comment', 'predicted_sentiment']])
        st.download_button("Download Results", df.to_csv(index=False), "results.csv")

        # Sentiment distribution pie chart
        st.subheader("Sentiment Distribution")
        sentiment_counts = df['predicted_sentiment'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#8fd9b6','#f6c85f','#f08a5d'])
        ax.axis('equal')
        st.pyplot(fig)

        # Summarization
        st.subheader("Summary of Comments")
        all_comments = ' '.join(df['comment'].astype(str).tolist())
        try:
            summary = textrank_summary(all_comments, num_sentences=3)
        except Exception:
            summary = "Summary not available."
        st.write(summary)

        # Topic modeling (LDA)
        st.subheader("Top Topics Discussed")
        try:
            topics = lda_topics(df['comment'].astype(str).tolist(), n_topics=3)
            for t in topics:
                st.write(t)
        except Exception:
            st.write("Topic modeling not available.")
    else:
        st.error("CSV must have a 'comment' column.")
