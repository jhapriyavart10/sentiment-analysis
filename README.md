# YouTube Comment Sentiment Analysis & Insights

## Overview
This project helps YouTube creators instantly analyze audience sentiment and expectations from video comments. It fetches comments using the YouTube Data API, cleans and labels them, performs sentiment analysis, summarizes feedback, extracts topics, and visualizes results in a Streamlit dashboard.

## Features
- Fetch comments from any YouTube video
- Clean and preprocess text
- Auto-label sentiment using VADER (no manual labeling needed)
- Train and evaluate ML models (Logistic Regression, Naive Bayes, Random Forest)
- Summarize audience feedback and extract top topics
- Interactive dashboard for instant insights

## Technologies
- Python
- pandas, numpy, scikit-learn, NLTK (VADER), transformers, gensim, BERTopic, matplotlib, seaborn, wordcloud, streamlit
- YouTube Data API v3

## Quick Start
1. **Clone the repository**
2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```
3. **Create a `.env` file in the project root** and add your API key:
   ```
   YOUTUBE_API_KEY=your_api_key_here
   ```
4. **Fetch comments from a YouTube video**:
   - Edit `src/data_collection.py` to set your video ID (e.g., `VIDEO_ID = 'dQw4w9WgXcQ'`)
   - Run:
     ```
     python src/data_collection.py
     ```
5. **Preprocess and label comments**:
   ```
   python src/preprocessing.py
   python src/auto_label_sentiment.py
   ```
6. **Train models and evaluate**:
   ```
   python src/modeling.py
   python src/evaluation.py
   ```
7. **Launch the dashboard**:
   ```
   streamlit run app.py
   ```

## File Structure
```
app.py
README.md
requirements.txt
.env
data/
    comments.csv
    comments_clean.csv
models/
notebooks/
src/
    data_collection.py
    preprocessing.py
    auto_label_sentiment.py
    modeling.py
    evaluation.py
    summarization.py
    topic_modeling.py
    utils.py
```

## .env Usage
- The API key is loaded from `.env` using `python-dotenv`.
- Never commit your `.env` file to GitHub.

## Cleaning Up for GitHub
- **Remove these before pushing:**
  - All files in `data/` except sample CSVs
  - All files in `models/`
  - Your `.env` file
- Only keep code, sample data, and documentation.

## Beginner Tips
- Each script is standalone and can be run step-by-step.
- Comments in code explain each step.
- You can use any YouTube video by changing the video ID in `src/data_collection.py`.

## License
MIT
# Sentiment Analysis of YouTube Comments

## Objective
Analyze comments from a YouTube video and classify them as Positive, Negative, or Neutral. Show results with metrics and simple visualizations. Deploy a Streamlit app for user input and sentiment prediction.

## Project Structure
```
Sentiment-Analysis/
│
├── data/
│   └── comments.csv
├── notebooks/
│   └── 01_data_collection.ipynb
│   └── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
│   └── 04_evaluation_visualization.ipynb
├── src/
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── utils.py
├── app.py
├── requirements.txt
└── README.md
```

## Steps

### 1. Data Collection
- Fetch comments using YouTube Data API (or use sample CSV from Kaggle).
- Save as `data/comments.csv`.

### 2. Data Preprocessing
- Clean text: remove emojis, URLs, punctuation, stopwords.
- Lowercase, tokenize, lemmatize (NLTK/spaCy).

### 3. Feature Engineering
- Convert text to features using TF-IDF Vectorizer.

### 4. Model Training
- Train Logistic Regression, Naive Bayes, Random Forest.
- Compare with DistilBERT (Hugging Face Transformers).

### 5. Evaluation
- Metrics: accuracy, precision, recall, F1-score.
- Confusion matrix, classification report.

### 6. Visualization
- Word clouds for positive/negative comments.
- Bar plots for sentiment distribution.

### 7. Deployment
- Streamlit app (`app.py`):
  - Input: text box for user comment.
  - Output: predicted sentiment + confidence score.
  - (Optional) Batch classify comments from CSV.

## How to Run
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run notebooks in order for EDA, preprocessing, modeling, and evaluation.
3. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Dataset
- Use your own YouTube API key or a sample CSV (see `data/comments.csv`).

## Technologies
- Python, pandas, numpy, scikit-learn, matplotlib, seaborn, wordcloud, NLTK/spaCy, Hugging Face Transformers, Streamlit

---

For detailed code and explanations, see the notebooks and `src/` modules.
