"""
summarization.py
- Extractive and abstractive summarization for YouTube comments
"""
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import pipeline

# Extractive summarization using TextRank

def textrank_summary(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

# Abstractive summarization using Hugging Face transformers

def hf_abstractive_summary(text, model_name="facebook/bart-large-cnn", min_length=30, max_length=120):
    summarizer = pipeline("summarization", model=model_name)
    summary = summarizer(text, min_length=min_length, max_length=max_length, do_sample=False)
    return summary[0]["summary_text"]
