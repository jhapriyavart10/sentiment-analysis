"""
topic_modeling.py
- LDA and BERTopic for topic modeling on YouTube comments
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from bertopic import BERTopic

def lda_topics(texts, n_topics=5):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform(texts)
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)
    words = vectorizer.get_feature_names_out()
    topics = []
    for idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[-10:][::-1]]
        topics.append(f"Topic {idx+1}: {' '.join(top_words)}")
    return topics

def bertopic_topics(texts, n_topics=5):
    topic_model = BERTopic(nr_topics=n_topics)
    topics, _ = topic_model.fit_transform(texts)
    return topic_model.get_topic_info()
