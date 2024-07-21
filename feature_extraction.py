from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np

# Calculate Term Frequency (TF) features
def tf_features(preprocessed_documents):
    # Convert the tokenized documents back into strings
    documents_str = [" ".join(doc) for doc in preprocessed_documents]
    vectorizer = CountVectorizer()
    tf_matrix = vectorizer.fit_transform(documents_str)
    return tf_matrix

# Calculate Term Frequency-Inverse Document Frequency (TF*IDF) features
def tfidf_features(preprocessed_documents):
    # Convert the tokenized documents back into strings
    documents_str = [" ".join(doc) for doc in preprocessed_documents]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents_str)
    return tfidf_matrix

# Calculate Word2Vec features
def word2vec_features(preprocessed_documents, embedding_size=100):
    # Train Word2Vec model
    model = Word2Vec(sentences=preprocessed_documents, vector_size=embedding_size, min_count=1, window=5)
    # Calculate document embeddings by averaging word embeddings
    document_embeddings = []
    for doc in preprocessed_documents:
        embedding_sum = np.zeros(embedding_size)
        for token in doc:
            embedding_sum += model.wv[token]
        document_embeddings.append(embedding_sum / len(doc))

    return np.array(document_embeddings)