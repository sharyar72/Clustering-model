from collections import Counter
from sklearn.metrics import confusion_matrix
import numpy as np

# Function which takes true and predicted lablels to calculate purity as mentioned 
def purity_score(labels_true, labels_pred):
    cm = confusion_matrix(labels_true, labels_pred)
    return np.sum(np.amax(cm, axis=0)) / np.sum(cm)


# function which evaluates which calls purity scores and evaluates for all 3 features on which clustering is done
def evaluate_clustering(labels_true, tf_clusters, tfidf_clusters, word2vec_clusters):
    tf_purity = purity_score(labels_true, tf_clusters)
    tfidf_purity = purity_score(labels_true, tfidf_clusters)
    word2vec_purity = purity_score(labels_true, word2vec_clusters)

    print("Purity Scores:")
    print(f"TF Clustering: {tf_purity}")
    print(f"TF*IDF Clustering: {tfidf_purity}")
    print(f"Word2Vec Clustering: {word2vec_purity}")
