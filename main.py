from preprocessing import preprocess_documents,load_documents_and_labels
from feature_extraction import tf_features,tfidf_features,word2vec_features
from kmeans_clustering import kmeans_clustering,find_optimal_k,visualize_clusters
from purity_scores import evaluate_clustering


# Replace "path_to_folder" with the actual path to the folder containing the class subfolders
folder_path = r"C:\Users\LENOVO\Desktop\IR_assignment_3\Doc50 GT"
documents, labels_true = load_documents_and_labels(folder_path)
preprocessed_documents = preprocess_documents(documents)

# Extract features and perform clustering
tf_matrix = tf_features(preprocessed_documents)
tfidf_matrix = tfidf_features(preprocessed_documents)
word2vec_embeddings = word2vec_features(preprocessed_documents)

# Perform K-Means clustering
# Find optimal K using the elbow method
k_range = range(2, 11)
optimal_k = find_optimal_k(tfidf_matrix, k_range)
print(f"Optimal K: {optimal_k}")

# visualise the clusters for each feature (baseline1, baseline2 and word2vec)
n_clusters = optimal_k
tf_clusters = kmeans_clustering(tf_matrix, n_clusters)
tfidf_clusters = kmeans_clustering(tfidf_matrix, n_clusters)
word2vec_clusters = kmeans_clustering(word2vec_embeddings, n_clusters)
visualize_clusters(tf_matrix.toarray(), tf_clusters, "TF Clusters")
visualize_clusters(tfidf_matrix.toarray(), tfidf_clusters, "TF*IDF")
visualize_clusters(word2vec_embeddings, word2vec_clusters, "Word2Vec")

# purity scores for values of k from 2 to 10
for i in range(2,11):
    n_clusters = i
    print("clusters: ",n_clusters)
    tf_clusters = kmeans_clustering(tf_matrix, n_clusters)
    tfidf_clusters = kmeans_clustering(tfidf_matrix, n_clusters)
    word2vec_clusters = kmeans_clustering(word2vec_embeddings, n_clusters)

    # Evaluate clustering results
    evaluate_clustering(labels_true, tf_clusters, tfidf_clusters, word2vec_clusters)
    print()