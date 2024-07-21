import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import numpy as np

# function to fit features to K Means
def kmeans_clustering(features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features)
    return kmeans.labels_

# Function to visualise Document Cluster
def visualize_clusters(features, labels, title):
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap="viridis")
    plt.title(title)
    plt.show()

# Function to find the optimal value of K
def find_optimal_k(features, k_range, plot=True):
    inertias = []
    silhouettes = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(features, kmeans.labels_))

    if plot:
        fig, ax1 = plt.subplots()
        ax1.set_title("Elbow Method")
        ax1.set_xlabel("Number of Clusters")
        ax1.set_ylabel("Inertia")
        ax1.plot(k_range, inertias, "b.-")
        ax1.tick_params("y", colors="b")
        ax2 = ax1.twinx()
        ax2.set_ylabel("Silhouette Score")
        ax2.plot(k_range, silhouettes, "r.-")
        ax2.tick_params("y", colors="r")
        fig.tight_layout()
        plt.show()

    return k_range[np.argmax(silhouettes)]