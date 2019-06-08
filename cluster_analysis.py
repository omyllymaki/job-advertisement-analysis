import numpy as np
from scipy.cluster import hierarchy as hc
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics.pairwise import cosine_similarity


def calculate_clusters(linkage_matrix, max_distance=None, n_clusters=None):
    if max_distance:
        clusters = fcluster(linkage_matrix, max_distance, criterion='distance')
        return clusters

    if n_clusters:
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        return clusters

    n_clusters = calculate_optimal_number_of_clusters(linkage_matrix)
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    return clusters


def calculate_optimal_number_of_clusters(Z):
    distances = Z[:, 2]
    acceleration = np.diff(distances, 2)
    acceleration_reversed = acceleration[::-1]
    n_clusters = acceleration_reversed.argmax() + 2
    return n_clusters


def cluster_analysis(X, max_distance):
    similarities = cosine_similarity(X)
    distances = 1 - similarities
    linkage_matrix = hc.linkage(distances, 'ward')
    clusters = calculate_clusters(linkage_matrix, max_distance)

    cluster_data = {
        'similarities': similarities,
        'linkage_matrix': linkage_matrix,
        'clusters': clusters
    }

    return cluster_data


def calculate_key_words_of_clusters(clusters, X, vocabulary, n):
    clusters_unique = list(set(clusters))

    key_words_list = []
    for k in clusters_unique:
        indices = np.where(clusters == k)[0]
        points = X[indices, :]
        centroid = np.mean(points, axis=0)

        idx = np.argsort(centroid)
        idx_largest = idx[-n:]
        key_words = [vocabulary[i] for i in idx_largest]

        key_words_list.append(key_words)

    return key_words_list
