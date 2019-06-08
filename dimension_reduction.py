from sklearn.decomposition import PCA
from sklearn.manifold import MDS


def calculate_mds_vectors(distances):
    mds_model = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
    positions = mds_model.fit_transform(distances)
    x = positions[:, 0]
    y = positions[:, 1]
    return x, y


def calculate_principal_components(X, n_components=None):
    pca_model = PCA(n_components=n_components)
    principal_components = pca_model.fit_transform(X)

    return principal_components, pca_model