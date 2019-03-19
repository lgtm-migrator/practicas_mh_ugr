import numpy as np

"""
In order to use the module pykdtree the system must have
OpenMP support. If there is any problem during the installation
process, we can use instead the KDTree implementation from
scipy.spatial module.
"""
try:
    from pykdtree.kdtree import KDTree
except ImportError:
    from scipy.spatial import cKDTree as KDTree


def _relief(X, Y):
    W = np.zeros(X.shape[1])
    for i in range(X.shape[0]):
        x, y = X[i, :], Y[i]
        X_same_class = X[Y == y]
        X_other_class = X[Y != y]
        # Calculate the second nearest neighbor with the same class.
        ally = KDTree(X_same_class).query(x.reshape(1, -1), k=2)[1][0][1]
        # Calculate the nearest neighbor with a different class.
        enemy = KDTree(X_other_class).query(x.reshape(1, -1), k=1)[1][0]
        ally = X_same_class[ally]
        enemy = X_other_class[enemy]
        # Update the weights using the L1 distance.
        W += np.abs(x - enemy) - np.abs(x - ally)
    W[W < 0] = 0
    W /= np.max(W)
    return W


class Relief():
    """
    Docstring: Wrapper class for Relief algorithm that provided
    sklearn-based syntax.
    """

    def __init__(self, threshold=0.2):
        self.feature_importances = []
        self.threshold = threshold
        self.reduction = 0

    def fit(self, X, Y):
        self.feature_importances = _relief(X, Y)

    def transform(self, X):
        if not np.any(self.feature_importances):
            return X
        return X * self.feature_importances

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
