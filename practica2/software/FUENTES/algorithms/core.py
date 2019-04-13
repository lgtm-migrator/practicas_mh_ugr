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


class AlgorithmBase():
    """
    Docstring: Wrapper class for Local Search algorithm that provided
    sklearn-based syntax.
    """

    def __init__(self, threshold=0.2, seed=77766814):
        self.threshold = threshold
        self.feature_importances = []
        self.reduction = 0
        self.seed = seed

    def fit(self, X, y):
        self.reduction = np.sum(self.feature_importances < self.threshold)
        self.reduction /= len(self.feature_importances)

    def transform(self, X):
        return (X * self.feature_importances
                )[:, self.feature_importances > self.threshold]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


def evaluate(weights, X, y):
    """Evaluate a solution transforming the input data
    and calculatig the accuracy.

    Returns:
        the fitness value for the specified weights based on
        the input and labels data.
    """
    X_transformed = (X * weights)[:, weights > 0.2]
    kdtree = KDTree(X_transformed)
    neighbours = kdtree.query(X_transformed, k=2)[1][:, 1]
    accuracy = np.mean(y[neighbours] == y)
    reduction = np.mean(weights < 0.2)
    return (accuracy + reduction) / 2
