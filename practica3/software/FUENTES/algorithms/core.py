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
    Wrapper class for all algorithms.
    Subclasses should implement **fit** method.
    """

    def __init__(self, threshold=0.2, seed=77766814):
        self.threshold = threshold
        self.feature_importances = []
        self.reduction = 0
        self.seed = seed

    def set_feature_importances(self, feature_importances):
        """
        Set feature importances attribute member. And computes
        the reduction metric.
        """
        self.feature_importances = feature_importances
        self.reduction = np.sum(self.feature_importances < self.threshold)
        self.reduction /= len(self.feature_importances)

    def transform(self, X):
        """
        Transform data according to feature weights.
        It means, multiply each column by its weight, and eliminate
        columns with weight < 0.2
        """
        return (X * self.feature_importances
                )[:, self.feature_importances > self.threshold]

    def fit_transform(self, X, y):
        """
        Performs fit, then transform.
        """
        self.fit(X, y)
        return self.transform(X)


def evaluate(weights, X, y):
    """Evaluate a solution transforming the input data
    and calculatig the accuracy with leave-one-out validation.

    :param weights: Solution to evaluate
    :param X: Input data
    :param y: Label data

    Returns the fitness value for the specified weights based on
    the input and labels data.
    """
    X_transformed = (X * weights)[:, weights > 0.2]
    if X_transformed.shape[1] == 0:
        return 0
    kdtree = KDTree(X_transformed)
    neighbours = kdtree.query(X_transformed, k=2)[1][:, 1]
    accuracy = np.mean(y[neighbours] == y)
    reduction = np.mean(weights < 0.2)
    return (accuracy + reduction) / 2
