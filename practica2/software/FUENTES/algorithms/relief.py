import numpy as np
from .core import KDTree, AlgorithmBase


def _relief(X, Y):
    """
    Relief algorithm.

    :param X: Train input data
    :param Y: Train label data
    """
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


class Relief(AlgorithmBase):
    """
    Wrapper class for Relief algorithm that provided
    sklearn-based syntax.
    """
    def fit(self, X, y):
        super().set_feature_importances(_relief(X, y))
