import numpy as np
from pykdtree.kdtree import KDTree


def _relief(X, Y):
    W = np.zeros(X.shape[1])
    for i in range(X.shape[0]):
        x, y = X[i, :], Y[i]
        X_same_class = X[Y == y]
        X_other_class = X[Y != y]
        ally = KDTree(X_same_class).query(x.reshape(1, -1), k=2)[1][0][1]
        enemy = KDTree(X_other_class).query(x.reshape(1, -1), k=1)[1][0]
        ally = X_same_class[ally]
        enemy = X_other_class[enemy]
        W += np.abs(x - enemy) - np.abs(x - ally)
    W[W < 0] = 0
    W /= np.max(W)
    return W


class Relief():
    """
    Docstring: Clase que envuelve el algoritmo Relief
    para poder usarlo con una sintaxis similar a
    Sklearn.
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
