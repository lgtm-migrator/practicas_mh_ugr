import numpy as np
#from scipy.spatial import cKDTree as KDTree
from pykdtree.kdtree import KDTree
from numba import jit, prange


def metric(weights, accuracy, alpha=0.5, threshold=0.2):
    reduction = np.sum(weights < threshold) / len(weights)
    return alpha * accuracy + (1 - alpha) * reduction


@jit(parallel=True)
def knn_accuracy(X, Y):
    kdtree = KDTree(X)
    accuracy = 0
    for index in prange(X.shape[0]):
        neighbour = kdtree.query(X[index].reshape(1, -1), k=2)[1][0][1]
        if Y[neighbour] == Y[index] and neighbour != index:
            accuracy += 1
    return accuracy / X.shape[0]


@jit()
def evaluate(weights, X, y):
    X_transformed = (X * weights)[:, weights > 0.2]
    acc = knn_accuracy(X_transformed, y)
    return metric(weights, acc)


@jit()
def local_search(X, y, max_neighbours, sigma, seed):
    n_features = X.shape[1]
    np.random.seed(seed)
    weights = np.random.rand(n_features)
    goodness = evaluate(weights, X, y)
    trace = np.zeros(max_neighbours)
    n_generated = 0
    last_improvement = 0
    while n_generated < max_neighbours:
        trace[n_generated] = goodness
        w_prime = np.copy(weights)
        for k in np.random.permutation(n_features):
            n_generated += 1
            w_prime[k] = np.clip(w_prime[k] + np.random.randn() * sigma, 0, 1)
            g = evaluate(w_prime, X, y)
            if goodness < g:
                weights = w_prime
                goodness = g
                last_improvement = n_generated
                break
            diff = n_generated - last_improvement
            if n_generated > max_neighbours or diff > (20 * n_features):
                return weights, trace[trace > 0], n_generated
    return weights, trace[trace > 0], n_generated


class LocalSearch():
    """
    Docstring: TODO
    """

    def __init__(self, threshold=0.2, max_neighbours=15000, sigma=0.3, seed=1):
        self.threshold = threshold
        self.max_neighbours = max_neighbours
        self.sigma = sigma
        self.seed = seed
        self.feature_importances = []
        self.trace = []
        self.neighbors_generated = 0
        self.reduction = 0

    def fit(self, X, y):
        result = local_search(X, y, self.max_neighbours, self.sigma, self.seed)
        self.feature_importances = result[0]
        self.trace = result[1]
        self.neighbors_generated = result[2]
        self.reduction = np.sum(self.feature_importances < self.threshold)
        self.reduction /= len(self.feature_importances)

    def transform(self, X):
        return (X * self.feature_importances)[:, self.feature_importances > self.threshold]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

