from .core import evaluate, AlgorithmBase
import numpy as np


def local_search(X, y, max_neighbours, sigma, seed=None, init_weights=None):
    """
    Local Search Algorithm

    :param X: Train input
    :param y: Train labels
    :param max_neighbours: Max number of neighbours to explore.
    :param sigma: Standard deviation of Gaussian mutation.
    :param seed: Seed to initialize the random generator.
                 It is recommended to specify this in order to replicate
                 the experiment across executions.
    """
    n_features = X.shape[1]
    if seed:
        np.random.seed(seed)
    if np.any(init_weights):
        weights = np.copy(init_weights)
    else:
        weights = np.random.rand(n_features)
    fitness = evaluate(weights, X, y)
    trace = np.zeros(max_neighbours)
    n_generated = 0
    no_improvement = 0
    while n_generated < max_neighbours:
        trace[n_generated] = fitness
        w_prime = np.copy(weights)
        for k in np.random.permutation(n_features):
            n_generated += 1
            no_improvement += 1
            last_state = w_prime[k]
            w_prime[k] = np.clip(last_state + np.random.normal(0, sigma), 0, 1)
            f = evaluate(w_prime, X, y)
            if fitness < f:
                weights = w_prime
                fitness = f
                no_improvement = 0
                break
            else:
                w_prime[k] = last_state
            if n_generated > max_neighbours or no_improvement >= (20 * n_features):
                return weights, trace[trace > 0], n_generated
    trace[-1] = fitness
    return weights, trace[trace > 0], n_generated


class LocalSearch(AlgorithmBase):
    """
    Wrapper class for Local Search algorithm that provided
    sklearn-based syntax.
    """

    def __init__(self, threshold=0.2, max_neighbours=15000, sigma=0.3, seed=1):
        self.max_neighbours = max_neighbours
        self.sigma = sigma
        self.trace = []
        self.neighbors_generated = 0
        super().__init__(threshold, seed)

    def fit(self, X, y):
        """
        Fit the a 1-NN model using Local search for feature weighting.

        :param X: Train inputs
        :param y: Train labels
        """
        result = local_search(X, y, self.max_neighbours, self.sigma, self.seed)
        self.trace = result[1]
        self.neighbors_generated = result[2]
        super().set_feature_importances(result[0])
