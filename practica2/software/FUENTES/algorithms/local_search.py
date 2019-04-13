from .core import evaluate, AlgorithmBase
import numpy as np


def local_search(X, y, max_neighbours, sigma, seed):
    """Local Search Algorithm

    Keyword arguments:
    X -- Train input
    y -- Train labels
    max_neighbours -- Max number of neighbours to explore.
    sigma -- Standard deviation of Gaussian mutation.
    seed -- Seed to initialize the random generator.
            It is recommended to specify this in order to replicate
            the experiment across executions.
    """
    n_features = X.shape[1]
    np.random.seed(seed)
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
    return weights, trace[trace > 0], n_generated


class LocalSearch(AlgorithmBase):
    """
    Docstring: Wrapper class for Local Search algorithm that provided
    sklearn-based syntax.
    """

    def __init__(self, threshold=0.2, max_neighbours=15000, sigma=0.3, seed=1):
        self.max_neighbours = max_neighbours
        self.sigma = sigma
        self.seed = seed
        self.trace = []
        self.neighbors_generated = 0
        super().__init__(threshold, seed)


    def fit(self, X, y):
        result = local_search(X, y, self.max_neighbours, self.sigma, self.seed)
        self.feature_importances = result[0]
        self.trace = result[1]
        self.neighbors_generated = result[2]
        super().fit(X, y)
