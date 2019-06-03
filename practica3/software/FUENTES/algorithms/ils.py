from .core import AlgorithmBase, evaluate
from .local_search import local_search
import numpy as np


def mutate(weights):
    """
    Mutation function for ILS.

    It selects randomly 10 percent of the weights
    and mutates it using Gaussian mutation with
    sigma = 0.4.

    :param weights: Candidate to apply the mutation
    :returns: a new candidate
    """
    candidate = np.copy(weights)
    N = len(weights)
    num_comp = int(N * 0.1)
    indices = np.random.permutation(N)[0:num_comp]
    candidate[indices] += np.random.normal(0, 0.4, num_comp)
    candidate = np.clip(candidate, 0, 1)
    return candidate


def ils(X, y, iters, seed):
    """
    Iterated local search algorithm.

    It performs ILS with a limit of 1000 neighbours
    generated per iteration.

    :param X: Input data for fitness evaluation
    :param y: Input labels for fitness evaluation
    :param iters: The number of iterations. Also the number
                   of application of Local search algorithms.
    :param seed: Seed to feed the random number generator.
    """
    dim = X.shape[1]
    np.random.seed(seed)
    init_weights = np.random.rand(dim)
    weights, init_trace, _ = local_search(
        X, y, 1000, 0.3, seed, init_weights, early_stopping=False)
    best_fitness = evaluate(weights, X, y)
    traces = [init_trace]
    for _ in range(iters):
        candidate = mutate(weights)
        candidate, trace, _ = local_search(
            X, y, 1000, 0.3, seed, candidate, early_stopping=False)
        fitness = trace[-1]
        if fitness > best_fitness:
            weights = candidate
            best_fitness = fitness
        traces.append(trace)
    traces = np.concatenate(traces)
    return weights, traces


class ILS(AlgorithmBase):
    """
    Wrapper class for Local Search algorithm that provided
    sklearn-based syntax.
    """

    def __init__(self, threshold=0.2, iterations=15, seed=1):
        self.iterations = iterations
        self.trace = []
        super().__init__(threshold, seed)

    def fit(self, X, y):
        """
        Fit the a 1-NN model using Iterated Local Search for feature weighting.

        :param X: Train inputs
        :param y: Train labels
        """
        weights, trace = ils(X, y, self.iterations, self.seed)
        self.trace = trace
        super().set_feature_importances(weights)
