from .core import evaluate, AlgorithmBase
import numpy as np


def mutate(weights):
    candidate = np.copy(weights)
    index = np.random.randint(len(weights))
    perturbation = np.random.normal(0, 0.3)
    candidate[index] = np.clip(candidate[index] + perturbation, 0, 1)
    return candidate


def annealing(X, y, max_eval, seed):
    np.random.seed(seed)
    weights = np.random.rand(X.shape[1])
    best_weights = weights
    fitness = evaluate(best_weights, X, y)
    best_fitness = fitness
    T0 = 0.3 * best_fitness / (-np.log(0.3))
    T = T0
    Tf = np.clip(1e-3, 0, T0)
    evaluations = 0
    accepted = 1
    K = 1
    max_neighbours = 10 * len(weights)
    max_accepted = len(weights)
    M = max_eval / max_neighbours
    trace = np.zeros(max_eval)
    while evaluations < max_eval and accepted > 0 and T > Tf:
        accepted = 0
        current_evals = 0
        while current_evals < max_neighbours and accepted < max_accepted:
            trace[evaluations] = best_fitness
            current_evals += 1
            w_prime = mutate(weights)
            fitness_prime = evaluate(w_prime, X, y)
            diff = fitness_prime - fitness
            prob = np.exp(diff / T)
            if diff > 0 or np.random.random() < prob:
                weights = w_prime
                fitness = fitness_prime
                accepted += 1
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_weights = weights
        evaluations += current_evals
        K += 1
        beta = (T0 - Tf) / (M * T0 * Tf)
        T = T / (1 + beta * T)
    return best_weights, trace[trace > 0]


class SimulatedAnnealing(AlgorithmBase):
    """
    Wrapper class for Local Search algorithm that provided
    sklearn-based syntax.
    """

    def __init__(self, threshold=0.2, max_evaluations=15000, seed=1):
        self.max_evaluations = max_evaluations
        self.trace = []
        super().__init__(threshold, seed)

    def fit(self, X, y):
        """
        Fit the a 1-NN model using Simulated Annealing for feature weighting.

        :param X: Train inputs
        :param y: Train labels
        """
        weights, trace = annealing(X, y, self.max_evaluations, self.seed)
        self.trace = trace
        super().set_feature_importances(weights)
