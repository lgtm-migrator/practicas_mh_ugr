from .core import AlgorithmBase, evaluate
import numpy as np


def rand_one(_, current_idx, pop, mut):
    idxs = [idx for idx in range(len(pop)) if idx != current_idx]
    a, b, c = pop[np.random.choice(idxs, 3, replace=False), :]
    return np.clip(a + mut * (b - c), 0, 1)


def current_to_best_one(best_idx, current_idx, pop, mut):
    idxs = [idx for idx in range(len(pop)) if idx != current_idx]
    a, b = pop[np.random.choice(idxs, 2, replace=False), :]
    x = pop[current_idx]
    best = pop[best_idx]
    return x + mut * (best - x) + mut * (a - b)


def de(X, y, iters, strategy=rand_one, mut=0.5, crossp=0.5, popsize=50, seed=1):
    np.random.seed(seed)
    N = X.shape[1]
    pop = np.random.rand(popsize, N)
    fitness = np.asarray([evaluate(ind, X, y) for ind in pop])
    best_idx = np.argmin(fitness)
    best = pop[best_idx]
    for _ in range(iters):
        for j in range(popsize):
            mutant = strategy(best_idx, j, pop, mut)
            cross_points = np.random.rand(N) < crossp
            trial = np.where(cross_points, mutant, pop[j])
            f = evaluate(trial, X, y)
            if f > fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f > fitness[best_idx]:
                    best_idx = j
                    best = trial
    return best


class DifferentialEvolution(AlgorithmBase):
    """
    Wrapper class for Local Search algorithm that provided
    sklearn-based syntax.
    """

    def __init__(self,
                 strategy,
                 popsize=50,
                 threshold=0.2,
                 max_evaluations=15000,
                 seed=1):
        self.popsize = popsize
        self.max_evaluations = max_evaluations
        self.trace = []
        if strategy == 'rand_one':
            self.strategy = rand_one
        else:
            self.strategy = current_to_best_one
        super().__init__(threshold, seed)

    def fit(self, X, y):
        """
        Fit the a 1-NN model using Differential Evolution for feature weighting.
        This algorithm uses Scipy.optimize function. That function is used for
        minimizing. So, we need to use the inverse of our fitness function.
        It is just -fitness(w, X, y)

        :param X: Train inputs
        :param y: Train labels
        """
        maxiter = int(self.max_evaluations / self.popsize)
        weights = de(X, y, maxiter, self.strategy, seed=self.seed)
        super().set_feature_importances(weights)
