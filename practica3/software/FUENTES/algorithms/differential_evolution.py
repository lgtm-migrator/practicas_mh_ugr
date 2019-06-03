from .core import AlgorithmBase, evaluate
import numpy as np


def rand_one(_, current_idx, pop, mut):
    """
    Mutation function for rand/one strategy.

    :param current_idx: Index of the individual to apply mutation.
    :param pop: Current opulation (list)
    :param mut: Mutation factor, also known as F.

    :returns: a new mutated candidate
    """
    indices = np.random.permutation(len(pop))[0:3]
    a, b, c = pop[indices, :]
    candidate = a + mut * (b - c)
    return np.clip(candidate, 0, 1)


def current_to_best_one(best_idx, current_idx, pop, mut):
    """
    Mutation function for current-to-best/one strategy.

    :param best_idx: Index of the best individual of the population.
    :param current_idx: Index of the individual to apply mutation.
    :param pop: Current opulation (list)
    :param mut: Mutation factor, also known as F.

    :returns: a new mutated candidate
    """
    indices = np.random.permutation(len(pop))[0:2]
    a, b = pop[indices, :]
    x = pop[current_idx]
    best = pop[best_idx]
    candidate = x + mut * (best - x) + mut * (a - b)
    return np.clip(candidate, 0, 1)


def de(X, y, iters, strategy=rand_one, mut=0.5, crossp=0.5, popsize=50, seed=1):
    """
    Differential evolution algorithm.

    :param X: Input data for fitness evaluation.
    :param y: Input labels for fitness evaluation.
    :param iters: Maximun number of generations.
    :param strategy: Mutation strategy function.
    :param mut: Mutation factor, also known as F.
    :param crossp: Crossover probability for each gene individually.
    :param popsize: Initial population size.
    :param seed: Seed to feed the random number generator.
    """
    np.random.seed(seed)
    N = X.shape[1]
    pop = np.random.rand(popsize, N)
    fitness = np.asarray([evaluate(ind, X, y) for ind in pop])
    best_idx = np.argmax(fitness)
    best = pop[best_idx]
    trace = np.zeros(popsize * iters)
    for i in range(iters):
        for j in range(popsize):
            best_fitness = fitness[best_idx]
            trace[i * popsize + j] = best_fitness
            mutant = strategy(best_idx, j, pop, mut)
            cross_points = np.random.rand(N) < crossp
            trial = np.where(cross_points, mutant, pop[j])
            f = evaluate(trial, X, y)
            if f > fitness[j]:
                fitness[j] = f
                pop[j] = trial
                if f > best_fitness:
                    best_idx = j
                    best = trial
    return best, trace


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

        :param X: Train inputs
        :param y: Train labels
        """
        maxiter = int(self.max_evaluations / self.popsize)
        weights, trace = de(X, y, maxiter, self.strategy, seed=self.seed)
        self.trace = trace
        super().set_feature_importances(weights)
