import random
from functools import partial

import numpy as np
from deap import base, creator, tools

from ..core import evaluate, AlgorithmBase
from .operators import mut_gaussian, cx_arithmetic, cx_blx
from .strategies import memetic_strategy, run

creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)


def evaluate_individual(*args, **kwargs):
    """
    Wrapper for fitness function to use with Deap framework.
    """
    return (evaluate(*args, **kwargs),)


def create_toolbox(X, y, mate='blx'):
    """
    Create a :class:`~deap.base.Toolbox` that contains
    the evolution operators (crossover, mutation and selection).
    And the population generation schema.
    """
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, n=X.shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual, X=X, y=y)
    toolbox.register("mutate", mut_gaussian, indpb=0.001, mu=0, sigma=0.3)
    toolbox.register("select", tools.selTournament, tournsize=2)
    if mate == 'blx':
        toolbox.register("mate", cx_blx, alpha=0.3)
    else:
        toolbox.register("mate", cx_arithmetic)
    return toolbox


class EvolutionaryAlgorithm(AlgorithmBase):
    """
    Wrapper class with sklearn-based syntax
    for evolutionary algorithms.
    """

    def __init__(self,
                 threshold=0.2,
                 population_size=30,
                 num_evaluations=15000,
                 mate='blx',
                 generational=True):
        self.mate = mate
        self.generational = generational
        self.population_size = population_size
        self.num_evaluations = num_evaluations
        self.trace = None
        super().__init__(threshold)

    def fit(self, X, y):
        toolbox = create_toolbox(X, y, self.mate)
        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)
            weights, self.trace = run(toolbox, self.population_size,
                                      self.num_evaluations, 0.7, 1,
                                      self.generational)
        self.set_feature_importances(np.array(weights))


class MemeticAlgorithm(EvolutionaryAlgorithm):
    """
    Wrapper class with sklearn-based syntax
    for memetic algorithms.
    """

    def __init__(self, strategy, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if strategy == 'AM-(1,1.0)':
            n_sel = self.population_size
            prob = 1
            sort = False
        elif strategy == 'AM-(1,0.1)':
            n_sel = self.population_size
            prob = 0.1
            sort = False
        elif strategy == 'AM-(1,0.1mej)':
            n_sel = self.population_size // 10
            prob = 1
            sort = True
        self.strategy = partial(
            memetic_strategy, prob=prob, sort=sort, num_selected=n_sel, sigma=0.3)

    def fit(self, X, y):
        toolbox = create_toolbox(X, y, self.mate)
        random.seed(self.seed)
        np.random.seed(self.seed)
        n = X.shape[1] * 2
        strategy = partial(
            self.strategy, X=X, y=y, seed=self.seed, max_neighbours=n)
        weights, self.trace = run(toolbox, self.population_size,
                                  self.num_evaluations, 0.7, 1,
                                  self.generational, strategy)
        self.set_feature_importances(np.array(weights))
