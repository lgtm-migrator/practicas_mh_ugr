import random
import numpy as np
from .core import evaluate, AlgorithmBase

from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)


def evaluate_individual(ind, X, y):
    return (evaluate(ind, X, y),)


def cx_arithmetic(ind1, ind2):
    c1 = (ind1 + ind2) / 2
    alphas = np.random.rand(len(ind1))
    c2 = (alphas * ind1 + (1 - alphas) * ind2) / 2
    ind1[:] = c1
    ind2[:] = c2
    return ind1, ind2


def cx_blx(ind1, ind2, alpha):
    c_max = np.maximum(ind1, ind2)
    c_min = np.minimum(ind1, ind2)
    inteval = c_max - c_min
    c1 = np.random.uniform(c_min - inteval * alpha, c_max + inteval * alpha)
    c2 = np.random.uniform(c_min - inteval * alpha, c_max + inteval * alpha)
    ind1[:] = c1
    ind2[:] = c2
    return ind1, ind2


def mut_gaussian(individual, mu, sigma, indpb):
    size = len(individual)
    mutated = False
    for i in range(size):
        if random.random() < indpb:
            mutated = True
            individual[i] += random.gauss(mu, sigma)
            if individual[i] > 1:
                individual[i] = 1
            elif individual[i] < 0:
                individual[i] = 0
    return individual, mutated


def create_toolbox(X, y, mate='blx'):
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


def elitism(population, offspring):
    """
    Change the worst individual of the population
    by the best individual of the offspring.
    """
    best = tools.selBest(population, 1)[0]
    worst = tools.selWorst(offspring, 1)[0]
    found = False
    for ind in offspring:
        if np.array_equal(ind, best):
            found = True
            break
    if not found:
        worst[:] = best[:]
        worst.fitness = best.fitness


def change_worst_ones(population, offspring):
    """
    Change the worst individuals of the population
    by the individuals of the offspring.
    """
    worst_ones = tools.selWorst(population, len(offspring))
    for ind_pop, ind_off in zip(worst_ones, offspring):
        if ind_pop.fitness.values[0] < ind_off.fitness.values[0]:
            ind_pop[:] = ind_off[:]
            ind_pop.fitness = ind_off.fitness


def evaluate_population(population, toolbox):
    """
    Evaluate the individuals with an invalid fitness
    """
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    return len(invalid_ind)


def crossover_and_mutate(population, toolbox, cxpb, mutpb):
    offspring = [toolbox.clone(ind) for ind in population]
    num_crossovers = int(cxpb * len(offspring))
    num_mutations = int(mutpb * len(offspring))
    for i in range(1, num_crossovers, 2):
        offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
        del offspring[i - 1].fitness.values, offspring[i].fitness.values
    for i in range(num_mutations):
        offspring[i], mutated = toolbox.mutate(offspring[i])
        if mutated:
            del offspring[i].fitness.values
    return offspring


def run_generational(toolbox, population_size, num_generations, cxpb, mupb):
    hof = tools.HallOfFame(1, similar=np.array_equal)
    pop = toolbox.population(n=population_size)
    counter = evaluate_population(pop, toolbox)
    while counter < 15000:
        offspring = toolbox.select(pop, len(pop))
        offspring = crossover_and_mutate(offspring, toolbox, cxpb, mupb)
        counter += evaluate_population(offspring, toolbox)
        elitism(pop, offspring)
        pop[:] = offspring
        hof.update(pop)
    return hof[0]


def run_stationary(toolbox, population_size, num_generations, cxpb, mupb):
    hof = tools.HallOfFame(1, similar=np.array_equal)
    pop = toolbox.population(n=population_size)
    counter = evaluate_population(pop, toolbox)
    hof.update(pop)
    while counter < 15000:
        offspring = toolbox.select(pop, 2)
        offspring = crossover_and_mutate(offspring, toolbox, cxpb, mupb)
        counter += evaluate_population(offspring, toolbox)
        change_worst_ones(pop, offspring)
        hof.update(offspring)
    return hof[0]


class EvolutionaryAlgorithm(AlgorithmBase):
    """
    Docstring: Wrapper class with sklearn-based syntax
    for evolutionary algorithms.
    """

    def __init__(self,
                 threshold=0.2,
                 population_size=30,
                 num_generations=250,
                 mate='blx',
                 has_elitism=True):
        self.mate = mate
        self.has_elitism = has_elitism
        self.population_size = population_size
        self.num_generations = num_generations
        super().__init__(threshold)

    def fit(self, X, y):
        toolbox = create_toolbox(X, y, self.mate)
        if self.seed:
            random.seed(self.seed)
            np.random.seed(self.seed)
        if self.has_elitism:
            weights = run_generational(toolbox, self.population_size,
                                       self.num_generations, 0.7, 1)
        else:
            weights = run_stationary(toolbox, self.population_size,
                                     self.num_generations, 1, 1)
        self.feature_importances = np.array(weights)
        super().fit(X, y)
