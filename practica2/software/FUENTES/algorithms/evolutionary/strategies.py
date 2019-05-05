import numpy as np
import random
from deap import tools
from ..local_search import local_search


def copy_individual(source, dest):
    """
    Copy the content of an individial (value + fitness)
    """
    dest[:] = source[:]
    dest.fitness = source.fitness


def elitism(population, offspring):
    """
    Change the worst individual of the population
    by the best individual of the offspring.
    """
    best = tools.selBest(population, 1)[0]
    worst = tools.selWorst(offspring, 1)[0]
    found = any(np.array_equal(ind, best) for ind in offspring)
    if not found and best.fitness.values[0] > worst.fitness.values[0]:
        copy_individual(best, worst)


def change_worst_ones(population, offspring):
    """
    Change the worst individuals of the population
    by the individuals of the offspring.
    """
    worst_ones = tools.selWorst(population, len(offspring))
    for ind_pop, ind_off in zip(worst_ones, offspring):
        if ind_off.fitness.values[0] > ind_pop.fitness.values[0]:
            copy_individual(ind_off, ind_pop)


def memetic_strategy(population, num_selected, prob, sort, *args, **kwargs):
    """
    Generic function for memetics strategies.

    :param population: list of individuals to apply local search.
    :param num_selected: Number of candidates to apply local search
    :param prob: Probability of each candidate to mutate
    :param sort: if true, it sorts the candidates by fitness.
    :param args: Positional arguments to local_search
    :param kargs: Keyword arguments to local_search
    """
    if sort:
        candidates = tools.selBest(population, num_selected)
    else:
        candidates = population[:num_selected]
    evaluations = 0
    for ind in candidates:
        if random.random() < prob:
            new_ind, trace, n_generated = local_search(
                init_weights=ind, *args, **kwargs)
            evaluations += n_generated
            ind[:] = new_ind[:]
            ind.fitness.values = (trace[-1],)
    return evaluations


def evaluate_population(population, toolbox):
    """
    Evaluate the individuals with an invalid fitness
    """
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    evaluations = 0
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
        evaluations += 1
    return evaluations


def crossover_and_mutate(population, toolbox, cxpb, mutpb):
    """Part of an evolutionary algorithm applying only the variation part
    (crossover and mutation). The modified individuals have their
    fitness invalidated. The individuals are cloned so the returned population
    is independent of the input population.
    """
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


def generational_step(toolbox, pop, cxpb, mupb, mem_strategy, num_generations):
    offspring = toolbox.select(pop, len(pop))
    offspring = crossover_and_mutate(offspring, toolbox, cxpb, mupb)
    num_evaluations = evaluate_population(offspring, toolbox)
    elitism(pop, offspring)
    if mem_strategy and num_generations % 10 == 0:
        num_evaluations += mem_strategy(population=offspring)
    pop[:] = offspring
    return num_evaluations


def stationary_step(toolbox, pop, cxpb, mupb, mem_strategy, num_generations):
    offspring = toolbox.select(pop, 2)
    offspring = crossover_and_mutate(offspring, toolbox, cxpb, mupb)
    num_evaluations = evaluate_population(offspring, toolbox)
    if mem_strategy and num_generations % 10 == 0:
        num_evaluations += mem_strategy(population=offspring)
    change_worst_ones(pop, offspring)
    return num_evaluations


def run(toolbox, population_size, max_evaluations, cxpb, mupb,
        generational=True, mem_strategy=None):
    """
    Run an evolutionary algorithm until a maximun number of evaluations.

    :param toolbox: A :class:`~deap.base.Toolbox` that contains the
                    evolution operators.
    :param population_size: The number of individials of each generation
    :param max_evaluations: Max number of fitness function evaluations.
    :param cxpb: Crossover probability
    :param mupb: Mutation probability
    :param generational: if true, it runs a generational strategy.
                         Else, it runs an stationary strategy
    :param mem_strategy: The memetic strategy to use after crossover and mutation.
    """
    hof = tools.HallOfFame(1, similar=np.array_equal)
    pop = toolbox.population(n=population_size)
    num_generations = 0
    num_evaluations = evaluate_population(pop, toolbox)
    hof.update(pop)
    trace = []
    step_func = generational_step if generational else stationary_step
    while num_evaluations < max_evaluations:
        num_generations += 1
        num_evaluations += step_func(toolbox, pop, cxpb, mupb, mem_strategy,
                                     num_generations)
        hof.update(pop)
        trace.append(hof[0].fitness.values[0])
    return hof[0], trace
