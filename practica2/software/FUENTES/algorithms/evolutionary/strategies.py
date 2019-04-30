import numpy as np
import random
from deap import tools
from ..local_search import local_search


def copy_individual(source, dest):
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


def memetic_strategy(X, y, max_neighbours, seed, population, num_selected,
                     prob, sort):
    if sort:
        candidates = tools.selBest(population, num_selected)
    else:
        candidates = population[:num_selected]
    evaluations = 0
    for ind in candidates:
        if random.random() < prob:
            new_ind, trace, n_generated = local_search(X, y, max_neighbours,
                                                       0.3, seed, ind)
            evaluations += n_generated
            ind[:] = new_ind[:]
            ind.fitness.values = (trace[-1], )
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


def run(toolbox,
        population_size,
        max_evaluations,
        cxpb,
        mupb,
        generational=True,
        mem_strategy=None):
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
