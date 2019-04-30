import random
import numpy as np


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
    ind1[:] = np.clip(c1, 0, 1)
    ind2[:] = np.clip(c2, 0, 1)
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
