from config import beta,alpha,pa,niter
from utils import init_population
import numpy as np
import random
import math


class Cuckoo:
    def __init__(self, problem, fitness_fn ,nest=None):
        self.nest = nest
        self.problem = problem
        self.fitness_fn = fitness_fn
        self.fitness = self.fitness_fn(self.problem,nest)

    def __levy_flight(self, nest, problem):  # discrete implementation for community detection

        dimension = len(nest)
        nest = np.array(nest)

        sigma1 = np.power((math.gamma(1 + beta) * np.sin((np.pi * beta) / 2)) / math.gamma((1 + beta) / 2) * np.power(2, (beta - 1) / 2), 1 / beta)
        sigma2 = 1
        mu = np.random.normal(0, sigma1, size=dimension)
        v = np.random.normal(0, sigma2, size=dimension)
        step = mu / np.power(np.fabs(v), 1 / beta)
        step *= alpha

        sig_step = np.fabs((1 - np.exp(-1 * step)) / (1 + np.exp(-1 * step)))
        new_nest = []
        for j in range(dimension):
            p = random.random()
            if p < sig_step[j]:
                new_nest.append(nest[j])
            else:
                k = random.choice(problem.adjList[j + 1]).opposite(j + 1)
                new_nest.append(k)
        return new_nest

    def abandon_nest(self, problem):
        new_nest = init_population(problem, 1)
        self.nest = new_nest[0]
        self.fitness = self.fitness_fn(self.problem,self.nest)

    def get_cuckoo(self, problem):  # by levy flight
        return self.__levy_flight(self.nest, problem)


def cuckoo_algorithm(population, fitness_fn, problem, f_thres=1):
    cuckoos = []
    N = len(population)
    for nest in population:
        cuckoos.append(Cuckoo(problem,fitness_fn,nest))

    cuckoos = sorted(cuckoos, key=lambda cuckoo: cuckoo.fitness,reverse=True)
    best_nest = cuckoos[0].nest
    best_fitness = cuckoos[0].fitness

    iter = 0
    while iter < niter and best_fitness < f_thres:

        for i, cuckoo in enumerate(cuckoos):

            candidate_nest = cuckoo.get_cuckoo(problem)
            candidate_fitness = fitness_fn(problem,candidate_nest)

            j = random.randint(0, N-1)
            while j == i:  # random j != i
                j = random.randint(0, N-1)

            if cuckoos[j].fitness < candidate_fitness:  # for maximization
                cuckoos[j].nest = candidate_nest
                cuckoos[j].fitness = candidate_fitness

        cuckoos = sorted(cuckoos, key=lambda cuckoo: cuckoo.fitness,reverse=True)

        for cuckoo in cuckoos[int(-1 * N * pa):]:
            cuckoo.abandon_nest(problem)

        cuckoos = sorted(cuckoos, key=lambda cuckoo: cuckoo.fitness,reverse=True)
        best_nest = cuckoos[0].nest
        best_fitness = cuckoos[0].fitness
        avg_fitness = sum([cuckoo.fitness for cuckoo in cuckoos])

        print(f"[+] iter {iter} - best score: {best_fitness} - avg score: {avg_fitness/N}")

        iter += 1

    return best_nest
