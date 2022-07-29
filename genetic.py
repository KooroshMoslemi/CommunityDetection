import random
from utils import weighted_sampler


def genetic_algorithm(population, fitness_fn, problem, f_thres=None, ngen=1000, pmut=0.1):

    fittest_individual = population[0]
    fittest_individual_score = -1
    fitness_fn2 = lambda x:fitness_fn(problem,x)

    for i in range(ngen):

        population = [mutate(crossover_uniform(*select(2, population, fitness_fn2)), problem, pmut)
                      for i in range(len(population))]

        fittest_individual_candidate = max(population, key=fitness_fn2)
        fittest_individual_candidate_score = fitness_fn2(fittest_individual_candidate)

        if fittest_individual_candidate_score > fittest_individual_score:
            fittest_individual = fittest_individual_candidate
            fittest_individual_score = fittest_individual_candidate_score

        fitnesses = map(fitness_fn2, population)
        avg_fitness = sum(fitnesses)/len(population)
        print(f"[+] iter {i} - best score: {fittest_individual_score} - avg score: {avg_fitness}")

        if f_thres is not None:
            if fittest_individual_score >= f_thres:
                return fittest_individual

    return fittest_individual


def select(r, population, fitness_fn2):
    fitnesses = map(fitness_fn2, population)
    sampler = weighted_sampler(population, fitnesses)
    return [sampler() for i in range(r)]


def crossover_uniform(x, y):
    n = len(x)
    result = [-1] * n
    tosses = random.choices([0, 1], k=n)
    for i in range(n):
        ix = tosses[i]
        result[i] = x[i] if ix < 0.5 else y[i]

    return result


def mutate(x, problem, pmut):
    if random.uniform(0, 1) >= pmut:
        return x

    n = len(x)
    c = random.randrange(0, n)

    new_gene = random.choice(problem.adjList[c+1]).opposite(c+1)  # safe mutate
    return x[:c] + [new_gene] + x[c + 1:]
