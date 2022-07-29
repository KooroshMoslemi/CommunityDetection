from genetic import genetic_algorithm
from cuckoo import cuckoo_algorithm
from config import *
from utils import *


if __name__ == '__main__':

    problem = UGraph()
    with open("graph.txt", "r") as f:
        lines = f.readlines()
        for i in range(1, len(lines)):
            u, v = lines[i].split()
            problem.add_node(u)
            problem.add_node(v)
            problem.add_edge(u, v)

    nx.draw(problem, edge_color='gray', node_color='purple', node_size=20, width=0.5 , with_labels=True)
    plt.show()

    # Verifying results
    # cuckoo_state = [2, 1, 4, 1, 7, 23, 23, 3, 18, 6, 2, 2, 2, 2, 1, 1, 30, 24, 24, 33, 17, 9, 6, 22, 22, 24, 24, 24, 24, 33, 20, 22, 20, 24]
    # genetic_state = [16, 16, 8, 1, 7, 7, 6, 4, 22, 5, 2, 4, 2, 2, 2, 2, 30, 9, 24, 31, 17, 24, 7, 32, 24, 24, 22, 22, 22, 33, 20, 34, 20, 32]
    # state = genetic_state
    # print(fitness1(problem,state))
    # print(fitness2(problem,state))
    # draw(state,problem)

    # Generating Population
    # population = init_population(problem, cuckoo_population_no)
    # print("[+] Generated first population")
    #
    # solution = genetic_algorithm(population, fitness2, problem, f_thres=genetic_fitness_thresh, ngen=ngen, pmut=mutation_rate)
    # print(solution)
    # print(fitness1(problem,solution))

    # solution = cuckoo_algorithm(population,fitness2,problem,f_thres=cuckoo_fitness_thresh)
    # print(solution)
    # print(fitness1(problem,solution))
