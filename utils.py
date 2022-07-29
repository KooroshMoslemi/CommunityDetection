import random
import bisect
import itertools
import networkx as nx
import matplotlib.pyplot as plt


class UGraph(nx.Graph):
    """Implementation of Simple Undirected Graph with Adjacency List"""

    class Edge:
        def __init__(self, u_idx, v_idx, x=1):
            self._endPoint1 = u_idx
            self._endPoint2 = v_idx
            self.value = x

        def opposite(self, v):
            return self._endPoint1 if v is self._endPoint2 else self._endPoint2

        def __str__(self):
            return self._endPoint1 + '<->' + self._endPoint2 + ":" + self.value

    def __init__(self):
        self.adjList = {}
        self.nodes_no = 0
        self.idx2label = {}
        self.label2idx = {}
        super(UGraph, self).__init__()

    def add_node(self, node_label, **attr):
        if node_label in self.label2idx.keys():
            return

        self.nodes_no += 1

        self.idx2label[self.nodes_no] = node_label
        self.label2idx[node_label] = self.nodes_no

        self.adjList[self.nodes_no] = []
        super(UGraph, self).add_node(self.nodes_no, **attr)

    def add_edge(self, u, v, **attr):
        u_idx = self.label2idx[u]
        v_idx = self.label2idx[v]

        e1 = self.Edge(u_idx, v_idx, 1)
        self.adjList[u_idx].append(e1)

        e2 = self.Edge(v_idx, u_idx, 1)
        self.adjList[v_idx].append(e2)
        super(UGraph, self).add_edge(u_idx, v_idx, **attr)


def weighted_sampler(seq, weights):
    """Return a random-sample function that picks from seq weighted by weights."""
    totals = []
    for w in weights:
        totals.append(w + totals[-1] if totals else w)
    return lambda: seq[bisect.bisect(totals, random.uniform(0, totals[-1]))]


def init_population(problem, pop_number):
    """Initializes population for genetic algorithm
    pop_number  :  Number of individuals in population
    gene_pool   :  List of possible values for individuals"""
    state_length = problem.nodes_no
    population = []

    for _ in range(pop_number):
        new_individual = [-1] * state_length
        node_idx = 1
        while node_idx <= state_length:
            new_individual[node_idx - 1] = random.choice(problem.adjList[node_idx]).opposite(node_idx)  # safe init
            node_idx += 1
        population.append(new_individual)

    return population


def draw(state,problem):
    n = len(state)
    g = nx.Graph()
    for i in range(1, n + 1):
        j = state[i - 1]

        i = int(problem.idx2label[i])
        j = int(problem.idx2label[j])

        g.add_node(i)
        g.add_node(j)
        g.add_edge(i, j)

    nx.draw(g, edge_color='gray', node_color='purple', node_size=20, width=0.5 , with_labels=True)
    plt.show()


def fitness1(problem,state):
    """Inefficient implementation of modularity"""

    n = len(state)
    g = nx.Graph()
    for i in range(1, n + 1):
        j = state[i - 1]
        g.add_node(i)
        g.add_node(j)
        g.add_edge(i, j)

    components = nx.connected_components(g)
    components = [component for component in components]
    m = problem.number_of_edges()

    q = 0
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            aij = int(problem.has_edge(i, j))
            ki = problem.degree[i]
            kj = problem.degree[j]
            cij = sum([(i in component) and (j in component) for component in components])

            q += (aij - (ki * kj / (2 * m))) * cij

    q /= (2 * m)
    return q



def fitness2(problem,state):
    """Efficient implementation of modularity"""

    n = len(state)
    g = nx.Graph()
    for i in range(1, n + 1):
        j = state[i - 1]
        g.add_node(i)
        g.add_node(j)
        g.add_edge(i, j)

    components = nx.connected_components(g)
    components = [component for component in components]

    out_degree = in_degree = dict(problem.degree(weight="weight"))
    deg_sum = sum(out_degree.values())
    m = deg_sum / 2
    norm = 1 / deg_sum ** 2

    def community_contribution(community):
        comm = set(community)
        L_c = sum(wt for u, v, wt in problem.edges(comm, data="weight", default=1) if v in comm)

        out_degree_sum = sum(out_degree[u] for u in comm)
        in_degree_sum = sum(in_degree[u] for u in comm)

        return L_c / m - out_degree_sum * in_degree_sum * norm

    return sum(map(community_contribution, components))


if __name__ == "__main__":

    # verifying weighted_sampler() functionality
    sampler = weighted_sampler([1,2,3,4,5],[0.1,0.2,0.1,0.3,0.3])
    m = 100000
    cc = [0,0,0,0,0]

    for _ in range(m):
        cc[sampler()-1]+=1

    print([x/m for x in cc])


