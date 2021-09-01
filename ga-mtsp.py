# pip install scikit-opt

import sys
import numpy as np
import copy
from numpy.lib.function_base import append
from tqdm import tqdm


class Chromosome:

    # Random generated Chromosome
    #  m - number of traveling salesmans
    # def __init__(self, number_of_cities, number_of_traveling_salesman, adj = coordinates_to_adjacency_matrix(data)):
    def __init__(self, number_of_cities, number_of_traveling_salesman, adj):
        self.n = number_of_cities
        self.m = number_of_traveling_salesman
        self.adj = adj
        c = np.array(range(1, number_of_cities))
        np.random.shuffle(c)
        self.solution = np.array_split(c, self.m)
        for i in range(len(self.solution)):
            self.solution[i] = np.insert(self.solution[i], 0, 0)
            self.solution[i] = np.append(self.solution[i], 0)
        self.fitness()

    # Evaluate the Chromosome - Fitness function
    #  based on 2 features:
    #   - overall cost (cumulated from all salesman)
    #   - worst (longest) salesman cost
    #  adj - adjacency matrix
    def fitness(self):
        self.cost = 0
        self.minmax = 0
        # longest_salesman_fitness = []
        # longest_salesman_length = 0
        for i in range(self.m):
            salesman = self.solution[i]
            salesman_fitness = 0
            for j in range(len(salesman) - 1):
                salesman_fitness += self.adj[salesman[j]][salesman[j + 1]]
            self.cost += salesman_fitness
            # if len(salesman) > longest_salesman_length or (len(salesman) == longest_salesman_length and salesman_fitness > self.minmax):
            #     longest_salesman_length = len(salesman)
            #     self.minmax = salesman_fitness
            # if len(salesman) > longest_salesman_length:
            #     longest_salesman_length = len(salesman)
            # if salesman_fitness > self.minmax:
            #     self.minmax = salesman_fitness
            self.minmax = max(self.minmax, salesman_fitness)
        # self.score = self.cost + self.minmax
        self.score = self.minmax
        # self.print()

    # Mutation operator - mutates a single Traveling Salesman
    #  by swaping 2 cities
    def mutate_local(self):
        index = np.random.randint(0, self.m)
        mutant = self.solution[index]
        i, j = (
            np.random.randint(1, len(mutant) - 1),
            np.random.randint(1, len(mutant) - 1),
        )
        mutant[i], mutant[j] = mutant[j], mutant[i]
        old_cost = self.cost
        self.fitness()

    # Mutation operator - mutates 2 Traveling Salesmans
    #  by removing a city from a salesman and asigning it to the second one
    def mutate_global(self):
        for i in range(self.m):
            if len(self.solution[i]) < 3:
                print(i, self.solution[i])

        index1, index2 = (
            np.random.randint(0, self.m),
            np.random.randint(0, self.m),
        )
        while index1 == index2:
            index1, index2 = (
                np.random.randint(0, self.m),
                np.random.randint(0, self.m),
            )
        while len(self.solution[index1]) < 4:
            index1, index2 = (
                np.random.randint(0, self.m),
                np.random.randint(0, self.m),
            )
        mutant1, mutant2 = self.solution[index1], self.solution[index2]
        i, j = (
            np.random.randint(1, len(mutant1) - 1),
            np.random.randint(1, len(mutant2) - 1),
        )
        self.solution[index2] = np.insert(mutant2, j, mutant1[i])
        self.solution[index1] = np.delete(mutant1, i)
        old_cost = self.cost
        self.fitness()

    # PMX Crossover
    def crossover(self, chromosome):
        for index in range(self.m):
            salesman1, salesman2 = (
                self.solution[index],
                chromosome.solution[index],
            )
            for i in range(1, min(len(salesman1), len(salesman2)) - 1):
                try:
                    ind = salesman1.tolist().index(salesman2[i])
                    (salesman1[i], salesman1[ind],) = (
                        salesman1[ind],
                        salesman1[i],
                    )
                except ValueError:
                    pass

                # if salesman2[i] in salesman1:
                #     (
                #         salesman1[i],
                #         salesman1[salesman1.tolist().index(salesman2[i])],
                #     ) = (
                #         salesman1[salesman1.tolist().index(salesman2[i])],
                #         salesman1[i],
                #     )
        self.fitness()

    def print(self):
        total_cost = 0
        minmax = 0
        for i in range(self.m):
            salesman = self.solution[i]
            cost = 0
            print(i + 1, ":  ", self.solution[i][0] + 1, end="", sep="")
            for j in range(1, len(self.solution[i])):
                # print("-", self.solution[i][j]+1, end="", sep="")
                dist = self.adj[salesman[j - 1]][salesman[j]]
                print(
                    "[%.0f]%d" % (dist, self.solution[i][j] + 1),
                    end="",
                    sep="",
                )
                cost += dist
            total_cost += cost
            if cost > minmax:
                minmax = cost
            print(" --- %.0f#" % (cost), len(self.solution[i]))
        print("Cost:   \t%.1f\t%.1f" % (self.cost, total_cost))
        print("Minmax: \t%.1f\t%.1f" % (self.minmax, minmax))
        # print("Cost:   \t%.1f"%(total_cost))
        # print("Minmax: \t%.1f"%(minmax))


class Population:
    # def __init__(self, adj, population_size=50):
    def __init__(
        self,
        population_size,
        number_of_cities,
        number_of_traveling_salesman,
        adj,
    ):
        self.n = number_of_cities
        self.m = number_of_traveling_salesman
        self.adj = adj
        self.population = []
        self.population_size = population_size
        for i in range(population_size):
            self.population.append(
                Chromosome(
                    number_of_cities=self.n,
                    number_of_traveling_salesman=self.m,
                    adj=self.adj,
                )
            )

    # a new crossover inspired by DPX crossover
    def crossover(self, chromosome1, chromosome2):
        c1 = copy.deepcopy(chromosome2)
        c2 = copy.deepcopy(chromosome1)
        c1.solution[0] = chromosome2.solution[-1]
        c1.solution[-1] = chromosome2.solution[0]
        c2.solution[0] = chromosome1.solution[-1]
        c2.solution[-1] = chromosome1.solution[0]
        return (c1, c2)

    # purge
    def purge(self):
        scores = [c.score for c in self.population]
        percentile = np.percentile(
            scores, 100.0 * self.population_size / len(scores)
        )
        # print(
        #     f"\npercentile={percentile},len(population)={len(self.population)}"
        # )
        self.population = [
            d
            for (d, survive) in zip(self.population, scores <= percentile)
            if survive
        ]

    # Genetic Algorithm
    def run_genetic_algorithm(
        self,
        number_of_iterations=1000,
        mutation_probability=0.7,
        crossover_probability=0.7,
    ):

        # Run for a fixed number of iterations
        pbar = tqdm(range(number_of_iterations))
        for it in pbar:

            # Tournament selection
            k = self.population_size
            j = (int)(self.population_size * 0.6)
            for _ in range(self.population_size - k):
                del self.population[
                    -np.random.randint(0, len(self.population))
                ]
            for _ in range(k - j):
                worst_chromosome_score = self.population[0].score
                worst_chromosome_index = 0
                for i in range(1, len(self.population)):
                    if self.population[i].score > worst_chromosome_score:
                        worst_chromosome_score = self.population[i].score
                        worst_chromosome_index = i
                del self.population[-worst_chromosome_index]

            for _ in range(self.population_size - len(self.population)):
                self.population.append(
                    Chromosome(
                        number_of_cities=self.n,
                        number_of_traveling_salesman=self.m,
                        adj=self.adj,
                    )
                )

            # Mutate globally
            for index in range(len(self.population)):
                if np.random.random(1)[0] < mutation_probability:
                    chromosome = copy.deepcopy(self.population[index])
                    chromosome.mutate_global()
                    # if chromosome.score < self.population[index].score:
                    #     self.population[index] = chromosome
                    self.population.append(chromosome)

            # Mutate locally
            for index in range(len(self.population)):
                if np.random.random(1)[0] < mutation_probability:
                    chromosome = copy.deepcopy(self.population[index])
                    chromosome.mutate_local()
                    # if chromosome.score < self.population[index].score:
                    #     self.population[index] = chromosome
                    self.population.append(chromosome)

            # Crossover
            for index1 in range(len(self.population)):
                if np.random.random(1)[0] < crossover_probability:
                    index2 = np.random.randint(0, len(self.population))
                    if index1 == index2:
                        index2 = np.random.randint(0, len(self.population))
                    # child1 = copy.deepcopy(self.population[index1])
                    # child2 = copy.deepcopy(self.population[index2])
                    # child1.crossover(self.population[index2])
                    # child2.crossover(self.population[index1])
                    child1, child2 = self.crossover(
                        self.population[index1], self.population[index2]
                    )
                    # if child1.score < self.population[index1].score:
                    #     self.population[index1] = child1
                    # if child2.score < self.population[index2].score:
                    #     self.population[index2] = child2
                    self.population.append(child1)
                    self.population.append(child2)
            self.purge()
            best_chromosome = self.get_best_result()
            pbar.set_description(f"len(population)={len(self.population)} best:{best_chromosome.cost}\t{best_chromosome.minmax}")
            # print(
            #     f"len(population)={len(self.population)}",
            #     f"best:{best_chromosome.cost}\t{best_chromosome.minmax}",
            # )

    # Print the overall cost and the minmax cost of the best chromosome
    def get_best_result(self):
        best_chromosome = self.population[0]
        for i in range(1, self.population_size):
            if self.population[i].score < best_chromosome.score:
                best_chromosome = self.population[i]
        return best_chromosome
        # print("Overall cost: ", best_chromosome.cost)
        # print("Minmax cost: ", best_chromosome.minmax)


# Helper function to convert the coordinates into an adjacency matrix
def coordinates_to_adjacency_matrix(data, ord=2):
    a = np.zeros((len(data), len(data)))
    for i in range(len(a)):
        for j in range(len(a)):
            if not i == j:
                a[i][j] = np.linalg.norm(data[i] - data[j], ord=ord)
    return a


def run_opt(coordinates, n_of_ts=2, order=1, cycle=10000, population_size=100):

    adjacency = coordinates_to_adjacency_matrix(coordinates, ord=order)
    pop = Population(
        population_size=population_size,
        number_of_cities=len(coordinates),
        number_of_traveling_salesman=n_of_ts,
        adj=adjacency,
    )
    pop.run_genetic_algorithm(number_of_iterations=cycle)
    best = pop.get_best_result()
    best.print()  # Print best solution


def main(argv):
    # Load the graph problem from a .tsp.txt file
    # data = np.loadtxt("data/eil51.tsp.txt", usecols=[1, 2])
    data = np.loadtxt('test/test1.txt', usecols=[0,1])
    n_of_ts = 2
    order = 1
    cycle = 1000  # 10000
    population_size = 100

    run_opt(data, n_of_ts, order, cycle, population_size)


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv)
