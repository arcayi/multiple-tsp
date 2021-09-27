# pip install scikit-opt

import sys
import numpy as np
import copy
from numpy.core.numeric import zeros_like
from tqdm import tqdm
from typing import List
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import concurrent


class Ecosystem:
    adj: np.ndarray
    n_iterations: int
    mutation_probability: float
    crossover_probability: float
    # population: Population
    # pop_size: int
    # n_cities: int
    # n_ts: int

    def __init__(
        self,
        coordinates,
        order=1,
        number_of_iterations=1000,
        mutation_probability=0.7,
        crossover_probability=0.7,
    ):
        self.adj = coordinates_to_adjacency_matrix(coordinates, ord=order)
        self.n_iterations = number_of_iterations
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability


class Chromosome:
    n_cities: int
    m_ts: int
    solution: List[np.ndarray]
    eco: Ecosystem
    cost: float
    minmax: float
    score: float

    # Random generated Chromosome
    #  m - number of traveling salesmans
    # def __init__(self, number_of_cities, number_of_traveling_salesman, adj = coordinates_to_adjacency_matrix(data)):
    def __init__(self, number_of_cities, number_of_traveling_salesman, eco):
        self.n_cities = number_of_cities
        self.m_ts = number_of_traveling_salesman
        self.eco = eco

        c = np.array(range(1, self.n_cities))
        np.random.shuffle(c)
        self.solution = np.array_split(c, self.m_ts)

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
        sm_fitness = [
            sum(
                [
                    self.eco.adj[self.solution[i][j]][self.solution[i][j + 1]]
                    for j in range(len(self.solution[i]) - 1)
                ]
            )
            for i in range(self.m_ts)
        ]
        # print(sm_fitness)

        # for i in range(self.m_ts):
        #     salesman = self.solution[i]
        #     sm_fitness.append(0)
        #     for j in range(len(salesman) - 1):
        #         sm_fitness[i] += self.eco.adj[salesman[j]][salesman[j + 1]]
        self.cost = sum(sm_fitness)
        self.minmax = max(sm_fitness)
        # self.score = self.cost + self.minmax
        self.score = self.minmax

    # Mutation operator - mutates a single Traveling Salesman
    #  by swaping 2 cities
    def mutate_local(self):
        index = np.random.randint(0, self.m_ts)
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
        for i in range(self.m_ts):
            if len(self.solution[i]) < 3:
                print(i, self.solution[i])

        index1, index2 = (
            np.random.randint(0, self.m_ts),
            np.random.randint(0, self.m_ts),
        )
        while index1 == index2:
            index1, index2 = (
                np.random.randint(0, self.m_ts),
                np.random.randint(0, self.m_ts),
            )
        while len(self.solution[index1]) < 4:
            index1, index2 = (
                np.random.randint(0, self.m_ts),
                np.random.randint(0, self.m_ts),
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
        for index in range(self.m_ts):
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
        for i in range(self.m_ts):
            salesman = self.solution[i]
            cost = 0
            print(i + 1, ":  ", self.solution[i][0] + 1, end="", sep="")
            for j in range(1, len(self.solution[i])):
                # print("-", self.solution[i][j]+1, end="", sep="")
                dist = self.eco.adj[salesman[j - 1]][salesman[j]]
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
    n_cities: int
    m_ts: int
    population: List[Chromosome]
    eco: Ecosystem

    # def __init__(self, adj, population_size=50):
    def __init__(
        self,
        population_size,
        number_of_cities,
        number_of_traveling_salesman,
        eco,
    ):
        self.population_size = population_size
        self.n_cities = number_of_cities
        self.m_ts = number_of_traveling_salesman
        self.eco = eco
        self.population = []
        for i in range(population_size):
            self.population.append(
                Chromosome(
                    number_of_cities=self.n_cities,
                    number_of_traveling_salesman=self.m_ts,
                    eco=self.eco,
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

    def reproduction(self, index):
        rand_probability = np.random.random(3)
        # Mutate globally

        if rand_probability[0] < self.eco.mutation_probability:
            chromosome = copy.deepcopy(self.population[index])
            chromosome.mutate_global()
            self.population.append(chromosome)

        # Mutate locally
        if rand_probability[1] < self.eco.mutation_probability:
            chromosome = copy.deepcopy(self.population[index])
            chromosome.mutate_local()
            self.population.append(chromosome)

        # Crossover
        if rand_probability[2] < self.eco.crossover_probability:
            index2 = np.random.randint(0, len(self.population))
            if index == index2:
                index2 = np.random.randint(0, len(self.population))
            child1, child2 = self.crossover(
                self.population[index], self.population[index2]
            )
            self.population.append(child1)
            self.population.append(child2)

    # Genetic Algorithm
    def run_genetic_algorithm_cc(self, max_workers=10):
        pbar = tqdm(range(self.eco.n_iterations))
        for it in pbar:
            # k = self.population_size
            # j = (int)(self.population_size * 0.6)
            # for _ in range(self.population_size - k):
            #     del self.population[
            #         -np.random.randint(0, len(self.population))
            #     ]
            # for _ in range(k - j):
            #     worst_chromosome_score = self.population[0].score
            #     worst_chromosome_index = 0
            #     for i in range(1, len(self.population)):
            #         if self.population[i].score > worst_chromosome_score:
            #             worst_chromosome_score = self.population[i].score
            #             worst_chromosome_index = i
            #     del self.population[-worst_chromosome_index]

            for _ in range((int)(self.population_size * 0.4)):
                self.population.append(
                    Chromosome(
                        number_of_cities=self.n_cities,
                        number_of_traveling_salesman=self.m_ts,
                        eco=self.eco,
                    )
                )

            range_pop = range(len(self.population))

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # pass
                # executor.map(self.reproduction, range_pop)
                futures = [
                    executor.submit(self.reproduction, item)
                    for item in range_pop
                ]
                for future in concurrent.futures.as_completed(futures):
                    pass
                    # print(future.result())

            self.purge()
            best_chromosome = self.get_best_result()
            pbar.set_description(
                f"len(pop)={len(self.population)} best:{best_chromosome.cost} {best_chromosome.minmax}"
            )

    # Genetic Algorithm
    def run_genetic_algorithm(self):

        # Run for a fixed number of iterations
        pbar = tqdm(range(self.eco.n_iterations))
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
                        number_of_cities=self.n_cities,
                        number_of_traveling_salesman=self.m_ts,
                        eco=self.eco,
                    )
                )

            range_pop = range(len(self.population))

            # Mutate globally
            for index in range_pop:
                if np.random.random(1)[0] < self.eco.mutation_probability:
                    chromosome = copy.deepcopy(self.population[index])
                    chromosome.mutate_global()
                    # if chromosome.score < self.population[index].score:
                    #     self.population[index] = chromosome
                    self.population.append(chromosome)

            # Mutate locally
            for index in range_pop:
                if np.random.random(1)[0] < self.eco.mutation_probability:
                    chromosome = copy.deepcopy(self.population[index])
                    chromosome.mutate_local()
                    # if chromosome.score < self.population[index].score:
                    #     self.population[index] = chromosome
                    self.population.append(chromosome)

            # Crossover
            for index1 in range_pop:
                if np.random.random(1)[0] < self.eco.crossover_probability:
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
            pbar.set_description(
                f"len(pop)={len(self.population)} best:{best_chromosome.cost} {best_chromosome.minmax}"
            )
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


adjacency = []
population = []
fitness = []
pop_size = 0
n_cities = 0
n_ts = 0


def pop_initialize(
    population_size, number_of_cities, number_of_traveling_salesman
):
    pop_size = population_size
    n_cities = number_of_cities
    n_ts = number_of_traveling_salesman
    population = []
    for i in range(pop_size):
        # self.population.append(
        #     Chromosome(
        #         number_of_cities=self.n_cities,
        #         number_of_traveling_salesman=self.m_ts,
        #         adj=eco.adj,
        #     )
        # )

        ch = np.array(range(1, n_cities))
        np.random.shuffle(ch)
        population.append(np.array_split(ch, n_ts))
        for j in range(len(population[i])):
            population[i][j] = np.insert(population[i][j], 0, 0)
            population[i][j] = np.append(population[i][j], 0)
        fitness(population[i])


def run_opt(
    coordinates,
    n_of_ts=2,
    order=1,
    cycle=10000,
    population_size=100,
    max_workers=10,
):

    # distance matrix
    eco = Ecosystem(coordinates, order=1, number_of_iterations=cycle)

    # pop_initialize()
    # pop_new_generation()
    # fitness_evaluation()
    # end_criteria()
    # selection()
    # crossover()
    # mutation()

    pop = Population(
        population_size=population_size,
        number_of_cities=len(coordinates),
        number_of_traveling_salesman=n_of_ts,
        eco=eco,
    )
    pop.run_genetic_algorithm()
    # pop.run_genetic_algorithm_cc(max_workers=max_workers)
    best = pop.get_best_result()
    best.print()  # Print best solution


def main(argv):
    # Load the graph problem from a .tsp.txt file
    # data = np.loadtxt("data/eil51.tsp.txt", usecols=[1, 2])
    data = np.loadtxt("test/test1.txt", usecols=[0, 1])
    n_of_ts = 2
    order = 1
    cycle = 1000  # 10000
    population_size = 100
    max_workers = 20

    # n_process = 12
    # with ThreadPoolExecutor(max_workers=n_process) as executor:
    #     # pass
    #     # executor.map(self.reproduction, range_pop)
    #     futures = [
    #         executor.submit(
    #             run_opt,
    #             data,
    #             n_of_ts,
    #             order,
    #             cycle,
    #             population_size,
    #             max_workers,
    #         )
    #         for i in range(n_process)
    #     ]
    #     for future in concurrent.futures.as_completed(futures):
    #         pass
    #         # print(future.result())

    run_opt(data, n_of_ts, order, cycle, population_size, max_workers)


if __name__ == "__main__":
    # execute only if run as a script
    main(sys.argv)
