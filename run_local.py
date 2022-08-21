"""Run GA locally."""

from models.genetic import Population

if __name__ == "__main__":
    bounds = [[0.0, 2.0]]
    n_iter = 100
    n_pop = 100
    r_cross = 0.98

    population: Population = Population(
        population_limit=n_pop,
        variable_limits=bounds,
        crossover_rate=r_cross,
    )
    population.evolve(n_iter)
    print("Done!")
    print(population.fittest_chromosome)
