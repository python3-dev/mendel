"""
Definition of core genetic algorithm models.

A signigicant part of this code is inspired from
https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/

However, I have made significant modifications to this code
adding new features and, importantly, adding readability.
"""

from typing import List, Literal, Tuple

from numpy.random import rand, randint

from config.config import Config
from exceptions.exceptions import InsufficientBitArrayLength, ZeroLengthBit
from models.objective import Objective


class Chromosome(object):
    """Chromosome class."""

    def __init__(
        self,
        bits_array: List[int],
        variable_limits: List[List[float]],
    ) -> None:
        """
        Initialise Chromosome class.

        Chromosome class represent an individual in the population.

        Parameters
        ----------
        bits_array : List[Literal[1, 0]]
            List of 1 and 0 representing the genotype of the individual.
        variable_limits : List[List[float]]
            Upper and lower bounds of variables as list of list of floats
        """
        self.bits_array: List[int] = bits_array
        self.variable_limits: List[List[float]] = variable_limits
        self.genes: List[Gene] = []
        if round(len(self.bits_array) / len(self.variable_limits)) == Config.BIT_LENGTH:
            self.constraint_penalty: int = 0
            self.__generate_genes()
            self.phenotype_array: List[float] = [
                gene.phenotype_value for gene in self.genes
            ]
            self.evaluation: float = Objective.objective(self.phenotype_array)
            self.__compute_constraint_penalty()
            self.__evaluate_fitness()
        else:
            raise InsufficientBitArrayLength

    def __hash__(self) -> int:
        """Chrosome hash function."""
        return int("".join([str(bit_) for bit_ in self.bits_array]))

    def __repr__(self) -> str:
        """Chromosome __repr__ function."""
        return f"Chromosome({self.bits_array}, {self.variable_limits})"

    def __str__(self) -> str:
        """__str__ function."""
        return f"P: {self.phenotype_array}, F: {self.fitness_score}"

    def __eq__(self, __o: object) -> bool:
        """Chromosomal equality."""
        if isinstance(__o, self.__class__):
            return self.fitness_score == __o.fitness_score
        return False

    def __lt__(self, __o: "Chromosome") -> bool:
        """Chromosomal less than comparison."""
        if isinstance(__o, self.__class__):
            return self.fitness_score < __o.fitness_score
        return False

    def __gt__(self, __o: "Chromosome") -> bool:
        """Chromosomal greater than comparison."""
        if isinstance(__o, self.__class__):
            return self.fitness_score > __o.fitness_score
        return False

    def __le__(self, __o: "Chromosome") -> bool:
        """Chromosomal less than or equal to comparison."""
        if isinstance(__o, self.__class__):
            return self.fitness_score <= __o.fitness_score
        return False

    def __ge__(self, __o: "Chromosome") -> bool:
        """Chromosomal greater than or equal to comparison."""
        if isinstance(__o, self.__class__):
            return self.fitness_score >= __o.fitness_score
        return False

    def __generate_genes(self) -> None:
        for index in range(len(self.variable_limits)):
            start_index: int = index * Config.BIT_LENGTH
            stop_index: int = (index * Config.BIT_LENGTH) + Config.BIT_LENGTH
            lower_bound: float = self.variable_limits[index][0]
            upper_bound: float = self.variable_limits[index][-1]
            self.genes.append(
                Gene(
                    bits_array=self.bits_array[start_index:stop_index],
                    upper_bound=upper_bound,
                    lower_bound=lower_bound,
                )
            )

    def __compute_constraint_penalty(self) -> None:
        for index, phenotype_value in enumerate(self.phenotype_array):
            lower_bound_penalty: Literal[1, 0] = (
                0 if phenotype_value >= self.variable_limits[index][0] else 1
            )
            upper_bound_penalty: Literal[1, 0] = (
                0 if phenotype_value <= self.variable_limits[index][-1] else 1
            )
            self.constraint_penalty = lower_bound_penalty + upper_bound_penalty + 1

    def __evaluate_fitness(self) -> None:
        self.fitness_score: float = self.evaluation * (1 / self.constraint_penalty)


class Gene:
    """Gene represent a variable."""

    def __init__(
        self,
        bits_array: List[int],
        upper_bound: float,
        lower_bound: float,
    ) -> None:
        """
        Initialise Gene class.

        Parameters
        ----------
        bits_array : List[Literal[1,0]]
            Bits composing a string.
        upper_bound : float
            Upper bound of phenotypical value of the variable (gene).
        lower_bound : float
            Lower bound of phenotypical value of the variable (gene).
        """
        self.bits_array: List[int] = bits_array
        self.upper_bound: float = upper_bound
        self.lower_bound: float = lower_bound
        self.maximum_bit_value: int = 2**Config.BIT_LENGTH
        self.bit_string: str = "".join([str(bit_) for bit_ in self.bits_array])
        if self.bit_string:
            self.bit_value: int = int(self.bit_string, 2)
        else:
            raise ZeroLengthBit
        self.phenotype_value: float = self.lower_bound + (
            self.bit_value / self.maximum_bit_value
        ) * (self.upper_bound - self.lower_bound)

    def __repr__(self) -> str:
        """Gene representation."""
        return f"Gene({self.bits_array}, {self.upper_bound}, {self.lower_bound}, {self.maximum_bit_value})"


class Population:
    """Population class."""

    def __init__(
        self,
        population_limit: int,
        variable_limits: List[List[float]],
        crossover_rate: float,
    ) -> None:
        """
        Initialise Population class.

        _extended_summary_

        Parameters
        ----------
        population_limit : int
            Maximum number of individuals allowed in a single generation.
        crossover_rate : float
            Rate of crossover.
        """
        self.population_limit: int = population_limit
        self.variable_limits: List[List[float]] = variable_limits
        self.crossover_rate: float = crossover_rate
        self.elitism_rate: float = 1 - crossover_rate
        self.mutation_rate: float = 1 / (Config.BIT_LENGTH * len(self.variable_limits))
        self.pool: List[Chromosome] = [
            Chromosome(
                randint(0, 2, Config.BIT_LENGTH * len(variable_limits)).tolist(),
                self.variable_limits,
            )
            for _ in range(population_limit)
        ]

    def evolve(self, generations: int) -> None:
        """
        Evolve the population for 'generations' count.

        Parameters
        ----------
        generations : int
            Number of generations.
        """

        def get_random_index(max_index: int = self.population_limit) -> int:
            return randint(0, max_index, dtype=int)

        def select_chromosomes_for_reproducing(
            __type: str = "tournament",
        ) -> List[Chromosome]:
            selected_population: List[Chromosome] = []
            if __type == "tournament":
                k: int = 2
                for _ in range(round(self.population_limit * self.crossover_rate)):
                    random_indices: List[int] = list(randint(0, len(self.pool), k))
                    fittest_individual: Chromosome = max(
                        self.pool[random_indices[0]], self.pool[random_indices[-1]]
                    )
                    selected_population.append(fittest_individual)
            return selected_population

        def create_new_generation() -> List[Chromosome]:
            new_generation: List[Chromosome] = []
            reproduction_pool_size: int = len(self.reproduction_pool)
            for _ in range(reproduction_pool_size // 2):
                parent_1: Chromosome = self.reproduction_pool[
                    get_random_index(reproduction_pool_size)
                ]
                parent_2: Chromosome = self.reproduction_pool[
                    get_random_index(reproduction_pool_size)
                ]
                children: Tuple[Chromosome, Chromosome] = crossover(
                    parent_1, parent_2, "multi_point"
                )
                for child in children:
                    child = mutate_chromosome(child)
                    new_generation.append(child)
            return new_generation

        def mutate_chromosome(chromosome: Chromosome) -> Chromosome:
            for bit_ in chromosome.bits_array:
                if rand() > self.mutation_rate:
                    chromosome.bits_array[bit_] = 1 - chromosome.bits_array[bit_]
            return chromosome

        def crossover(
            parent_1: Chromosome, parent_2: Chromosome, __type: str = "single_point"
        ) -> Tuple[Chromosome, Chromosome]:
            child_1: Chromosome = parent_1
            child_2: Chromosome = parent_2
            if __type == "single_point":
                crossover_point: int = randint(1, Config.BIT_LENGTH - 2)
                child_1 = Chromosome(
                    parent_1.bits_array[:crossover_point]
                    + parent_2.bits_array[crossover_point:],
                    self.variable_limits,
                )
                child_2 = Chromosome(
                    parent_2.bits_array[:crossover_point]
                    + parent_1.bits_array[crossover_point:],
                    self.variable_limits,
                )
            elif __type == "multi_point":
                crossover_points: List[int] = sorted(
                    list(randint(1, Config.BIT_LENGTH, 2))
                )
                child_1 = Chromosome(
                    parent_1.bits_array[: crossover_points[0]]
                    + parent_2.bits_array[crossover_points[0] : crossover_points[-1]]
                    + parent_1.bits_array[crossover_points[-1] :],
                    variable_limits=self.variable_limits,
                )
                child_2 = Chromosome(
                    parent_2.bits_array[: crossover_points[0]]
                    + parent_1.bits_array[crossover_points[0] : crossover_points[-1]]
                    + parent_2.bits_array[crossover_points[-1] :],
                    variable_limits=self.variable_limits,
                )
            return child_1, child_2

        def select_elite_chromosomes() -> List[Chromosome]:
            elite_chromosomes_limit: int = round(
                self.elitism_rate * self.population_limit
            )
            old_pool_copy: List[Chromosome] = self.pool.copy()
            old_pool_copy.sort(reverse=True)
            return old_pool_copy[:elite_chromosomes_limit]

        random_index: int = get_random_index()
        __best_individual: Chromosome = self.pool[random_index]
        __best_individual_fitness_score: float = __best_individual.fitness_score

        for _ in range(generations):
            for chromosome in self.pool:
                if chromosome.fitness_score > __best_individual_fitness_score:
                    __best_individual = chromosome
                    __best_individual_fitness_score = chromosome.fitness_score
            self.reproduction_pool: List[
                Chromosome
            ] = select_chromosomes_for_reproducing()
            new_pool: List[Chromosome] = create_new_generation()
            self.pool = new_pool + select_elite_chromosomes()
            self.__get_fittest_chromosome()
            print(f"Gen. {_}. {self.fittest_chromosome}")

    def __get_fittest_chromosome(self) -> None:
        """Get fittest individual in the pool."""
        self.fittest_chromosome: Chromosome = max(
            self.pool, key=lambda x: x.fitness_score
        )

    def get_optimal_values(self) -> List[float]:
        """Optimal values of the variables."""
        return self.fittest_chromosome.phenotype_array
