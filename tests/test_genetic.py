"""Unit tests for models.genetic.Chromosome model."""

import unittest

from config.config import Config
from exceptions.exceptions import InsufficientBitArrayLength
from models.genetic import Chromosome


class TestChromosomeEightBit(unittest.TestCase):
    """Unittest cases for models.genetic.Chromosome class for 8 bit length."""

    def setUp(self) -> None:
        """Unittest setup."""
        Config.BIT_LENGTH = 8
        self.variable_limits_1: list[list[float]] = [[0.0, 1.0]]
        self.variable_limits_2: list[list[float]] = [[0.0, 1.0], [0.0, 1.0]]
        self.test_array_8_0: list[int] = [0] * Config.BIT_LENGTH
        self.test_array_8_1: list[int] = [1] * Config.BIT_LENGTH
        self.test_array_8_0_1: list[int] = [0, 1] * (Config.BIT_LENGTH // 2)
        self.test_array_8_1_0: list[int] = [1, 0] * (Config.BIT_LENGTH // 2)

    def test_number_of_genes_1_0(self) -> None:
        """Unittest case."""
        chromosome: Chromosome = Chromosome(self.test_array_8_0, self.variable_limits_1)
        self.assertEqual(len(chromosome.genes), len(self.variable_limits_1))

    def test_number_of_genes_2_0(self) -> None:
        """Unittest case."""
        self.assertRaises(
            InsufficientBitArrayLength,
            Chromosome,
            self.test_array_8_0,
            self.variable_limits_2,
        )

    def test_number_of_genes_1_1(self) -> None:
        """Unittest case."""
        chromosome: Chromosome = Chromosome(self.test_array_8_1, self.variable_limits_1)
        self.assertEqual(len(chromosome.genes), len(self.variable_limits_1))

    def test_number_of_genes_2_1(self) -> None:
        """Unittest case."""
        self.assertRaises(
            InsufficientBitArrayLength,
            Chromosome,
            self.test_array_8_1,
            self.variable_limits_2,
        )

    def test_number_of_genes_1_0_1(self) -> None:
        """Unittest case."""
        chromosome: Chromosome = Chromosome(
            self.test_array_8_0_1, self.variable_limits_1
        )
        self.assertEqual(len(chromosome.genes), len(self.variable_limits_1))

    def test_number_of_genes_2_0_1(self) -> None:
        """Unittest case."""
        self.assertRaises(
            InsufficientBitArrayLength,
            Chromosome,
            self.test_array_8_0_1,
            self.variable_limits_2,
        )

    def test_number_of_genes_1_1_0(self) -> None:
        """Unittest case."""
        chromosome: Chromosome = Chromosome(
            self.test_array_8_1_0, self.variable_limits_1
        )
        self.assertEqual(len(chromosome.genes), len(self.variable_limits_1))

    def test_number_of_genes_2_1_0(self) -> None:
        """Unittest case."""
        self.assertRaises(
            InsufficientBitArrayLength,
            Chromosome,
            self.test_array_8_1_0,
            self.variable_limits_2,
        )


class TestChromosomeSixteenBit(unittest.TestCase):
    """Unittest cases for models.genetic.Chromosome class for 16 bit length."""

    def setUp(self) -> None:
        """Fixtures."""
        Config.BIT_LENGTH = 16
        self.variable_limits: list[list[int]] = [[0, 1]]
        self.test_array_16_0: list[int] = [0] * Config.BIT_LENGTH
        self.test_array_16_1: list[int] = [1] * Config.BIT_LENGTH
        self.test_array_16_0_1: list[int] = [0, 1] * (Config.BIT_LENGTH // 2)
        self.test_array_16_1_0: list[int] = [1, 0] * (Config.BIT_LENGTH // 2)
