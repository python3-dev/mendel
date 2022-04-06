"""Chromosome class."""

import struct

from typing import Union, List
from config.config import Config


class Gene:
    """
    Gene class.

    Genes represent a variable in the
    objective function.
    """
    def __init__(self, var: Union[float, int]):
        """__init__ Initialise Gene class.

        Parameters
        ----------
        var : Union[float, int]
            Float or int values representing the phenotype.
        """
        self.var: float = float(var)
        self.hex_string: str = self.__hex_convert()
        self.bin_string: str = self.__bin_convert()
        self.length: int = len(self.bin_string)

    def __hex_convert(self) -> str:
        """Convert float to hex."""
        return float.hex(self.var)

    def __bin_convert(self):
        """
        Convert float to binary.
        Source: https://stackoverflow.com/questions/53538504/float-to-binary-and-binary-to-float-in-python
        """
        return format(struct.unpack('!I', struct.pack('!f', self.var))[0], f'0{Config.BIT_LENGTH}b')

    def bin_to_float(self):
        """
        Convert binary to float.
        Source: https://stackoverflow.com/questions/53538504/float-to-binary-and-binary-to-float-in-python
        """
        return struct.unpack('!f', struct.pack('!I', int(self.bin_string, 2)))[0]


class Chromosome:
    """
    Chromosome class.

    Chromosomes represents a collection of
    variables encoded as genes.
    """
    def __init__(self) -> None:
        self.genes: List[Gene] = []
        self.chromosome_string: str = ''
        self.length: int = len(self.chromosome_string)

    def add_gene(self, gene: Gene) -> None:
        self.genes.append(gene)

    def generate_chromosome_string(self) -> None:
        for gene in self.genes:
            self.chromosome_string += gene.bin_string
        self.__determine_chromosome_length()

    def __determine_chromosome_length(self) -> None:
        self.length = len(self.chromosome_string)


class Environment:
    """
    Environment class.

    Chromosomes interact with each other
    in Environment class.
    """
    def __init__(self) -> None:
        self.population: List[Chromosome] = []
        self.max_population: int = 100
        self.mutation: float = 0.1
        self.elitism: bool = True
        self.elite_preservation: float = 0.1



if __name__ == '__main__':
    x_2 = Gene(1)
    print(x_2.bin_string)
    print(x_2.bin_to_float())
    x_2 = Gene(-1)
    print(x_2.bin_string)
    print(x_2.bin_to_float())
