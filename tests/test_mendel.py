from mendel import __version__
from mendel.app import Gene


def test_version():
    assert __version__ == '0.1.0'


def test_gene_1():
    gene = Gene(1)
    assert gene.bin_string == '111111100000000000000000000000'
