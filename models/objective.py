"""Definition of objective function."""


class Objective:
    """Objective class."""

    @staticmethod
    def objective(variable_phenotype_array: list[float]) -> float:
        """
        Objective function.

        Parameters
        ----------
        variable_phenotype_array : list[float]
            List of phenotypes.

        Returns
        -------
        float
            Objective function value.
        """
        return abs(1 / ((variable_phenotype_array[0] ** 2) - 2))
