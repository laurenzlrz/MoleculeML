from abc import ABC, abstractmethod


class GeometryCalculator(ABC):
    """
    Abstract base class for geometry calculations.

    Attributes:
        unit (str): The unit of measurement.
        key (str): The key identifier for the calculator.
    """

    def __init__(self, unit, key):
        """
        Initializes the GeometryCalculator with a unit and key.

        Args:
            unit (str): The unit of measurement.
            key (str): The key identifier for the calculator.
        """
        self.key = key
        self.unit = unit

    @abstractmethod
    def calculate(self, geometries, elements):
        """
        Abstract method to perform geometry calculations.

        Args:
            geometries (list): A list of geometries to be calculated.
            elements (list): A list of elements involved in the calculation.

        Returns:
            The result of the geometry calculation.
        """
        pass

    def get_unit(self):
        """
        Gets the unit of measurement.

        Returns:
            str: The unit of measurement.
        """
        return self.unit

    def get_key(self):
        """
        Gets the key identifier for the calculator.

        Returns:
            str: The key identifier for the calculator.
        """
        return self.key
