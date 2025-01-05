from abc import ABC, abstractmethod
from src.general.Property import Property
from src.general.Units import Units
from numpy import ndarray


class GeometryCalculator(ABC):
    """
    Abstract base class for geometry calculations.

    Attributes:
        unit (Units): The unit of measurement.
        key (Property): The key identifier for the calculator.
    """

    def __init__(self, unit, key):
        """
        Initializes the GeometryCalculator with a unit and key.

        Args:
            unit (Units): The unit of measurement.
            key (Property): The key identifier for the calculator.
        """
        self.key = key
        self.unit = unit

    @abstractmethod
    def calculate(self, geometries, elements):
        """
        Abstract method to perform geometry calculations.

        Args:
            geometries (ndarray): A list of geometries to be calculated.
            elements (ndarray): A list of elements involved in the calculation.

        Returns:
            The result of the geometry calculation.
        """
        pass

    def get_unit(self):
        """
        Gets the unit of measurement.

        Returns:
            Unit: The unit of measurement.
        """
        return self.unit

    def get_key(self):
        """
        Gets the key identifier for the calculator.

        Returns:
            Property: The key identifier for the calculator.
        """
        return self.key
