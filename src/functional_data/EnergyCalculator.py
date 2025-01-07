from ase import Atoms
import numpy as np
from src.general.Property import Property
from src.general.Units import Units
from src.functional_data.GeometryCalculator import GeometryCalculator

ENERGY_PROPERTY = Property.TOTAL_ENERGY
ENERGY_UNIT = Units.EV


class EnergyCalculator(GeometryCalculator):
    """
    A class used to calculate the energy of molecular geometries.

    Attributes:
        calculation_method: The method used to calculate the energy.
    """

    def __init__(self, calculation_method):
        """
        Initializes the EnergyCalculator with a calculation method.

        Args:
            calculation_method: The method used to calculate the energy.
        """
        self.calculation_method = calculation_method
        super().__init__(ENERGY_UNIT, ENERGY_PROPERTY)

    def calculate(self, geometries, elements):
        """
        Calculates the energy for a list of geometries.

        Args:
            geometries (numpy.ndarray): A 2D array where each row represents a geometry.
            elements (list): A list of elements corresponding to the atoms in the geometries.

        Returns:
            numpy.ndarray: An array of calculated energies for each geometry.
        """
        energy_calculation = lambda x: self.do_energy_calculation(x, elements)

        #result = []
        #for i in range(geometries.shape[0]):
        #    print(f"Calculating energy for geometry {i + 1}...")
        #    geometry = (geometries[i])
        #    total_energy = energy_calculation(geometry)
        #    result.append(total_energy)
        #
        result = np.array([energy_calculation(geometries[i]) for i in range(geometries.shape[0])])
        return result

    def do_energy_calculation(self, geometry, elements):
        """
        Performs the energy calculation for a single geometry.

        Args:
            geometry (numpy.ndarray): A 2D array representing the positions of atoms in the geometry.
            elements (list): A list of elements corresponding to the atoms in the geometry.

        Returns:
            float: The calculated energy for the geometry.
        """
        molecule = Atoms(numbers=elements, positions=geometry)
        molecule.calc = self.calculation_method
        energy = molecule.get_total_energy()
        return energy
