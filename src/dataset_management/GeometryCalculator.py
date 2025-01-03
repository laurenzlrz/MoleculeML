from abc import ABC, abstractmethod
from ase import Atoms

import numpy as np

class GeometryCalculator(ABC):

    @abstractmethod
    def calculate(self, geometries, elements):
        pass

class EnergyCalculator(GeometryCalculator):

    def __init__(self):
        self.calculation_method = None

    def calculate(self, geometries, elements):
        energy_calculation = lambda x: self.do_energy_calculation(x, elements)
        result = np.array([energy_calculation(geometries[i]) for i in range(geometries.shape[0])])
        return result

    def do_energy_calculation(self, geometry, elements):
        molecule = Atoms(numbers=elements, positions=geometry)
        molecule.calc = self.calculation_method
        energy = molecule.get_total_energy()
        return energy

    def set_calculation_method(self, calculation_method):
        self.calculation_method = calculation_method