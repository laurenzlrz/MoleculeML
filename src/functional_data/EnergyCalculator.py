from ase import Atoms

import numpy as np

from src.general.Property import Property
from src.general.Units import Units
from src.functional_data.GeometryCalculator import GeometryCalculator

ENERGY_PROPERTY = Property.TOTAL_ENERGY
ENERGY_UNIT = Units.EV


class EnergyCalculator(GeometryCalculator):

    def __init__(self, calculation_method):
        self.calculation_method = calculation_method
        super().__init__(ENERGY_UNIT, ENERGY_PROPERTY)

    def calculate(self, geometries, elements):
        energy_calculation = lambda x: self.do_energy_calculation(x, elements)
        result = np.array([energy_calculation(geometries[i]) for i in range(geometries.shape[0])])
        return result

    def do_energy_calculation(self, geometry, elements):
        molecule = Atoms(numbers=elements, positions=geometry)
        molecule.calc = self.calculation_method
        energy = molecule.get_total_energy()
        return energy
