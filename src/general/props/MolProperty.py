from enum import Enum


class MolProperty(Enum):
    TOTAL_ENERGY = 'total_energy'
    TOTAL_ENERGY_TRUTH = 'total_energy_truth'
    TOTAL_ENERGY_CALCULATED = 'calculated_total_energy'
    TOTAL_ENERGY_DIFFERENCE = 'energy_difference'
    POTENTIAL_ENERGY = 'potential_energy'
    KINETIC_ENERGY = 'kinetic_energy'
    FORCES = 'forces'
    DIPOLE = 'dipole'
    ELEMENTS = 'elements'
    COORDINATES = 'coordinates'
    OLD_ENERGIES = 'old_energies'
    OLD_FORCES = 'old_forces'
    OLD_ELEMENTS = 'old_elements',
    BASE_ENERGY = 'base_energy'
    BASE_ENERGY_DIFFERENCE = 'base_energy_difference'
