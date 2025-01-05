from enum import Enum


class Property(Enum):
    TOTAL_ENERGY = 'total_energy',
    POTENTIAL_ENERGY = 'potential_energy',
    KINETIC_ENERGY = 'kinetic_energy',
    FORCES = 'forces',
    DIPOLE = 'dipole',
    ELEMENTS = 'elements',
    COORDINATES = 'coordinates',
    OLD_ENERGIES = 'old_energies',
    OLD_FORCES = 'old_forces',
    OLD_ELEMENTS = 'old_elements',
