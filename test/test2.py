from tblite.interface import Calculator
import tblite.ase
import numpy as np
from ase import Atoms

calc = Calculator(
    method="GFN2-xTB",
    numbers=np.array([14, 1, 1, 1, 1]),
    positions=np.array([
        [0.000000000000, 0.000000000000, 0.000000000000],
        [1.617683897558, 1.617683897558, -1.617683897558],
        [-1.617683897558, -1.617683897558, -1.617683897558],
        [1.617683897558, -1.617683897558, 1.617683897558],
        [-1.617683897558, 1.617683897558, 1.617683897558],
    ]),
)

numbers = np.array([14, 1, 1, 1, 1])
positions = np.array([
                  [0.000000000000, 0.000000000000, 0.000000000000],
                  [1.617683897558, 1.617683897558, -1.617683897558],
                  [-1.617683897558, -1.617683897558, -1.617683897558],
                  [1.617683897558, -1.617683897558, 1.617683897558],
                  [-1.617683897558, 1.617683897558, 1.617683897558]])


atoms = Atoms(numbers=numbers,
              positions=positions)
atoms.calc = tblite.ase.TBLite(method="GFN2-xTB")
total_energy = atoms.get_potential_energy()

print("\n \n \n")

res = calc.singlepoint()
print(total_energy)

numpy = np.load("../data/md17_data/rmd17_benzene.npz")["energies"]
print(numpy)
