from ase import Atoms
from tblite.ase import TBLite
from tblite.interface import Calculator

from src.data_origins.MD17DataLoader import MD17Dataloader
from src.data_origins.MD17MoleculeData import MD17Molecule

MD17_loader = MD17Dataloader()
MD17_loader.set_data_to_load()
MD17_loader.load_split("train", 1)
MD17_loader.load_molecules()
aspirin = MD17_loader.molecules_dict['aspirin']


def visualise_molecule(molecule: MD17Molecule):
    min_index = molecule.npy_arrays_dict['energies'].argmin()
    max_index = molecule.npy_arrays_dict['energies'].argmax()
    elements = molecule.npy_arrays_dict['nuclear_charges']
    min_positions = molecule.npy_arrays_dict['coords'][min_index]
    max_positions = molecule.npy_arrays_dict['coords'][max_index]

    min_atom = Atoms(numbers=elements, positions=min_positions)
    min_atom.calc = TBLite(method="GFN2-xTB")
    min_energy = min_atom.get_total_energy()

    max_atom = Atoms(numbers=elements, positions=max_positions)
    max_atom.calc = TBLite(method="GFN2-xTB")
    max_energy = max_atom.get_total_energy()

    calculator = Calculator("GFN2-xTB", elements, min_positions)
    res = calculator.singlepoint()
    res_energy = res.get('energies')
    print(res_energy)

    min_ref_energy = molecule.npy_arrays_dict['old_energies'][min_index]
    max_ref_energy = molecule.npy_arrays_dict['old_energies'][max_index]

    print(f"Estimated minimum energy: {min_energy} Estimated maximum energy: {max_energy}")
    print(f"Actual minimum energy: {min_ref_energy} Actual maximum energy: {max_ref_energy}")
    print(f"factor: {min_energy / min_ref_energy} {max_energy / max_ref_energy}")
    print(f"factor: {min_ref_energy / min_energy} {max_ref_energy / max_energy}")

visualise_molecule(MD17_loader.molecules_dict['benzene'])

#print((4200 / (6.6 * 10**23)) / (1.6 * (10 ** (-19))))