import pandas as pd
from ase import Atoms
from tblite.ase import TBLite
from tblite.interface import Calculator
from ase.calculators.morse import MorsePotential

from src.general.props.MolProperty import MolProperty

from src.data_origins.MD17DataLoader import MD17Dataloader
from src.data_origins.MD17MoleculeData import MD17Molecule

MD17_loader = MD17Dataloader()
MD17_loader.set_data_to_load()
MD17_loader.load_split("train", 1)
MD17_loader.load_molecules()


def visualise_molecule(molecule: MD17Molecule):
    min_index = molecule.npy_arrays_dict[MolProperty.TOTAL_ENERGY].argmin()
    elements = molecule.npy_arrays_dict[MolProperty.ELEMENTS]
    min_positions = molecule.npy_arrays_dict[MolProperty.COORDINATES][min_index]

    atoms_MP = Atoms(numbers=elements, positions=min_positions)
    atoms_MP.calc = MorsePotential()
    MP_energy = atoms_MP.get_potential_energy()

    atoms_XTB = Atoms(numbers=elements, positions=min_positions)
    atoms_XTB.calc = TBLite(method="GFN2-xTB")
    XTB_energy = atoms_XTB.get_potential_energy()

    calculator = Calculator("GFN2-xTB", elements, min_positions)
    res = calculator.singlepoint()
    CALC_enery = res.dict()

    REF_energy = molecule.npy_arrays_dict[MolProperty.TOTAL_ENERGY][min_index]

    dict = {"Name" : molecule.name,
            "MP" : MP_energy, "XTB" : XTB_energy, "REF" : REF_energy, "CALC" : CALC_enery['energy'],
            "MP/REF" : MP_energy / REF_energy, "XTB/REF" : XTB_energy / REF_energy,
            "CALC/REF" : CALC_enery['energy'] / REF_energy}
    return dict

    #print(f"Estimated minimum energy: {MP_energy} Estimated maximum energy: {max_energy}")
    #print(f"Actual minimum energy: {min_ref_energy} Actual maximum energy: {max_ref_energy}")
    #print(f"factor: {MP_energy / min_ref_energy} {max_energy / max_ref_energy}")
    #print(f"factor: {min_ref_energy / MP_energy} {max_ref_energy / max_energy}")

liste = []
for key, value in MD17_loader.molecules_dict.items():
    liste.append(visualise_molecule(value))

frames = []
for dict in liste:
    print(dict)
    frames.append(pd.DataFrame(dict, index=[0]))

result = pd.concat(frames)
result.to_csv("calculated_energies.csv")

print(result)
