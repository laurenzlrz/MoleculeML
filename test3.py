import faulthandler

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from src.training.GeometrySchnetDB import GeometrySchnetDB
from tblite.ase import TBLite
from src.functional_data.EnergyCalculator import EnergyCalculator
from src.functional_data.GeometryData import GeometryData
from src.data_origins.MD17DataLoader import MD17Dataloader
from src.training.SchnetNN import test_train
from schnetpack.transform import RemoveOffsets
from src.general.MolProperty import MolProperty

def print_db(dataset, data_module):
    print("\n\nAvailable properties:\n")

    for p in data_module.available_properties:
        print(p)

    print("Molecules:\n")
    print(data_module[0])
    print(data_module)

    count = 0
    for molecule in data_module:
        count += 1
        print(count)
        print(molecule)
        for k, v in molecule.items():
            print(k, v.shape)

if __name__ == "__main__":
    faulthandler.enable()
    MD17_loader = MD17Dataloader()
    MD17_loader.set_data_to_load()
    MD17_loader.set_random_split(99000, 10)

    MD17_loader.load_molecules()

    geometry_calculator = EnergyCalculator(TBLite(method="GFN2-xTB"))
    molecule = MD17_loader.get_molecule('aspirin')
    molecule_geometry_data = GeometryData(molecule)

    molecule_geometry_data.perform_calculations(geometry_calculator)
    geometrySchnetDBHandler = GeometrySchnetDB.create_new("Geometry", [molecule_geometry_data])
    geometrySchnetDB = geometrySchnetDBHandler.get_schnet_db()

    data_module = geometrySchnetDBHandler.create_schnet_module(pin_memory=False,
                                                               transforms=[RemoveOffsets(MolProperty.TOTAL_ENERGY.value, remove_mean=True,
                                                                                         remove_atomrefs=True)])
    print("\n\n TRAINING \n\n")
    test_train(geometrySchnetDB, data_module)
