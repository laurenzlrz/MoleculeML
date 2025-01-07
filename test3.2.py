import faulthandler
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import schnetpack.transform as trn
from tblite.ase import TBLite

from src.general.Property import Property
from src.training.GeometrySchnetDB import GeometrySchnetDB
from src.functional_data.EnergyCalculator import EnergyCalculator
from src.functional_data.GeometryData import GeometryData
from src.data_origins.MD17DataLoader import MD17Dataloader
from src.training.SchnetNN import test_train


def print_db(dataset):
    print('Number of reference calculations:', len(dataset.dataset))
    print('Number of train data:', len(dataset.train_dataset))
    print('Number of validation data:', len(dataset.val_dataset))
    print('Number of test data:', len(dataset.test_dataset))
    print("\n\nAvailable properties:")

    for p in dataset.available_properties:
        print(p)

    print("\nMolecules:")
    print(dataset[0])
    print(dataset)

    count = 0
    for molecule in dataset:
        count += 1
        print(f"{count}\n")
        # print(molecule)
        for k, v in molecule.items():
            print(k, v.shape)
        if count > 0:
            break


if __name__ == "__man__":
    faulthandler.enable()
    MD17_loader = MD17Dataloader()
    MD17_loader.set_data_to_load()
    MD17_loader.set_random_split(99000, 10)

    MD17_loader.load_molecules()

    geometry_calculator = EnergyCalculator(TBLite(method="GFN2-xTB"))
    molecule = MD17_loader.get_molecule('aspirin')
    molecule_geometry_data = GeometryData(molecule, [Property.TOTAL_ENERGY, Property.OLD_ENERGIES])

    # molecule_geometry_data.perform_calculations(geometry_calculator)
    geometrySchnetDBHandler = GeometrySchnetDB.create_new("Geometry4", [molecule_geometry_data], overwrite_db=True)
    geometrySchnetDBHandler.create_schnet_module()
    print(geometrySchnetDBHandler)


if __name__ == "__main__":
    faulthandler.enable()
    schnet = GeometrySchnetDB.load_existing("Geometry4")
    data_module = schnet.create_schnet_module(transforms=[
        trn.ASENeighborList(cutoff=5.),
        trn.RemoveOffsets(Property.TOTAL_ENERGY.value, remove_mean=True, remove_atomrefs=False),
        trn.CastTo32()
    ])
    print(schnet)
    test_train(data_module)
