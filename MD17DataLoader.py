import numpy as np
from numpy.lib import recfunctions as rfn
from MD17MoleculeData import MD17Molecule

from Utility import generate_indices

MOLECULE_DIRECTORY = "data/npz_data"
DATA_SEPARATOR = "/"
DATA_PREFIX = "rmd17_"
DATA_SUFFIX = ".npz"
MOLECULE_NAMES = ["aspirin", "azobenzene", "benzene", "ethanol", "malonaldehyde",
                  "naphthalene", "paracetamol", "salicylic", "toluene", "uracil"]

MOLECULE_ATTRIBUTES = ['nuclear_charges', 'coords', 'energies', 'forces', 'old_indices', 'old_energies', 'old_forces']

"""
Dataloader is responsible to create data objects from downloaded files. Provides a dataobject in the end and can be
configured in the beginning.
"""


class MD17Dataloader:

    def __init__(self):
        self.molecule_npz_files = None
        self.molecule_attributes = None
        self.molecule_npz_arrays = None
        self.molecules = None
        self.molecule_arrays = None

    ### Loading data from the files
    def set_data_to_load(self, molecule_names, molecule_attributes):
        self.molecule_npz_files = {
            molecule_name: MOLECULE_DIRECTORY + DATA_SEPARATOR + DATA_PREFIX + molecule_name + DATA_SUFFIX
            for molecule_name in molecule_names}

        self.molecule_attributes = molecule_attributes.copy()

    def load_molecules(self):
        self.molecules = {name: MD17Molecule(name, np.load(file), self.molecule_attributes.copy())
                          for name, file in self.molecule_npz_files.items()}

    def get_data(self):
        data = np.load(MOLECULE_DIRECTORY + DATA_SEPARATOR + DATA_PREFIX + "aspirin" + DATA_SUFFIX)
        print(data.files)


MD17_loader = MD17Dataloader()
MD17_loader.set_data_to_load(MOLECULE_NAMES, MOLECULE_ATTRIBUTES)
MD17_loader.load_molecules()
print(MD17_loader.molecules["aspirin"].__str__())
MD17_loader.molecules["aspirin"].load2DStructuredNumpy()
MD17_loader.molecules["aspirin"].loadDataFrame()

