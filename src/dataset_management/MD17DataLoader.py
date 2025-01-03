import numpy as np

from src.dataset_management.MD17MoleculeData import MD17Molecule

# Path constants
PATH_SEPARATOR = "/"

MOLECULE_DIRECTORY = "data/npz_data"
MOLECULE_DATA_FORMAT_STRING = "rmd17_{molecule_name}.npz"

SPLIT_DIRECTORY = "data/splits"
SPLIT_DATA_FORMAT_STRING = "index_{usetype}_0{number}.csv"

# Molecule constants
MOLECULE_NAMES = ["aspirin", "azobenzene", "benzene", "ethanol", "malonaldehyde",
                  "naphthalene", "paracetamol", "salicylic", "toluene", "uracil"]

MOLECULE_ATTRIBUTES = ['nuclear_charges', 'coords', 'energies', 'forces', 'old_indices', 'old_energies', 'old_forces']

"""
Dataloader is responsible to create data objects from downloaded files. Provides a dataobject in the end and can be
configured in the beginning.
"""


class MD17Dataloader:
    """
    A class used to load and manage MD17 molecule data.

    Attributes:
        molecules_npz_files (dict): Dictionary of molecule names and their corresponding npz file paths.
        molecule_attributes (list): List of molecule attributes to be loaded.
        molecules_dict (dict): Dictionary of molecule names and their corresponding MD17Molecule objects.
        split (numpy.ndarray): Array of indices used to split the data.
        max_split (int): Maximum index value in the split array.
    """

    def __init__(self):
        """
        Initializes the MD17Dataloader with default values.
        """
        self.molecules_npz_files = None
        self.molecule_attributes = None
        self.molecules_dict = None
        self.split = None
        self.max_split = None

    def set_data_to_load(self, molecule_names=MOLECULE_NAMES, molecule_attributes=MOLECULE_ATTRIBUTES):
        """
        Sets the data to be loaded by specifying molecule names and attributes.

        Args:
            molecule_names (list): List of molecule names to be loaded.
            molecule_attributes (list): List of attributes to be loaded for each molecule.
        """
        self.molecules_npz_files = {
            molecule_name: MOLECULE_DIRECTORY + PATH_SEPARATOR +
                           MOLECULE_DATA_FORMAT_STRING.format(molecule_name=molecule_name)
            for molecule_name in molecule_names}

        self.molecule_attributes = molecule_attributes.copy()

    def load_split(self, split_name, split_number):
        """
        Loads the split indices from a CSV file.

        Args:
            split_name (str): The type of split (e.g., 'train', 'test').
            split_number (int): The split number to be loaded.
        """
        split_path = (SPLIT_DIRECTORY + PATH_SEPARATOR +
                      SPLIT_DATA_FORMAT_STRING.format(usetype=split_name, number=split_number))
        self.split = np.loadtxt(split_path, delimiter=',', dtype=int)
        self.max_split = np.max(self.split)

    def set_random_split(self, selection_range, split_size):
        """
        Sets a random split of the given size.

        Args:
            split_size (int): The size of the random split.
        """
        self.split = np.random.choice(selection_range, split_size, replace=False)
        self.max_split = np.max(self.split)

    def load_molecules(self):
        """
        Loads the molecule data from npz files and applies the split.
        """
        self.molecules_dict = {}
        for molecule_name, file in self.molecules_npz_files.items():
            molecule_npz_array = np.load(file)

            # Copies each selected (listed in molecule_attributes) array in the molecule_npz_array to a new dictionary
            # The split is applied to each array
            molecule_npy_arrays_dict = {molecule_attribute: self.apply_sample(molecule_npz_array[molecule_attribute])
                                        for molecule_attribute in self.molecule_attributes}

            self.molecules_dict[molecule_name] = MD17Molecule(molecule_name, molecule_npy_arrays_dict,
                                                              molecule_npz_array)

    def apply_sample(self, molecule_npy_array):
        """
        Applies the split to the given numpy array.

        Args:
            molecule_npy_array (numpy.ndarray): The numpy array to which the split will be applied.

        Returns:
            numpy.ndarray: The filtered numpy array after applying the split.
        """
        if self.split is None or len(molecule_npy_array) < self.max_split:
            return molecule_npy_array

        return molecule_npy_array[self.split]

    def get_molecule(self, molecule_name):
        """
        Returns the molecule with the given name.

        Args:
            molecule_name (str): The name of the molecule to be returned.

        Returns:
            MD17Molecule: The molecule object with the given name.
        """
        return self.molecules_dict[molecule_name]



