import numpy as np
import pandas as pd
from numpy.lib import recfunctions as rfn

from src.data_origins.AbstractMoleculeData import AbstractMoleculeData
from src.general.Property import Property
from src.general.Utility import generate_indices


class MD17Molecule(AbstractMoleculeData):
    """
    A class used to represent a molecule and manage its data.

    Attributes:
        name (str): The name of the molecule.
        npy_arrays_dict (dict): Dictionary of numpy arrays containing molecule data.
        flattened_npy_dtypes_dict (dict): Dictionary of flattened numpy array data types.
        flattened_npy_arrays_dict (dict): Dictionary of flattened numpy arrays.
        numpy2D_structured (numpy.ndarray): 2D structured numpy array.
        pd_dataframe (pandas.DataFrame): DataFrame containing the molecule data.
    """

    def __init__(self, name, molecule_npy_arrays_dict, units_dict):
        """
        Initializes the MD17Molecule with a name and a dictionary of numpy arrays.
        Numpy2D_structured and pd_dataframe are set to None and have to be loaded separately if needed.

        Args:
            name (str): The name of the molecule.
            molecule_npy_arrays_dict (dict): Dictionary of numpy arrays containing molecule data.
        """
        self.name = name
        self.npy_arrays_dict = molecule_npy_arrays_dict
        self.units = units_dict

        self.flattened_npy_dtypes_dict = None
        self.flattened_npy_arrays_dict = None
        self.numpy2D_structured = None
        self.pd_dataframe = None

    def __str__(self):
        """
        Returns a string representation of the molecule and its arrays.

        Returns:
            str: A string describing the molecule and its arrays.
        """
        file_string = f'Molecule: {self.name}\n'
        for molecule_npy_key, molecule_npy_array in self.npy_arrays_dict.items():
            file_string += (f'Array: {molecule_npy_key}, '
                            f'Form: {molecule_npy_array.shape}, '
                            f'dtype: {molecule_npy_array.dtype}\n')
        return file_string

    def flattenArraysAndNames(self):
        """
        Flattens the numpy arrays and generates new names for the flattened arrays.
        Index combinations are generated for each array and used to create new names.
        """
        self.flattened_npy_arrays_dict = {}
        self.flattened_npy_dtypes_dict = {}

        # TODO more abstract, so that the flattening is done in a more abstract way and
        #  can be substantiated in concrete data subclasses
        for molecule_npy_key, molecule_npy_array in self.npy_arrays_dict.items():
            if molecule_npy_key == 'nuclear_charges':
                continue

            idx_combs = generate_indices(molecule_npy_array.shape)
            flattened = molecule_npy_array.reshape(molecule_npy_array.shape[0], -1)
            molecule_npy_dtype = []

            for idx in range(flattened.shape[-1]):
                name = f'{molecule_npy_key}_' + '_'.join(map(str, idx_combs[idx]))
                molecule_npy_dtype.append((name, molecule_npy_array.dtype))

            self.flattened_npy_arrays_dict[molecule_npy_key] = flattened
            self.flattened_npy_dtypes_dict[molecule_npy_key] = molecule_npy_dtype

    def to2DStructuredNumpy(self):
        """
        Loads the flattened numpy arrays into a 2D structured numpy array.
        """
        npy_flattened_array_list = []
        npy_flattened_dtype_list = []

        self.flattenArraysAndNames()

        for key, value in self.flattened_npy_arrays_dict.items():
            npy_flattened_array_list.append(value)
            npy_flattened_dtype_list.extend(self.flattened_npy_dtypes_dict[key])

        self.numpy2D_structured = rfn.unstructured_to_structured(np.hstack(self.flattened_npy_arrays_dict),
                                                                 dtype=npy_flattened_dtype_list)

    def toDataFrame(self):
        """
        Loads the flattened numpy arrays into a pandas DataFrame.
        """
        dataframes = []
        for (array, dtypes) in zip(self.flattened_npy_arrays_dict, self.flattened_npy_dtypes_dict):
            dtypes = [dtype[0] for dtype in dtypes]
            dataframes.append(pd.DataFrame(array, columns=dtypes))

        self.pd_dataframe = pd.concat(dataframes, axis=1)

    def toArrayDict(self):
        return self.npy_arrays_dict.copy()

    def addAttribute(self, array, attribute_key, attribute_unit):
        self.npy_arrays_dict[attribute_key] = array
        self.units[attribute_key] = attribute_unit
        # TODO add size check

    def getAttribute(self, attribute: Property):
        return self.npy_arrays_dict[attribute]

    def getUnit(self, attribute: Property):
        return self.units[attribute]
