import numpy as np
import pandas as pd
from numpy.lib import recfunctions as rfn

from itertools import chain
from Utility import generate_indices


class MD17Molecule:
    def __init__(self, name, molecule_npz_array, molecule_attributes):
        self.name = name

        #TODO Remove npz_array because some attributes are not used
        self.npz_array = molecule_npz_array
        self.molecule_attributes = molecule_attributes

        self.dtypes = None
        self.flattened_arrays = None

        self.flattenArraysAndNames()

        self.numpy2D_structured = None
        self.pd_dataframe = None

    def __str__(self):
        file_string = f'Molecule: {self.name}\nAttributes: {self.molecule_attributes}\n'
        for molecule_npy_key in self.npz_array.files:

            file_string += (f'Array: {molecule_npy_key}, '
                            f'Form: {self.npz_array[molecule_npy_key].shape}, '
                            f'dtype: {self.npz_array[molecule_npy_key].dtype}\n')
        return file_string

    def flattenArraysAndNames(self):
        self.flattened_arrays = []
        self.dtypes = []

        for molecule_npy_key in self.npz_array.files:

            if molecule_npy_key == 'nuclear_charges':
                continue

            if molecule_npy_key not in self.molecule_attributes:
                continue

            molecule_npy_array = self.npz_array[molecule_npy_key]
            idx_combs = generate_indices(molecule_npy_array.shape)
            flattened = molecule_npy_array.reshape(molecule_npy_array.shape[0], -1)
            molecule_npy_dtype = []

            for idx in range(flattened.shape[-1]):
                name = f'{molecule_npy_key}_' + '_'.join(map(str, idx_combs[idx]))
                molecule_npy_dtype.append((name, molecule_npy_array.dtype))

            self.flattened_arrays.append(flattened)
            self.dtypes.append(molecule_npy_dtype)

    def load2DStructuredNumpy(self):
        dtypes_flattened = list(chain(*self.dtypes))

        self.numpy2D_structured = rfn.unstructured_to_structured(np.hstack(self.flattened_arrays),
                                                                 dtype=dtypes_flattened)

    def loadDataFrame(self):
        dataframes = []
        for (array, dtypes) in zip(self.flattened_arrays, self.dtypes):
            dtypes = [dtype[0] for dtype in dtypes]
            dataframes.append(pd.DataFrame(array, columns=dtypes))

        self.pd_dataframe = pd.concat(dataframes, axis=1)
