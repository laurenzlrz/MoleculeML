import numpy as np
import pandas as pd
from numpy.lib import recfunctions as rfn

from Utility import generate_indices


class MD17Molecule:
    def __init__(self, name, molecule_npy_arrays_dict):
        self.name = name

        self.npy_arrays_dict = molecule_npy_arrays_dict

        self.flattened_npy_dtypes_dict = None
        self.flattened_npy_arrays_dict = None

        self.flattenArraysAndNames()

        self.numpy2D_structured = None
        self.pd_dataframe = None

    def __str__(self):
        file_string = f'Molecule: {self.name}\n'
        for molecule_npy_key, molecule_npy_array in self.npy_arrays_dict.items():

            file_string += (f'Array: {molecule_npy_key}, '
                            f'Form: {molecule_npy_array.shape}, '
                            f'dtype: {molecule_npy_array.dtype}\n')
        return file_string

    def flattenArraysAndNames(self):
        self.flattened_npy_arrays_dict = {}
        self.flattened_npy_dtypes_dict = {}

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

    def load2DStructuredNumpy(self):
        npy_flattened_array_list = []
        npy_flattened_dtype_list = []

        for key, value in self.flattened_npy_arrays_dict.items():
            npy_flattened_array_list.append(value)
            npy_flattened_dtype_list.extend(self.flattened_npy_dtypes_dict[key])

        self.numpy2D_structured = rfn.unstructured_to_structured(np.hstack(self.flattened_npy_arrays_dict),
                                                                 dtype=npy_flattened_dtype_list)

    def loadDataFrame(self):
        dataframes = []
        for (array, dtypes) in zip(self.flattened_npy_arrays_dict, self.flattened_npy_dtypes_dict):
            dtypes = [dtype[0] for dtype in dtypes]
            dataframes.append(pd.DataFrame(array, columns=dtypes))

        self.pd_dataframe = pd.concat(dataframes, axis=1)
