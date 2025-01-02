"""
Utility functions for the main program.

This module contains functions that are used as utilities in the main program.
"""

from itertools import product


def generate_indices(shape):
    """
    Generate all possible index combinations for a multidimensional shape.
    This function produces tuples representing every possible coordinate in the array's shape.

    Args:
        shape (tuple): A tuple representing the dimensions of the array.

    Returns:
        list: A list of tuples, where each tuple represents a coordinate in the array's shape.

    Example:
        If shape = (3, 2), the ranges are [0, 1, 2] and [0, 1].
        The Cartesian product will produce:
        (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)
        This means that for a (3, 2) shape, we will generate 6 combinations of indices.
    """
    return list(product(*[range(dim) for dim in shape]))
