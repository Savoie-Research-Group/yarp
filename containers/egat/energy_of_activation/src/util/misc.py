"""
Miscellaneous utility functions for the yarp package.
(I hate this, but we'll use this for now - ERM)
"""

import numpy as np

def prepare_list(x):
    """
    Many of the functions in the yarp library expect a list of objects, although in many use cases the user may
    only supply a single object. It is awkward behavior to ask the user to supply a single object inside of a list
    so this function handles performs the list wrapping for the user when functions expect a list and instead
    are passed a single instance of the object. 

    Parameters
    ----------
    x: list or object
       In a typical use case, this is an input to a function or method in the yarp package. 

    Returns
    -------
    x: list
       The purpose of this function is to ensure that the supplied object is wrapped in a list if it 
       is not already a list.
    """
    if isinstance(x, (tuple, list)):
        return x
    else:
        return [x]


def merge_arrays(list_of_arrays):
    """
    This function takes a list of arrays and concatenates them along the diagonal of a new array, 
    such that each array in the original list occupies a subblock of the new array and the off-diagonal
    elements where the sub-blocks overlap are zero. For example, suppose the input to this function is 
    a list of three arrays. The first array in `list_of_arrays` is of size 2x2, the second array is of
    size 3x3, and the third array is of size 4x4. `merge_arrays` will return a new array of size 9x9 
    (i.e., 2+3+4) with the elements of the 2x2 array in the [0,0],[0,1],[1,0], and [1,1] positions, 
    the elements of the 3x3 array in postitions [2,2],[2,3],[2,4],[3,2],[3,3],[3,4], etc.  

    Parameters
    ----------
    list_of_arrays: list of arrays
                    A list of square numpy arrays.

    Returns
    -------
    merged_array: array
                  The merged array consisting of the sub-arrays concatenated along the diagonal blocks.
    """

    # Handle singular use-case
    list_of_arrays = prepare_list(list_of_arrays)

    # Get the dimensions of each input array
    dimensions = [arr.shape[0] for arr in list_of_arrays]

    # Calculate the dimensions of the merged array
    merged_dim = sum(dimensions)

    # Initialize the merged array with zeros
    merged_array = np.zeros((merged_dim, merged_dim),
                            dtype=list_of_arrays[0].dtype)

    # Iterate through each input array and copy its elements to the merged array
    row_offset = 0
    for arr in list_of_arrays:
        size = arr.shape[0]
        merged_array[row_offset:row_offset + size,
                     row_offset:row_offset + size] = arr
        row_offset += size

    return merged_array


