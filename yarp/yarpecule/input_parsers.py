"""
Convenience functions for parsing molecular information from a variety of input formats.
"""

import numpy as np


def xyz_parse(xyz, read_types=False, multiple=False):
    """
    Simple wrapper function for grabbing the coordinates and elements from an xyz file.

    Parameters
    ----------
    xyz : filename
          This is the xyz file being parsed.

    read_types : bool, default=False
                 If this is set to `True` then the function will try and grab optional data from a fourth column of
                 the file (i.e., a column after the x y and z information).

    multiple : bool, default=False
                Allows for multiple coordinates/elements to be read from the same XYZ file.
                If set to False, only the final set of coordinates and elements will be returned.

    Returns
    -------
    elements : list
               A list with the element labels indexed to the geometry.

    geo : array
          An nx3 numpy array holding the cartesian coordinates for the user supplied geometry.

    atom_types : list (optional)
                 If the `read_types=True` option is supplied then an optional third list is returned. 
    """

    elements = []
    geo = []

    if len(open(xyz, 'r+').readlines()) == 0:
        # this seems like it should be a runtime error - ERM
        print('An empty file: {xyz}')
        return elements, geo

    # Commands for reading only the coordinates and the elements
    if read_types is False:

        # Iterate over the remainder of contents and read the
        # geometry and elements into variable. Note that the
        # first two lines are considered a header
        with open(xyz, 'r') as f:
            for lc, lines in enumerate(f):
                fields = lines.split()

                # Parse header
                if lc == 0 or (len(fields) == 1 and fields[0].isdigit()):
                    if len(fields) < 1:
                        print(
                            "ERROR in xyz_parse: {} is missing atom number information".format(xyz))
                        quit()
                    else:
                        N_atoms = int(fields[0])
                        Elements = ["X"]*N_atoms
                        Geometry = np.zeros([N_atoms, 3])
                        count = 0

                # Parse body
                if lc > 1:

                    # Skip empty lines
                    if len(fields) == 0:
                        continue

                    # Write geometry containing lines to variable
                    if len(fields) == 4:
                        # Parse commands
                        Elements[count] = fields[0]
                        Geometry[count, :] = np.array(
                            [float(fields[1]), float(fields[2]), float(fields[3])])
                        count = count + 1
                        if count == N_atoms:
                            elements.append(Elements)
                            geo.append(Geometry)

        # Consistency check
        if count != len(Elements):
            # Also should be a runtime error? - ERM
            print(
                "ERROR in xyz_parse: {} has less coordinates than indicated by the header.".format(xyz))
        if multiple is True:
            return elements, geo
        else:
            return Elements, Geometry

    # Commands for reading the atomtypes from the fourth column
    if read_types is True:

        # Iterate over the remainder of contents and read the
        # geometry and elements into variable. Note that the
        # first two lines are considered a header
        with open(xyz, 'r') as f:
            for lc, lines in enumerate(f):
                fields = lines.split()

                # Parse header
                if lc == 0 or (len(fields) == 1 and fields[0].isdigit()):
                    if len(fields) < 1:
                        # Also should be a runtime error? - ERM
                        print(
                            "ERROR in xyz_parse: {} is missing atom number information".format(xyz))
                        quit()
                    else:
                        N_atoms = int(fields[0])
                        Elements = ["X"]*N_atoms
                        Geometry = np.zeros([N_atoms, 3])
                        Atom_types = [None]*N_atoms
                        count = 0

                # Parse body
                if lc > 1:

                    # Skip empty lines
                    if len(fields) == 0:
                        continue

                    # Write geometry containing lines to variable
                    if len(fields) > 3:
                        Elements[count] = fields[0]
                        Geometry[count, :] = np.array(
                            [float(fields[1]), float(fields[2]), float(fields[3])])
                        if len(fields) > 4:
                            Atom_types[count] = fields[4]
                        count = count + 1
                        if count == N_atoms:
                            elements.append(Elements)
                            geo.append(Geometry)

        # Consistency check
        if count != len(Elements):
            # Also should be a runtime error? - ERM
            print(
                "ERROR in xyz_parse: {} has less coordinates than indicated by the header.".format(xyz))

        if multiple is True:
            return elements, geo, Atom_types
        else:
            return Elements, Geometry, Atom_types
