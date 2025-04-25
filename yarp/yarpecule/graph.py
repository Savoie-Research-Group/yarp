"""
Helper functions related to the graphical representation of molecules in YARP
"""
import numpy as np
from scipy.spatial.distance import cdist

from yarp.util.properties import el_radii, el_max_bonds


def table_generator(elements, geometry, scale_factor=1.2, filename=None):
    """ 
    Algorithm for finding the adjacency matrix of a geometry based on atomic separations. 

    Parameters
    ----------
    elements : list 
               Contains elemental information indexed to the supplied adjacency matrix. 
               Expects a list of lower-case elemental symbols.

    geo : array
          nx3 array of atomic coordinates (cartesian) in angstroms. 

    scale_factor: float, default=1.2
                  Used to scale the atomic radii to determine if a bond exists. 

    Returns
    -------
    adj_mat : array
              An nxn array indexed to elements containing ones bonds occur. 
    """

    # Print warning for uncoded elements.
    for i in elements:
        if i not in el_radii.keys():
            print("ERROR in Table_generator: The geometry contains an element ({}) that the Table_generator function doesn't have bonding information for. This needs to be directly added to the Radii".format(i) +
                  " dictionary before proceeding. Exiting...")
            quit()

    # Generate distance matrix holding atom-atom separations (only save upper right)
    dist_mat = np.triu(cdist(geometry, geometry))

    # Find plausible connections
    x_ind, y_ind = np.where((dist_mat > 0.0) & (
        dist_mat < max([el_radii[i]**2.0 for i in el_radii.keys()])))

    # Initialize the adjacency matrix
    adj_mat = np.zeros([len(geometry), len(geometry)])

    # Iterate over plausible connections and determine actual connections
    for count, i in enumerate(x_ind):

        # Assign connection if the ij separation is less than the UFF-sigma value times the scaling factor
        if dist_mat[i, y_ind[count]] < (el_radii[elements[i]]+el_radii[elements[y_ind[count]]])*scale_factor:
            adj_mat[i, y_ind[count]] = 1

        # Special treatment of hydrogens
        if elements[i] == 'H' and elements[y_ind[count]] == 'H':
            if dist_mat[i, y_ind[count]] < (el_radii[elements[i]]+el_radii[elements[y_ind[count]]])*1.5:
                adj_mat[i, y_ind[count]] = 1

    # Hermitize Adj_mat
    adj_mat = adj_mat + adj_mat.transpose()

    # Perform some simple checks on bonding to catch errors
    problem_dict = {i: 0 for i in el_radii.keys()}

    # conditions isn't used in the code....? - ERM
    conditions = {"h": 1, "c": 4, "f": 1, "cl": 1,
                  "br": 1, "i": 1, "o": 2, "n": 4, "b": 4}
    for count_i, i in enumerate(adj_mat):

        if el_max_bonds[elements[count_i]] is not None and sum(i) > el_max_bonds[elements[count_i]]:
            problem_dict[elements[count_i]] += 1
            cons = sorted([(dist_mat[count_i, count_j], count_j) if count_j > count_i else (
                dist_mat[count_j, count_i], count_j) for count_j, j in enumerate(i) if j == 1])[::-1]
            while sum(adj_mat[count_i]) > el_max_bonds[elements[count_i]]:
                sep, idx = cons.pop(0)
                adj_mat[count_i, idx] = 0
                adj_mat[idx, count_i] = 0

    # Print warning messages for obviously suspicious bonding motifs.
    if sum([problem_dict[i] for i in problem_dict.keys()]) > 0:
        print("Table Generation Warnings:")
        for i in sorted(problem_dict.keys()):
            if problem_dict[i] > 0:
                if filename is None:
                    if i == "H":
                        print("WARNING in Table_generator: {} hydrogen(s) have more than one bond.".format(
                            problem_dict[i]))
                    if i == "C":
                        print("WARNING in Table_generator: {} carbon(s) have more than four bonds.".format(
                            problem_dict[i]))
                    if i == "Si":
                        print("WARNING in Table_generator: {} silicons(s) have more than four bonds.".format(
                            problem_dict[i]))
                    if i == "F":
                        print("WARNING in Table_generator: {} fluorine(s) have more than one bond.".format(
                            problem_dict[i]))
                    if i == "Cl":
                        print("WARNING in Table_generator: {} chlorine(s) have more than one bond.".format(
                            problem_dict[i]))
                    if i == "Br":
                        print("WARNING in Table_generator: {} bromine(s) have more than one bond.".format(
                            problem_dict[i]))
                    if i == "I":
                        print("WARNING in Table_generator: {} iodine(s) have more than one bond.".format(
                            problem_dict[i]))
                    if i == "O":
                        print("WARNING in Table_generator: {} oxygen(s) have more than two bonds.".format(
                            problem_dict[i]))
                    if i == "N":
                        print("WARNING in Table_generator: {} nitrogen(s) have more than four bonds.".format(
                            problem_dict[i]))
                    if i == "B":
                        print("WARNING in Table_generator: {} bromine(s) have more than four bonds.".format(
                            problem_dict[i]))
                else:
                    if i == "H":
                        print("WARNING in Table_generator: parsing {}, {} hydrogen(s) have more than one bond.".format(
                            filename, problem_dict[i]))
                    if i == "C":
                        print("WARNING in Table_generator: parsing {}, {} carbon(s) have more than four bonds.".format(
                            filename, problem_dict[i]))
                    if i == "Si":
                        print("WARNING in Table_generator: parsing {}, {} silicons(s) have more than four bonds.".format(
                            filename, problem_dict[i]))
                    if i == "F":
                        print("WARNING in Table_generator: parsing {}, {} fluorine(s) have more than one bond.".format(
                            filename, problem_dict[i]))
                    if i == "Cl":
                        print("WARNING in Table_generator: parsing {}, {} chlorine(s) have more than one bond.".format(
                            filename, problem_dict[i]))
                    if i == "Br":
                        print("WARNING in Table_generator: parsing {}, {} bromine(s) have more than one bond.".format(
                            filename, problem_dict[i]))
                    if i == "I":
                        print("WARNING in Table_generator: parsing {}, {} iodine(s) have more than one bond.".format(
                            filename, problem_dict[i]))
                    if i == "O":
                        print("WARNING in Table_generator: parsing {}, {} oxygen(s) have more than two bonds.".format(
                            filename, problem_dict[i]))
                    if i == "N":
                        print("WARNING in Table_generator: parsing {}, {} nitrogen(s) have more than four bonds.".format(
                            filename, problem_dict[i]))
                    if i == "B":
                        print("WARNING in Table_generator: parsing {}, {} bromine(s) have more than four bonds.".format(
                            filename, problem_dict[i]))
        print("")

    return adj_mat
