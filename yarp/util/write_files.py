"""
Functions to generate common molecular structure files from yarpecules
"""

import numpy as np
from yarp.yarpecule.lewis.be_mat import return_formals

def mol_write_yp(file, elements, geo, bond_mat, adj_mat, append_opt=False):
    """
    Write a MOL file to disk from a yarpecule object.
    Or rather, from attributes passed in from a yarpecule object.
    We don't take kindly to attempts to update the core attributes of a class
    when we're not a class function.

    Parameters:
    -----------
    name : str
        Name of file to be generated

    elements : list of str
        Elements of molecule

    geo : numpy.ndarray
        3D cartesian coordinates of atoms in molecule

    bond_mat : numpy.ndarray
        bond electron matrix of the molecule

    adj_mat : numpy.ndarray
        adjacency matrix of the molecule

    append_opt : bool (default = False)
        If true, MOL file will be generated in append mode.
        If false, MOL file will be generated in write mode.
    """

    # Consistency check
    if len(elements) >= 1000:
        print("ERROR in mol_write: the V2000 format can only accomodate up to 1000 atoms per molecule.")
        return
    mol_dict = {3: 1, 2: 2, 1: 3, -1: 5, -2: 6, -3: 7, 0: 0}
    # Check for append vs overwrite condition
    if append_opt == True:
        open_cond = 'a'
    else:
        open_cond = 'w'

    # Parse the basename for the mol header
    base_name = file.split(".")
    if len(base_name) > 1:
        base_name = ".".join(base_name[:-1])
    else:
        base_name = base_name[0]

    keep_lone = [count_i for count_i, i in enumerate(
        bond_mat) if i[count_i] % 2 == 1]

    # deal with radicals
    fc = list(return_formals(bond_mat, elements))

    # deal with charges
    chrg = len([i for i in fc if i != 0])
    valence = []  # count the number of bonds for mol file
    for count_i, i in enumerate(bond_mat):
        bond = 0
        for count_j, j in enumerate(i):
            if count_i != count_j:
                bond = bond+int(j)
        valence.append(bond)

    # Write the file
    with open(file, open_cond) as f:
        # Write the header (3 lines)
        f.write(f"{base_name}\n")
        f.write("  yarp{}*3D*\n".format(
            __import__('datetime').datetime.now().strftime("%m%d%H%M%S")))
        f.write("\n")

        # Write the number of atoms and bonds
        f.write("{:>3d}{:>3d}  0  0  0  0  0  0  0  0  1 V2000\n".format(
            len(elements), int(np.sum(adj_mat/2.0))))

        # Write the geometry
        for count_i, i in enumerate(elements):
            f.write(" {:> 9.4f} {:> 9.4f} {:> 9.4f} {:<3s} 0 {:>2d}  0  0  0  {:>2d}  0  0  0  0  0  0\n".format(
                geo[count_i][0], geo[count_i][1], geo[count_i][2], i.capitalize(), mol_dict[fc[count_i]], valence[count_i]))

        # Write the bonds
        bonds = [(count_i, count_j) for count_i, i in enumerate(adj_mat)
                 for count_j, j in enumerate(i) if j == 1 and count_j > count_i]
        for i in bonds:

            # Calculate bond order from the bond_mat
            bond_order = int(bond_mat[i[0], i[1]])

            # add fix of bond order for dative bonds around the transition metal
            bond_elements = [elements[i[0]], elements[i[1]]]

            # print(f"bond_elements: {bond_elements}", flush = True)
            # if (_ in el_metals for _ in bond_elements):# and (bond_order == 0):
            #    #print(f"FOUND METAL! bond_elements: {bond_elements}", flush = True)
            #    bond_order = 1

            if bond_order == 0:
                bond_order = 1
            f.write("{:>3d}{:>3d}{:>3d}  0  0  0  0\n".format(
                i[0]+1, i[1]+1, bond_order))

        # write radical info if exist
        if len(keep_lone) > 0:
            if len(keep_lone) == 1:
                f.write("M  RAD{:>3d}{:>4d}{:>4d}\n".format(
                    1, keep_lone[0]+1, 2))
            elif len(keep_lone) == 2:
                f.write("M  RAD{:>3d}{:>4d}{:>4d}{:>4d}{:>4d}\n".format(
                    2, keep_lone[0]+1, 2, keep_lone[1]+1, 2))
            else:
                print("Only support one/two radical containing compounds, "
                      "radical info will be skipped in the output mol file...")

        if chrg > 0:
            if chrg == 1:
                charge = [i for i in fc if i != 0][0]
                f.write("M  CHG{:>3d}{:>4d}{:>4d}\n".format(
                    1, fc.index(charge)+1, int(charge)))
            else:
                info = ""
                fc_counter = 0
                for count_c, charge in enumerate(fc):
                    if charge != 0:
                        if (fc_counter % 8 == 0):  # Only 8 items a line#
                            info += "M  CHG{:>3d}".format(
                                chrg - fc_counter if chrg - fc_counter <= 8 else 8)
                        info += '{:>4d}{:>4d}'.format(count_c+1, int(charge))
                        fc_counter += 1
                info += '\n'
                f.write(info)

        f.write("M  END\n$$$$\n")

    return

def xyz_write(name, elements, geo, append_opt=False):
    """
    Write cartesian coordinates of a molecule to an XYZ file

    name : str
        Name of XYZ file to be generated
    
    elements : list of str
        Elements of molecule.
        Will be written to file in uppercase.

    geo : numpy array
        Cartesian coordinates (N x 3) numpy array

    append_opt : bool (default = False)
        Option to append structure on to an already existing XYZ file
    """
    if append_opt == False: 
        out=open(name, 'w+')
    else: 
        out=open(name, 'a+')

    file_str = xyz_generate_string(elements=elements, geo=geo)
    
    elements = [el.upper() for el in elements]
    
    out.write(file_str)
    out.close()

    return

def xyz_generate_string(elements, geo):
    """
    Generates a string in XYZ format from cartesian coordinates.

    elements : list of str
        Elements of molecule.
    geo : numpy array or list of lists
        Cartesian coordinates (N x 3).

    Returns:
        str: The formatted XYZ data as a single string.
    """
    lines = []
    
    # 1. Number of atoms
    lines.append(str(len(elements)))
    
    # 2. Comment line (standard XYZ format requires a blank or comment line here)
    lines.append("")
    
    # 3. Element and coordinates
    for i, element in enumerate(elements):
        symbol = element.upper()
        x, y, z = geo[i]
        lines.append(f"{symbol} {x:>12.8f} {y:>12.8f} {z:>12.8f}")
    
    # Join everything with newlines and add a trailing newline
    return "\n".join(lines) + "\n"