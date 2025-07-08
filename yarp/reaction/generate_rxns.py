import os
import fnmatch
import numpy as np
from openbabel import pybel

from yarp.yarpecule.yarpecule import yarpecule
from yarp.reaction.enum import break_bonds, form_n_bonds, bmfn
from yarp.yarpecule.lewis.be_mat import return_formals

# NOTE: Long-term I'd like to have all the default user inputs and error handling set up in a different centralized location
# But I need to recruit some users and devs to help me hammer this out well


def generate_rxns(inp):
    """
    init : dict
        literally the stuff contained under "initialize" in the input YAML file

    Returns a dictionary of reaction objects

    Should this be a class?
    """
    output = dict()

    # Initialize reactions for product enumeration
    if inp.enum_on:

        print("Enumeration ahoy!")

        if fnmatch.fnmatch(inp.d0_node, "*.p") or fnmatch.fnmatch(inp.d0_node, "*.pickle"):
            print("Processing starting node as YARP generated pickle file")
            raise RuntimeError("Not yet implemented!")
        else:
            print("Letting yarpecule object figure out what this is")
            molecule = yarpecule(inp.d0_node, mode="yarp")
            print(molecule.elements)
            print(molecule.adj_mat)

        products = enumerate_products(
            molecule, inp.n_break, inp.n_form, inp.enum_mode, inp.l_cutoff)

        for prod in products:
            # Do a quick optimization to make product geometries reflect new bonding
            prod = quick_geom_opt(prod, inp.quick_opt_lot)

            # Generate a reaction object from reactant/product pairs
            rxn = None

    else:
        raise RuntimeError("Non-enumeration routines are not yet implemented!")

    return output


def enumerate_products(r_yp, n_break, n_form, mode="concerted", cutoff=0.0, ring_mode=False):
    """
    r_yp : yarpecule object
        The reactant from which all products are enumerated

    n_break : int
        Number of bonds to break

    n_form : int
        Number of bonds to form

    mode : string
        Toggle between the two available product enumeration modes:
        Concerted (default) and sequential enumeration.

    cutoff : float
        Threshold used in sequential enumeration to discard unphysical Lewis structures
        with bond-electron matrix scores above this value.
    """

    print(f"Product enumeration with break {n_break}, form {n_form} "
          f"will be performed in {mode} mode.")

    if mode == "sequential":
        print(f" * WARNING: Sequential mode is expensive and "
              "may cause memory blow-up issues!")

        # Break bonds
        break_mol = list(break_bonds(r_yp, n=n_break))
        print(f" - Breaking {n_break} bonds formed "
              f"{len(break_mol)} intermediates")

        # Form bonds
        products = form_n_bonds(break_mol, n=n_form)
        print(f" - Forming {n_form} bonds formed "
              f"{len(products)} potential products")

        # Filter out the garbage potential products
        products = [_ for _ in products if _.bond_mat_scores[0]
                    <= cutoff and sum(np.abs(_.fc)) < 2.0]

        # This makes no sense to me... it only seems to throw away
        # ring-open structures... - ERM
        if ring_mode:
            product = []
            for _ in products:
                if _.rings != []:
                    if len(_.rings[0]) > 4:
                        product.append(_)
                    else:
                        product.append(_)
            products = product
        print(f" - {len(products)} cleaned products after filtering")

    elif mode == "concerted":
        products = list(bmfn(r_yp, n_break, n_form))
        print(f" - Enumerated {len(products)} products")
    else:
        raise RuntimeError("Please select either concerted or sequential as the "
                           "product enumeration mode!")

    return products


def quick_geom_opt(molecule, lot="uff"):
    '''
    Perform low-level level geometry optimization on yarpecule using openbabel.

    ERM: Can we just change the forcefield from UFF if we want?

    Parameters:
    ----------
    molecule : yarpecule object
        molecule to be optimized 

    lot : string
        Level of theory used for quick optimization

    Returns
    -------
    molecule : yarpecule object
        optimized molecule
    '''

    # Write yarpecule object to a temporary mol file
    mol_file = '.tmp.mol'
    mol_write_yp(mol_file, molecule, append_opt=False)

    # Use openbabel to perform geometry optimization
    mol = next(pybel.readfile("mol", mol_file))
    mol.localopt(forcefield=lot)

    # Update yarpecule with optimized geometry coordinates
    for count_i, i in enumerate(molecule.geo):
        molecule.geo[count_i] = mol.atoms[count_i].coords

    # Delete temporary mol file
    os.system("rm {}".format(mol_file))

    return molecule


def mol_write_yp(name, molecule, append_opt=False):
    """
    Write a MOL file to disk from a yarpecule object

    Parameters:
    -----------
    name : str
        Name of file to be generated

    molecule : yarpecule
        Yarpecule object to be written to file

    append_opt : bool (default = False)
        If true, MOL file will be generated in append mode.
        If false, MOL file will be generated in write mode.
    """
    elements = molecule.elements
    geo = molecule.geo
    bond_mat = molecule.bond_mats[0]
    adj_mat = molecule.adj_mat

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
    base_name = name.split(".")
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
    with open(name, open_cond) as f:
        # Write the header
        f.write('{}\nGenerated by mol_write.py\n\n'.format(base_name))

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
