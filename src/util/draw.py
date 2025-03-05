"""
This module contains the helper functions for drawing molecules from yarpecules.
Also put the functions to write out to XYZ and mol files here?
"""
from rdkit.Chem import AllChem, rdchem, BondType, MolFromSmiles, Draw, Atom
from util.misc import prepare_list
from util.properties import el_to_an
from util.find_lewis import return_formals


def draw_yarpecules(yarpecules, name, label_ind=False, mol_labels=None):
    """
    Wrapper for drawing main bond_mat of a set of yarpecules using rdkit.

    Parameters
    ----------
    yarpecules: list
                list of yarpecule objects. 

    name: str
          filename for the save (should end in an image format supported by rdkit like \*.pdf).

    label_ind: bool, default=False
               Controls whether the atom indices are drawn in the structure. Default is no labels. 

    mol_labels: list
                This option can be used to display a label beneath each molecule (e.g., a score). 

    Returns
    -------
    None
    """

    # Handles the singular use-case
    yarpecules = prepare_list(yarpecules)

    # Return if there is nothing to draw
    if len(yarpecules) == 0:
        return

    # Keep rdkit happy
    elif len(yarpecules) > 250:
        print(
            f"Skipping draw_yarpecules() call. {len(yarpecules)} is too many for rdkit to render")
        return

    # Initialize the preferred lone electron dictionary the first time this function is called
    if not hasattr(draw_yarpecules, "bond_to_type"):
        draw_yarpecules.bond_to_type = {0: BondType.DATIVE, 1: BondType.SINGLE, 2: BondType.DOUBLE,
                                        3: BondType.TRIPLE, 4: BondType.QUADRUPLE, 5: BondType.QUINTUPLE, 6: BondType.HEXTUPLE}

    # loop over yarpecules, create an rdkit mol for each, then plot on a grid
    mols = []
    for count_i, i in enumerate(yarpecules):
        # throwaway molecule
        mol = MolFromSmiles("C")
        mol = rdchem.RWMol(mol)
        mol.RemoveAtom(0)
        # add atoms
        [mol.AddAtom(Atom(el_to_an[_])) for _ in i.elements]
        # add bonds
        for count_j, j in enumerate(i.adj_mat):
            for count_k, k in enumerate(j):
                if count_k < count_j:
                    if k == 0:
                        continue
                    else:
                        mol.AddBond(
                            count_j, count_k, draw_yarpecules.bond_to_type[i.bond_mats[0][count_j, count_k]])
                else:
                    break
        # set explicit H-atoms and formals
        fc = return_formals(i.bond_mats[0], [_.lower() for _ in i.elements])
        for count_j, j in enumerate(i.bond_mats[0]):
            atom = mol.GetAtomWithIdx(count_j)
            mol.GetAtomWithIdx(count_j).SetNumExplicitHs(0)
            mol.GetAtomWithIdx(count_j).SetFormalCharge(int(fc[count_j]))
            mol.GetAtomWithIdx(count_j).SetNumRadicalElectrons(
                int(j[count_j] % 2))
            try:
                mol.GetAtomWithIdx(count_j).UpdatePropertyCache()
            except:
                print(
                    f"problem is with atom {count_j=} {fc[count_j]=} {i.bond_mat_scores[0]=}:\n{i.elements}\n{i.adj_mat}\n{i.bond_mats[0]}")

        # generate coordinates
        AllChem.Compute2DCoords(mol)

        # assign index label
        if label_ind:
            # Iterate over the atoms
            for i, atom in enumerate(mol.GetAtoms()):
                # For each atom, set the property "molAtomMapNumber" to the index of a atom
                atom.SetProp("molAtomMapNumber", str(atom.GetIdx()+1))

        mols += [mol]
    # save the molecule
    if len(mols) <= 3:
        n_per_row = len(mols)
    else:
        n_per_row = 3
    if mol_labels:
        img = Draw.MolsToGridImage(mols, subImgSize=(
            400, 400), molsPerRow=n_per_row, legends=[str(_) for _ in mol_labels])
    else:
        img = Draw.MolsToGridImage(
            mols, subImgSize=(400, 400), molsPerRow=n_per_row)
    img.save(name)
    return


def draw_bmats(yarpecule, name):
    """
    Wrapper for drawing the bond_mats of a yarpecule using rdkit.

    Parameters
    ----------
    yarpecule: yarpecule
               yarpecule instance. 

    name: str
          filename for the save (should end in an image format supported by rdkit like `\*.pdf`).

    Returns
    -------
    None
    """

    # Initialize the preferred lone electron dictionary the first time this function is called
    if not hasattr(draw_bmats, "bond_to_type"):
        draw_bmats.bond_to_type = {0: BondType.DATIVE, 1: BondType.SINGLE, 2: BondType.DOUBLE,
                                   3: BondType.TRIPLE, 4: BondType.QUADRUPLE, 5: BondType.QUINTUPLE, 6: BondType.HEXTUPLE}
    # loop over bond_mats, create an rdkit mol for each, then plot on a grid with the scores
    mols = []
    for count_i, i in enumerate(yarpecule.bond_mats):
        # throwaway molecule
        mol = MolFromSmiles("C")
        mol = rdchem.RWMol(mol)
        mol.RemoveAtom(0)
        # add atoms
        [mol.AddAtom(Atom(el_to_an[_])) for _ in yarpecule.elements]
        # add bonds
        for count_j, j in enumerate(yarpecule.adj_mat):
            for count_k, k in enumerate(j):
                if count_k < count_j:
                    if k == 0:
                        continue
                    else:
                        mol.AddBond(
                            count_j, count_k, draw_bmats.bond_to_type[i[count_j, count_k]])
                else:
                    break
        # set explicit H-atoms and formals
        fc = return_formals(i, [_.lower() for _ in yarpecule.elements])
        for count_j, j in enumerate(i):
            atom = mol.GetAtomWithIdx(count_j)
            mol.GetAtomWithIdx(count_j).SetNumExplicitHs(0)
            mol.GetAtomWithIdx(count_j).SetFormalCharge(int(fc[count_j]))
            mol.GetAtomWithIdx(count_j).SetNumRadicalElectrons(
                int(j[count_j] % 2))
            mol.GetAtomWithIdx(count_j).UpdatePropertyCache()
        # generate coordinates
        AllChem.Compute2DCoords(mol)
        mols += [mol]

    # save the molecule
    if len(mols) <= 3:
        n_per_row = len(mols)
    else:
        n_per_row = 3
    # save the molecule
    img = Draw.MolsToGridImage(mols, subImgSize=(400, 400), molsPerRow=n_per_row, legends=[
                               "score: {: <4.3f}".format(_) for _ in yarpecule.bond_mat_scores])
    img.save(name)
    return
