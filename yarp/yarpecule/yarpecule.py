"""
Definition of yarpecule object class
"""
import os
import numpy as np
from copy import deepcopy
from openbabel import openbabel, pybel
from rdkit import Chem

from yarp.yarpecule.input_parsers import xyz_parse, xyz_q_parse, mol_parse, xyz_from_smiles
from yarp.yarpecule.graph.adjacency import table_generator, graph_seps
from yarp.yarpecule.lewis.be_mat import return_bo_dict
from yarp.yarpecule.atom_mapping import canon_order
from yarp.yarpecule.hashes import atom_hash, yarpecule_hash
from yarp.util.properties import el_mass
from yarp.util.misc import mol_write_yp, xyz_write
from yarp.yarpecule.lewis.lewis_structure import lewis_struct


class yarpecule:
    """
    Base class for describing a molecule in YARP

    MISSING: update_masses() <-- ERM: I see this defined, but never used in classy YARP. Do we need it?

    Parameters:
    -----------

    mol : var
          The input that supplies the molecular graph information. This can either be a smiles string, a tuple holding
          (adj_mat, elements, charge),  or one or more filenames. (<-- ERM: CAN it handle multiple files?)
          For strings the extension is used to determine which parser to use (e.g., .xyz etc),
          otherwise the constructor will attempt to parse the input as a smiles string using rdkit.

    canon : bool, default=True
            Controls whether the atoms are indexed based on a canonicalization routine. Default is `True`. 

    mode : str, default=yarp
            When parsing SMILES this controls whether RDKIT is used or the in-house parser. By default
            the in-house parser is used. This variable is unused if the molecular info
            is passed through another method besides SMILES.
            Thoughts on renaming this to smi_mode? - ERM

    Attributes:
    -----------
    geo : numpy.ndarray
            An (N_atom, 3) array containing the cartesian coordinates of each atom in the molecule.
            Units are in Angstroms.
            Array is indexed based on atomic ordering of the `yarpecule`.

    elements : list (str)
            A list of lower-case element labels indexed to the atomic ordering of the `yarpecule`.

    q : int
            The total charge on the `yarpecule`. 

    masses : numpy.array
            A list of the atomic masses in the yarpecule. These masses are used in the determination of uniqueness,
            such that isotopomers will be considered unique.

    adj_mat : numpy.ndarray
            The adjacency matrix of the graphical representation of the molecular structure.
            Array is indexed to atoms in the `yarpecule`. If atom_i and atom_j are
            bonded, matrix elements M_ij and M_ji are equal to 1. Otherwise,
            all elements are 0.

    atom_hashes : array
            A list of hash values for each atom, based on graph connectivity and the masses of the atoms.

    mapping : ???
            Oh dang, what is this friend?????

    lewis_struct : list of `lewis_struct` object(s)
            Lewis structure(s) of the yarpecule. Multiple structures are generated for cases involving resonance.

    yarpecule_hash : float
            A unique identifier for the yarpecule based on atom hashes and bond-electron matrices
            generated from the Lewis structure(s) of the yarpecule.
  """

    ###############
    # Constructor #
    ###############

    def __init__(self, mol, mode='yarp', canon=True):
        self._geo = None
        self._elements = None
        self._q = 0
        self._masses = None
        self._adj_mat = None

        self._read_structure(mol, mode)

        self._atom_hashes = None
        self._mapping = None

        self._order_atoms(canon=canon)

        self._lewis_struct = None
        self._bond_order_dict = None
        self._yarpecule_hash = None

        self._gen_lewis_struct()

        self._canon_smi = None
        self._map_smi = None

        self._get_smiles()

        self._inchi = None

        self._get_inchi()

    ###############
    # Properties  #
    ###############

    # Functions acting on yarpecules should pretty much never modify class attributes directly,
    # but often need to access them.
    # Therefore, these access function should be used, as they are "getters", but not "setters".
    # They return class attributes as (sort of) read-only.
    # However, mutable values are still mutable and can be modified, but I think that's just
    # what you get to deal with in python. - ERM

    @property
    def geo(self):
        return self._geo

    @property
    def elements(self):
        return self._elements

    @property
    def adj_mat(self):
        return self._adj_mat

    @property
    def q(self):
        return self._q

    @property
    def lewis(self):
        return self._lewis_struct

    @property
    def bond_mats(self):
        return self._lewis_struct._bond_mats

    @property
    def bond_mat_scores(self):
        return self._lewis_struct._scores

    @property
    def bo_dict(self):
        return self._bond_order_dict

    @property
    def hash(self):
        return self._yarpecule_hash

    @property
    def atom_hashes(self):
        return self._atom_hashes

    @property
    def n_e_accept(self):
        return self._lewis_struct._e_acceptors

    @property
    def n_e_donate(self):
        return self._lewis_struct._e_donors

    @property
    def atom_neighbors(self):
        return self._lewis_struct._atom_neighbors

    @property
    def fc(self):
        return self._lewis_struct._formal_charge

    @property
    def rings(self):
        return self._lewis_struct._rings
    
    @property
    def inchi(self):
        return self._inchi
    
    @property
    def canon_smi(self):
        return self._canon_smi
    
    @property
    def map_smi(self):
        return self._map_smi

    ######################
    # Internal Functions #
    ######################

    def _read_structure(self, mol, mode):
        """
        Read in an externally provided molecular structure and update
        core attributes of the yarpecule object.

        Parameters:
        -----------
        mol : str or tuple
                Input structure

        mode : str
                Mode to control SMILES parsing.

        Updated Attributes:
        ------------------
        self._adj_mat : numpy.ndarray
                Set to reflect input structure.

        self._geo : numpy.ndarray
                Set to reflect input structure.

        self._elements : list
                Set to reflect input structure.

        self._q : int
                Set to reflect input structure.

        self._masses : numpy.ndarray
                Atomic masses are computed from `elements`
                according to `el_mass` from `yarp.util.properties.py`
        """

        # direct branch: user passes core attributes directly
        if isinstance(mol, (tuple, list)) and len(mol) == 4:
            # consistency checks
            if (isinstance(mol[0], np.ndarray) is False or
                isinstance(mol[1], np.ndarray) is False or
                isinstance(mol[2], list) is False or
                    isinstance(mol[3], int) is False):
                raise TypeError(
                    "The yarpecule constructor expects a string or a tuple containing (adj_mat,geo,elements,q).")
            elif (len(mol[0]) != len(mol[1]) or len(mol[0]) != len(mol[2])):
                raise TypeError(
                    "The size of the adjacency array, geometry array, and elements list do not match.")

            # assign core attributes
            self._adj_mat = mol[0]
            self._geo = mol[1]
            self._elements = mol[2]
            self._q = mol[3]

        # xyz branch
        elif len(mol) > 4 and mol[-4:] == ".xyz":
            self._elements, self._geo = xyz_parse(mol)
            self._adj_mat = table_generator(self._elements, self._geo)
            self._q = xyz_q_parse(mol)

        # mol branch
        elif len(mol) > 4 and mol[-4:] == ".mol":
            self._elements, self._geo, self._adj_mat, self._q = mol_parse(mol)

        # SMILES branch
        else:
            try:
                self._elements, self._geo, self._adj_mat, self._q = xyz_from_smiles(
                    mol, mode=mode)
            except:
                raise TypeError(
                    "The yarpecule constructor expects either an xyz file, mol file, or a smiles string.")

        # Calculate elementary attributes
        # eventually all functions will expect lowercase element labels
        self._elements = [_.lower() for _ in self._elements]

        # User can update via mass update function.
        self._masses = np.array([el_mass[_] for _ in self._elements])

    def _order_atoms(self, canon=False, mapping=None):
        """
        Either canonically order the atoms or apply a user defined mapping.
        Not sure if the adjacency matrix is updated here, but I think it should be. - ERM

        Parameters:
        -----------
        canon : bool
                If True, the atoms are ordered based on a canonicalization routine.
                If False, the atoms are ordered based on the order they are provided.

        mapping : TBD! - ERM

        Updated Attributes:
        ------------------
        self._atom_hashes
                If canon is True, atom hashes are updated according to the `canon_order()` function.
                If canon is False, atom hashes are calculated directly from the `atom_hash()` function.

        self._mapping
                I don't know what is currently/should be done with this yet. - ERM
        """

        # TO-DO: send read-only copies to canon_order() and atom_hash()?
        if canon:
            self._elements, self._adj_mat, self._atom_hashes, self._mapping, self._geo, self._masses = canon_order(
                self._elements, self._adj_mat, masses=self._masses, things_to_order=[self._geo, self._masses])
        else:
            self._atom_hashes = np.array(
                [atom_hash(_, self._adj_mat, self._masses) for _ in range(len(self._elements))])
            # self._mapping = list(range(len(self)))

    def _gen_lewis_struct(self):
        """
        Compute Lewis structure(s) for the yarpecule.
        Also update the yarpecule hash while we're at it?

        Updated Attributes:
        ------------------
        self.lewis_struct
        """

        self._lewis_struct = lewis_struct(
            self._adj_mat, self._elements, self._q)

        self._bond_order_dict = return_bo_dict(self)

        self._yarpecule_hash = yarpecule_hash(self)

    def _get_smiles(self):
        """
        Generate a SMILES representation of the yarpecule.
        This shouldn't ever change any of the attributes of the yarpecule.
        Option to export SMILES with explicit atom mappings.
        Maybe also make it so we can optionally map the H atoms, but default to only reporting heavy atoms?

        
        Modifies
        --------
        self._canon_smi : str
        self._map_smi : str
        """
        # Generate a temporary MOL file from yarpecule
        tmp_file = ".tmp.mol"
        mol_write_yp(tmp_file, self.elements, self.geo,
                     self.bond_mats[0], self.adj_mat)

        # Use RDKit to get canonical SMILES string
        # ERM Note: RDKit has an annoying "Warning: molecule is tagged as 2D, but at least one Z coordinate is not zero. Marking the mol as 3D."
        # which triggers whenever you initialize from a .mol file for various and sundry reasons.
        # I have decided it is not worth my time to continue troubleshooting how to avoid this.
        mol1 = Chem.rdmolfiles.MolFromMolFile(tmp_file, removeHs=True)
        atoms = mol1.GetNumAtoms()
        for idx in range(atoms):
            mol1.GetAtomWithIdx(idx).ClearProp("molAtomMapNumber")
        self._canon_smi = Chem.MolToSmiles(mol1, canonical=True)

        # Use RDKit to get atom-mapped SMILES string
        mol2 = Chem.rdmolfiles.MolFromMolFile(tmp_file, removeHs=False)
        atoms = mol2.GetNumAtoms()
        for idx in range(atoms):
            mol2.GetAtomWithIdx(idx).SetProp("molAtomMapNumber", str(mol2.GetAtomWithIdx(idx).GetIdx()))
        self._map_smi = Chem.MolToSmiles(mol2)

        # Remove temporary file
        os.remove(tmp_file)

    def _get_inchi(self):
        """
        Generate the InChIKey for a given yarpecule using Open Babel.
        Each separable group within the yarpecule will have an independently
        generated InChIKey, with dashes connecting them together.

        Modifies
        --------
        self._inchi : str
        """
        # Access yarpecule information via "getter" property functions
        # This should raise an error if code is modifying the
        # class attributes (which it should NOT be doing here!!!)
        E = self.elements
        G = self.geo
        bond_mat = self.bond_mats[0]
        adj_mat = self.adj_mat

        # Identify separated graphs
        gs = graph_seps(adj_mat)

        groups = []
        loop_ind = []
        for i in range(len(gs)):
            if i not in loop_ind:
                new_group = [count_j for count_j,
                             j in enumerate(gs[i, :]) if j >= 0]
                loop_ind += new_group
                groups += [new_group]
        inchikey = []

        # Generate a temporary mol file for each separated graph
        # Then use it to get an INCHI key
        tmp_file = ".tmp.mol"
        for group in groups:
            # extract subgroup information
            N_atom = len(group)
            geo = np.zeros([N_atom, 3])
            for count_i, i in enumerate(group):
                geo[count_i, :] = G[i, :]
            elements = [E[ind] for ind in group]
            bem = [bond_mat[group][:, group]]
            adj = adj_mat[group][:, group]
            
            # generate mol file and read in to Open Babel
            mol_write_yp(tmp_file, elements, geo, bem[0], adj)
            mol = next(pybel.readfile("mol", tmp_file))
            
            try:
                inchi = mol.write(format='inchikey').strip().split()[0]
            except:
                print("WARNING: ERROR in INCHI key generation!")
                print(f"  --> {mol.write(format='inchikey')}")
                continue
            
            inchikey += [inchi]
            
            os.remove(tmp_file)

        if len(inchikey) == 0:
            self._inchi = "ERROR"
        elif len(groups) == 1:
            self._inchi = inchikey[0][:14]
        else:
            self._inchi = '-'.join(sorted([i[:14] for i in inchikey]))


    ######################
    # External Functions #
    ######################
    def update_atom_order(self, atom_index=None, canon=True):
        """
        Update the atom order of the yarpecule.
        And then update all the other attributes that depend on the atom order.

        User can just ask to canonicalize the yarpecule,
        or they can provide a magic little list to tell us how to reorder the atoms.
        Not sure what exactly this should look like yet. - ERM
        """

    def join(self, other_yps, canon=True, mode='rdkit'):
        """
        Join two yarpecules together to form a new yarpecule.
        """

    def export_geometry(self, filename, format='xyz'):
        """
        Export the geometry of the yarpecule to a file.
        This shouldn't ever change any of the attributes of the yarpecule.

        Parameters
        ----------
        filename : str
            The name of the file to export the geometry to.

        format : str, default='xyz'
            The format of the file to export the geometry to.
        """
        if format == 'xyz':
            xyz_write(filename, self.elements, self.geo)
        elif format == 'mol':
            mol_write_yp(filename, self.elements, self.geo, self.bond_mats[0], self.adj_mat)
        else:
            raise RuntimeError("Valid export formats: xyz or mol")

    def draw_bmats(self, outfile="be_mats.pdf", show_inline=False):
        self._lewis_struct.draw_bmats(outfile, show_inline)
        return

    def __len__(self):
        return len(self._elements)
