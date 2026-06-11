"""
Definition of yarpecule object class
"""
import os
import re
import numpy as np
from openbabel import pybel
from rdkit import Chem

from yarp.yarpecule.input_parsers import xyz_parse, xyz_q_parse, mol_parse, xyz_from_smiles
from yarp.yarpecule.graph.adjacency import table_generator, graph_seps
from yarp.yarpecule.lewis.bem_score import return_bo_dict, return_formals
from yarp.yarpecule.atom_mapping import canon_order
from yarp.yarpecule.hashes import atom_hash, yarpecule_hash
from yarp.util.properties import el_mass
from yarp.util.misc import prepare_list, merge_arrays
from yarp.util.write_files import mol_write_yp, xyz_write
from yarp.yarpecule.lewis.lewis_structure import lewis_struct

# Package-level warning suppression is configured in yarp/__init__.py.

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

    def __init__(self, mol, mode='yarp', canon=True, strict=False, atom_info=None):
        self._geo = None
        self._elements = None
        self._q = 0
        self._masses = None
        self._adj_mat = None
        self._atom_info = None

        self._read_structure(mol, mode, strict=strict, atom_info=atom_info)

        self._atom_hashes = None
        self._mapping = None

        self._order_atoms(canon=canon)

        self._lewis_struct = None
        self._bond_order_dict = None
        self._yarpecule_hash = None

        self._gen_lewis_struct()

        # These attributes should be set once enumeration is complete
        # Too many errors/warnings show up in RDkit/Open Babel
        self._canon_smi = None
        self._map_smi = None
        self._inchi = None


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

    def _read_structure(self, mol, mode, strict=False, atom_info=None):
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
        carried_atom_info = atom_info

        if isinstance(mol, (tuple, list)) and len(mol) in [4, 5]:
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
            if len(mol) == 5:
                if carried_atom_info is not None:
                    raise TypeError("Pass carried atom_info either as the fifth tuple item or as atom_info=..., not both.")
                carried_atom_info = mol[4]
            self._atom_info = carried_atom_info

        # xyz branch
        elif len(mol) > 4 and mol[-4:] == ".xyz":
            self._elements, self._geo = xyz_parse(mol)
            self._adj_mat = table_generator(self._elements, self._geo)
            self._q = xyz_q_parse(mol)
            self._atom_info = carried_atom_info

        # mol branch
        elif len(mol) > 4 and mol[-4:] == ".mol":
            self._elements, self._geo, self._adj_mat, self._q, self._atom_info = mol_parse(mol)

        # SMILES branch
        else:
            try:
                self._elements, self._geo, self._adj_mat, self._q, self._atom_info = xyz_from_smiles(
                    mol, mode=mode)
            except ValueError:
                raise
            except Exception as e:
                if mode == "yarp" and strict is False:
                    print(f"WARNING: yarp SMILES parsing failed, falling back to RDKit: {e}")
                    self._elements, self._geo, self._adj_mat, self._q, self._atom_info = xyz_from_smiles(
                        mol, mode="rdkit")
                else:
                    raise TypeError(
                        "The yarpecule constructor expects either an xyz file, mol file, or a smiles string.")

        # Calculate elementary attributes
        # eventually all functions will expect lowercase element labels
        self._elements = [_.lower() for _ in self._elements]

        # User can update via mass update function.
        self._masses = np.array([el_mass[_] for _ in self._elements])
        normalized_atom_info = {}
        for i in range(len(self._elements)):
            if self._atom_info is None:
                record = {}
            elif isinstance(self._atom_info, dict):
                record = dict(self._atom_info[i]) if i in self._atom_info else {}
            else:
                record = dict(self._atom_info[i])

            normalized_atom_info[i] = {
                "atom_index": i,
                "atom_map": record.get("atom_map", None),
                "element": self._elements[i],
                "formal_charge": record.get("formal_charge", None),
                "mass": record.get("mass", el_mass[self._elements[i]]),
                "stereo": {
                    "atom": record.get("stereo", {}).get("atom", None),
                    "bonds": dict(record.get("stereo", {}).get("bonds", {})),
                },
                "aromatic_input": record.get("aromatic_input", False),
            }

        provided_maps = [normalized_atom_info[i]["atom_map"] for i in normalized_atom_info if normalized_atom_info[i]["atom_map"] is not None]
        if len(provided_maps) != len(set(provided_maps)):
            dupes = sorted({m for m in provided_maps if provided_maps.count(m) > 1})
            raise ValueError(f"Duplicate atom-map indices in input structure: {dupes}")
        self._atom_info = normalized_atom_info

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
        if self._atom_info is not None:
            used = {self._atom_info[i]["atom_map"] for i in self._atom_info if self._atom_info[i]["atom_map"] is not None}
            next_map = 0
            for i in self._atom_info:
                if self._atom_info[i]["atom_map"] is None:
                    while next_map in used:
                        next_map += 1
                    self._atom_info[i]["atom_map"] = next_map
                    used.add(next_map)
                    next_map += 1

        if canon:
            self._elements, self._adj_mat, self._atom_hashes, self._mapping, self._geo, self._masses = canon_order(
                self._elements, self._adj_mat, masses=self._masses, things_to_order=[self._geo, self._masses])
            if self._atom_info is not None:
                reordered_atom_info = {}
                for new_idx, old_idx in enumerate(self._mapping):
                    record = dict(self._atom_info[old_idx])
                    record["atom_index"] = new_idx
                    reordered_atom_info[new_idx] = record
                self._atom_info = reordered_atom_info
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

    ######################
    # External Functions #
    ######################

    def get_smiles(self, verbose=False):
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
                     self.bond_mats[0], self.adj_mat, atom_info=self._atom_info)

        # Use RDKit to get canonical SMILES string
        # ERM Note: RDKit has an annoying "Warning: molecule is tagged as 2D, but at least one Z coordinate is not zero. Marking the mol as 3D."
        # which triggers whenever you initialize from a .mol file for various and sundry reasons.
        # I have decided it is not worth my time to continue troubleshooting how to avoid this.
        mol1 = Chem.rdmolfiles.MolFromMolFile(tmp_file, removeHs=True)
        if verbose:
            print("RDKit mol dump before mapping:")
            for line in Chem.MolToMolBlock(mol1).splitlines():
                print(line)
        atoms = mol1.GetNumAtoms()
        for idx in range(atoms):
            mol1.GetAtomWithIdx(idx).ClearProp("molAtomMapNumber")
        self._canon_smi = Chem.MolToSmiles(mol1, canonical=True)

        # Use RDKit to get atom-mapped SMILES string
        mol2 = Chem.rdmolfiles.MolFromMolFile(tmp_file, removeHs=False)
        atoms = mol2.GetNumAtoms()
        for idx in range(atoms):
            mol2.GetAtomWithIdx(idx).SetProp("molAtomMapNumber", str(self._atom_info[idx]["atom_map"]))
        if verbose:
            print("RDKit mol dump after mapping:")
            for line in Chem.MolToMolBlock(mol2).splitlines():
                print(line)
        self._map_smi = Chem.MolToSmiles(mol2, canonical=True)

        # Remove temporary file
        os.remove(tmp_file)

    def reactive_map_smi(self, react, debug=False):
        """
        Return a display-only mapped SMILES with selected atom maps marked.

        The `react` values are zero-based atom-map ids. This helper does not
        mutate map_smi and the returned string should not be parsed as SMILES.
        """
        if self.map_smi is None:
            self.get_smiles()

        if react is None or react == []:
            return self.map_smi
        if isinstance(react, set):
            react_maps = set(react)
        elif isinstance(react, tuple):
            react_maps = set(react)
        elif isinstance(react, list) and len(react) == 1 and isinstance(react[0], (set, list, tuple)):
            react_maps = set(react[0])
        else:
            react_maps = set(react)

        def mark_atom(match):
            body = match.group("body")
            atom_map = int(match.group("map"))
            if atom_map not in react_maps:
                return match.group(0)
            return f"[{body}*:{atom_map}]"

        marked = re.sub(r"\[(?P<body>[^\]:]+):(?P<map>\d+)\]", mark_atom, self.map_smi)
        if debug:
            print(f"Reactive atom maps requested: {sorted(react_maps)}")
            print(f"Canonical mapped SMILES: {self.map_smi}")
            print(f"Marked reactive-map display: {marked}")
        return marked

    def get_inchi(self, verbose=False):
        """
        Generate the InChIKey for a given yarpecule using RDKit.
        Requires the yarpecule to already have SMILES
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
                if verbose:
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

    def update_atom_order(self, atom_index=None, canon=True):
        """
        Update the atom order of the yarpecule.
        And then update all the other attributes that depend on the atom order.

        User can just ask to canonicalize the yarpecule,
        or they can provide a magic little list to tell us how to reorder the atoms.
        Not sure what exactly this should look like yet. - ERM
        """

    def join(self, yarpecules, canon=True):
        """
        Method for creating a new yarpecule containing the union of the current yarpecule and all supplied yarpecules.

        Parameters
        ----------
        yarpecules: list of yarpecules
                    A list of the yarpecules that the user wants to merge with this yarpecule.
                    Can also handle a single yarpecule being submitted.

        canon: bool, default=True
            Controls whether or not the resulting yarpecule is subjected to the canonicalization ordering procedure.

        Returns
        -------
        yarpecule: yarpecule
                A new yarpecule containing the union of the chemical graphs contained in the supplied yarpecules.

        Notes
        -----
        The resulting yarpecule will not retain any of the bond-electron matrix information of the parent yarpecules.
        Atom maps are preserved when possible. If joined components have missing
        or overlapping atom maps, new zero-based maps are assigned where needed
        and the before/after mapped SMILES are printed.
        """
        yarpecules = prepare_list(yarpecules) # handles the singular case
        all_y = [self] + yarpecules # add self to the list

        adj_mat = merge_arrays([ y.adj_mat for y in all_y ])
        geo = np.vstack([ y.geo for y in all_y])
        elements = [ e for y in all_y for e in y.elements ]
        q = int(sum([ y.q for y in all_y ]))

        atom_info = {}
        offset = 0
        used_maps = set()
        next_map = 0
        remapped = []

        for count_y, y in enumerate(all_y):
            for i in range(len(y.elements)):
                original_info = dict(y._atom_info[i])
                original_map = original_info.get("atom_map")

                if original_map is None or original_map in used_maps:
                    while next_map in used_maps:
                        next_map += 1
                    atom_map = next_map
                    next_map += 1
                    used_maps.add(atom_map)
                else:
                    atom_map = original_map
                    used_maps.add(atom_map)

                if original_map is not None and original_map != atom_map:
                    remapped.append((count_y, i, original_map, atom_map))

                atom_info[offset + i] = {
                    **original_info,
                    "atom_index": offset + i,
                    "atom_map": atom_map,
                    "formal_charge": None,
                    "stereo": {"atom": None, "bonds": {}},
                }
            offset += len(y.elements)

        joined = yarpecule((adj_mat, geo, elements, q, atom_info), canon=canon)
        if remapped:
            before = []
            for count_y, y in enumerate(all_y):
                if y.map_smi is None:
                    y.get_smiles()
                before.append(f"component {count_y}: {y.map_smi}")
            joined.get_smiles()
            print("WARNING: yarpecule.join() remapped overlapping atom maps.")
            print("Before join:")
            for line in before:
                print(f"  {line}")
            print(f"After join: {joined.map_smi}")
            for count_y, atom_i, old_map, new_map in remapped:
                print(
                    f"  component {count_y}, atom index {atom_i}: "
                    f"atom map {old_map} -> {new_map}"
                )
        return joined

    def separate(self, canon=True):
        """
        Method for separating discrete molecules into their own standalone yarpecule objects.
        Returns a copy of itself if there is only one discrete molecule.

        Parameters
        ----------
        canon: bool, default=True
            Controls whether or not the resulting yarpecules are subjected to the canonicalization ordering procedure.

        Returns
        -------
        mols: list of yarpecules
            If there are no distinct molecules, returns a single yarpecule object as a list of length 1.
        """

        # Find disconnected graphs based on adjacency matrix
        gs = graph_seps(self.adj_mat)

        groups  = [] # list of indexes for each disconnected graph
        loop_ind= []
        for i in range(len(gs)):
            if i not in loop_ind:
                new_group = [count_j for count_j,j in enumerate(gs[i,:]) if j >= 0]
                loop_ind += new_group
                groups += [new_group]

        if len(groups) == 1:
            # If there are no distinct molecules, return a new yarpecule with same info
            # NOTE: This is a case where it would be nice to have a "skip Lewis" option,
            # where we can just feed in the BEMs we already have.
            atom_info = {
                i: {
                    **dict(self._atom_info[i]),
                    "atom_index": i,
                    "formal_charge": None,
                    "stereo": {"atom": None, "bonds": {}},
                }
                for i in self._atom_info
            }
            return [yarpecule((self.adj_mat, self.geo, self.elements, self.q, atom_info), canon=canon)]
        else:
            # Iterate over each disconnected graph and generate new yarpecule
            mols = []
            for g in groups:
                # Isolate subsection of adjacency matrix
                frag_adj = self.adj_mat[g][:, g]

                # Isolate subsection of elements list
                frag_e = [self.elements[ind] for ind in g]

                # Isolate subsection of geometry coordinates
                N_atom = len(g)
                frag_geo = np.zeros([N_atom, 3])
                for count_i, i in enumerate(g):
                    frag_geo[count_i,:] = self.geo[i,:]

                # Calculate charge of subgraph
                # NOTE: We're basing this off of the best scoring BEM of original,
                # and I'm not really sure how robust of a strategy that is (ERM)
                frag_bem = [self.bond_mats[0][i] for i in g]
                frag_formals = return_formals(frag_bem, frag_e)
                frag_q = int(sum(frag_formals))

                old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(g)}
                frag_atom_info = {}
                for old_idx in g:
                    new_idx = old_to_new[old_idx]
                    frag_atom_info[new_idx] = {
                        **dict(self._atom_info[old_idx]),
                        "atom_index": new_idx,
                        "formal_charge": None,
                        "stereo": {"atom": None, "bonds": {}},
                    }
                mols.append(yarpecule((frag_adj, frag_geo, frag_e, frag_q, frag_atom_info), canon=canon))

            return mols

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
            if self._canon_smi is None:
                self.get_smiles()
            xyz_write(filename, self.elements, self.geo, comment=self._canon_smi)
        elif format == 'mol':
            mol_write_yp(filename, self.elements, self.geo, self.bond_mats[0], self.adj_mat, atom_info=self._atom_info)
        else:
            raise RuntimeError("Valid export formats: xyz or mol")

    def draw_bmats(self, outfile="be_mats.pdf", show_inline=False):
        self._lewis_struct.draw_bmats(outfile, show_inline)
        return

    def describe_atom_pair(self, pair):
        """
        Given a pair of atom indices, 
        return a human-readable description of the pair 
        using the atom mapping information and element types.
        """
        i, j = sorted(tuple(pair))
        i_map = self._atom_info[i]["atom_map"]
        j_map = self._atom_info[j]["atom_map"]
        i_el = self.elements[i].upper()
        j_el = self.elements[j].upper()

        return f"atom {i_map} ({i_el}) and atom {j_map} ({j_el})"

    def describe_bond_tuple(self, bond):
        """
        Given a bond tuple, 
        return a human-readable description of the bond 
        using the atom mapping information and element types.
        """
        i, j = bond[:2]
        return self.describe_atom_pair((i, j))

    def describe_bond_pattern(self, pattern):
        """
        Given a bond pattern, 
        return a human-readable description of the pattern 
        using the atom mapping information and element types.
        """

        return [self.describe_atom_pair(pair) for pair in pattern]

    def __len__(self):
        return len(self._elements)

    def __eq__(self, other):
        return self.hash == other.hash

    def __hash__(self):
        return hash(self.hash)
