"""
Wrapper function to manage the generation of reaction objects during main_yarp routine
"""
import os
import fnmatch
import pickle
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, cdist
from rdkit import Chem
from rdkit.Chem import AllChem

from yarp.yarpecule.yarpecule import yarpecule
from yarp.yarpecule.input_parsers import load_reaction_from_xyz_file, load_reactions_from_xyz_directory, load_reactions_from_smiles_file
from yarp.reaction.reaction import reaction
from yarp.reaction.enum import enumerate_products
from yarp.reaction.filters import filter_enum_candidates, filter_enum_products
from yarp.util.write_files import mol_write_yp


def generate_rxns(inp):
    """
    Wrapper function to manage the generation of reaction objects during main_yarp routine

    Parameters:
    -----------
    inp : InputParser object
        Settings parsed from user provided input file.
        TO-DO: Take advantage of more modular InputParser object and feed in only the pieces of
        the input file that are relevant to this part of the code.
        Will allow for more straightforward testing of this function.

    Returns:
    --------
    output : dict of reaction objects
        Reactant-product pairs contained in a reaction object and ready for further processing!
    """
    output = None
    verbose = inp.verbose

    # Initialize reactions for product enumeration
    if inp.enum.ON:
        print("Product enumeration enabled. Enumerating products.")
        if inp.init_struct.type == 'yarp_pickle':
            if verbose:
                print(" - Processing starting node(s) as YARP generated pickle file")

            og_rxns = pickle.load(open(inp.init_struct.source, 'rb'))
            assert isinstance(og_rxns, dict), "Input pickle file must contain a dictionary!"
            assert all(isinstance(v, reaction) for v in og_rxns.values()), "YARP requires a dictionary of reaction objects to continue"

            og_rxns_hash = set(og_rxns.keys())

            candidates = filter_enum_candidates(
                og_rxns, separate_prods=inp.enum.pre_enum_filters.separate_prods,
                prop_filter=inp.enum.pre_enum_filters.property_filter,
                netconfig=inp.enum.pre_enum_filters.product_blinders, verbose=verbose)

            new_rxns = dict()
            for mol in candidates:
                if verbose:
                    print(f" - Enumerating from {mol.inchi} ({mol.canon_smi}) node")

                # Reactive atom maps are resolved inside enumerate_products
                # against this final candidate graph. That is deliberately
                # later than candidate filtering/product separation, because a
                # split fragment can have a smaller atom-map subset and a new
                # local atom index order.
                raw_products = enumerate_products(
                    r_yp=mol, n_break=inp.enum.n_break, n_form=inp.enum.n_form,
                    react=inp.enum.react_atoms, mode=inp.enum.mode, verbose=verbose,
                )

                clean_products = filter_enum_products(
                    raw_products, l_cutoff=inp.enum.post_enum_filters.lewis_score,
                    fc_cutoff=inp.enum.post_enum_filters.formal_charge, ring_filter=inp.enum.post_enum_filters.ring_filter,
                    verbose=verbose
                )

                for prod in clean_products:
                    prod = quick_geom_opt(prod)
                    r2p = reaction(mol, prod)
                    p2r = reaction(mol, prod)

                    # Skip reactions already discovered (forward/reverse)
                    if r2p.hash in og_rxns_hash or p2r.hash in og_rxns_hash:
                        continue
                    new_rxns[r2p.hash] = r2p
            
            output = og_rxns | new_rxns
            
        else:
            if verbose:
                print(f" - Initializing starting reactant node from {inp.init_struct.source}")
            output = dict()
            reactant = yarpecule(inp.init_struct.source, mode="yarp")

            raw_products = enumerate_products(
                    r_yp=reactant, n_break=inp.enum.n_break, n_form=inp.enum.n_form,
                    react=inp.enum.react_atoms, mode=inp.enum.mode, verbose=verbose
                )

            clean_products = filter_enum_products(
                raw_products, l_cutoff=inp.enum.post_enum_filters.lewis_score,
                fc_cutoff=inp.enum.post_enum_filters.formal_charge, ring_filter=inp.enum.post_enum_filters.ring_filter,
                verbose=verbose
            )

            for prod in clean_products:
                prod = quick_geom_opt(prod)
                r2p = reaction(reactant, prod)
                output[r2p.hash] = r2p

    else:
        source = Path(inp.init_struct.source)
        print(f"Product enumeration not enabled. Initializing reactions from input node(s).")
        print(f" - Input node source: {source}")
        if inp.init_struct.type == 'yarp_pickle':
            if verbose:
                print(" - Processing starting node(s) as YARP generated pickle file")

            og_rxns = pickle.load(open(inp.init_struct.source, 'rb'))
            assert isinstance(og_rxns, dict), "Input pickle file must contain a dictionary!"
            assert all(isinstance(v, reaction) for v in og_rxns.values()), "YARP requires a dictionary of reaction objects to continue"

            output = og_rxns
        elif source.is_dir():
            print(f" - Processing starting node(s) as reaction xyz files in {source}")
            output = load_reactions_from_xyz_directory(source)

        elif source.is_file() and source.suffix.lower() == ".xyz":
            print(f" - Processing starting node(s) as reaction xyz file {source}")
            rxn = load_reaction_from_xyz_file(source)
            output = {rxn.hash: rxn}

        elif source.is_file():
            print(f" - Processing starting node(s) as mapped reaction SMILES in {source}")
            output = load_reactions_from_smiles_file(source)

        else:
            raise RuntimeError("We can only start from a YARP pickle file, a reaction xyz file, a directory of reaction xyz files, or a mapped reaction SMILES file currently, sorry friend!")

    return output


def quick_geom_opt(molecule, max_iters=2000, min_dist=0.7):
    '''
    Perform a low-level UFF geometry optimization on a yarpecule using RDKit.

    Each connected fragment is optimized independently. This is deliberate:
    enumerated products keep the reactant's coordinates, so a newly formed
    small molecule (e.g. an H2 leaving group) is placed wherever its atoms
    happened to sit in the reactant and can start grossly mispositioned
    relative to its bonding partner. Optimizing per fragment keeps a misplaced
    fragment from dragging the rest of the structure into a collapsed geometry.

    RDKit's UFF implementation is used rather than Open Babel's because the
    latter collapses certain strained topologies (e.g. an alpha-lactone: a
    3-membered C-C-O ring bearing a carbonyl), driving bonded atoms on top of
    one another regardless of the starting geometry or optimizer.

    The result is guarded: if RDKit cannot parse the structure or the optimized
    geometry fails a sanity check (any two atoms closer than `min_dist`
    Angstroms), the unoptimized geometry is kept rather than returning a broken
    one.

    Parameters:
    ----------
    molecule : yarpecule object
        molecule to be optimized

    max_iters : int
        Maximum UFF iterations per fragment.

    min_dist : float
        Minimum allowed interatomic distance (Angstroms) for an optimized
        geometry to be accepted. The shortest real bond (H-H in H2) is ~0.74 A,
        so anything below this floor indicates collapsed atoms.

    Returns
    -------
    molecule : yarpecule object
        optimized molecule (or the original molecule if optimization could not
        produce a physically valid geometry)
    '''

    # Write yarpecule object to a temporary mol file (carries connectivity,
    # bond orders, formal charges and radical info needed for UFF typing).
    mol_file = '.tmp.mol'
    mol_write_yp(mol_file, molecule.elements, molecule.geo,
                 molecule.bond_mats[0], molecule.adj_mat)

    try:
        opt_geo = _rdkit_uff_opt(mol_file, max_iters, min_dist)
    finally:
        os.system("rm {}".format(mol_file))

    # If RDKit could not produce a valid geometry, keep the unoptimized one.
    if opt_geo is None:
        print("WARNING: RDKit UFF optimization could not produce a valid "
              "geometry; keeping unoptimized geometry.")
        return molecule

    # Update yarpecule with optimized geometry coordinates
    for count_i in range(len(molecule.geo)):
        molecule.geo[count_i] = opt_geo[count_i]

    return molecule


def _rdkit_uff_opt(mol_file, max_iters, min_dist):
    '''
    UFF-optimize each connected fragment of a mol file with RDKit and return the
    optimized coordinates mapped back to the original atom order, or None if the
    structure could not be parsed or the result is degenerate.

    Parameters:
    ----------
    mol_file : str
        Path to the mol file holding the structure and connectivity.

    max_iters : int
        Maximum UFF iterations per fragment.

    min_dist : float
        Minimum allowed interatomic distance (Angstroms). A geometry with any
        pair of atoms closer than this is rejected.

    Returns
    -------
    geo : numpy.ndarray or None
        (N_atom, 3) optimized coordinates indexed to the original atom order,
        or None if parsing failed or atoms collapsed.
    '''
    mol = Chem.MolFromMolFile(mol_file, removeHs=False, sanitize=True)
    if mol is None:
        return None

    geo = np.zeros((mol.GetNumAtoms(), 3))

    # Split into connected fragments, tracking each fragment atom's index in the
    # parent molecule so optimized coordinates can be written back in order.
    frag_atom_mapping = []
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True,
                             fragsMolAtomMapping=frag_atom_mapping)

    for frag, parent_idx in zip(frags, frag_atom_mapping):
        try:
            AllChem.UFFOptimizeMolecule(frag, maxIters=max_iters)
        except Exception:
            return None
        frag_geo = frag.GetConformer().GetPositions()
        for local_i, parent_i in enumerate(parent_idx):
            geo[parent_i] = frag_geo[local_i]

    # Fragments are relaxed independently, so a small molecule formed from atoms
    # that started far apart (e.g. an H2 leaving group) relaxes near its original
    # centroid and can overlap another fragment. Translate clashing fragments
    # apart so the combined geometry is physically valid.
    if len(frag_atom_mapping) > 1:
        geo = _separate_clashing_fragments(geo, [list(m) for m in frag_atom_mapping])

    # Reject geometries where any two atoms have collapsed onto each other.
    if len(geo) > 1 and pdist(geo).min() < min_dist:
        return None

    return geo


def _separate_clashing_fragments(geo, frag_indices, buffer=2.0, step=0.5,
                                 max_shift=100.0):
    '''
    Translate connected fragments apart until no atom of one fragment is within
    `buffer` Angstroms of any atom of an already-placed fragment. The largest
    fragment is held fixed as the anchor; each remaining fragment is pushed out
    along the direction from the placed assembly's centroid toward its own
    centroid.

    Parameters:
    ----------
    geo : numpy.ndarray
        (N_atom, 3) coordinates, modified in place and returned.

    frag_indices : list of list of int
        Atom indices (into `geo`) for each connected fragment.

    buffer : float
        Minimum allowed inter-fragment atom-atom distance (Angstroms).

    step : float
        Translation increment (Angstroms) applied while pushing a fragment out.

    max_shift : float
        Safety cap on the total translation applied to any one fragment.

    Returns
    -------
    geo : numpy.ndarray
        Coordinates with clashing fragments separated.
    '''
    # Place the largest fragment first as the anchor.
    order = sorted(range(len(frag_indices)), key=lambda k: -len(frag_indices[k]))
    placed = list(frag_indices[order[0]])

    for k in order[1:]:
        idx = frag_indices[k]
        # Push direction: away from the placed assembly's centroid.
        direction = geo[idx].mean(axis=0) - geo[placed].mean(axis=0)
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-3 else np.array([1.0, 0.0, 0.0])

        shift = 0.0
        while (cdist(geo[idx], geo[placed]).min() < buffer and shift < max_shift):
            geo[idx] += direction * step
            shift += step

        placed += idx

    return geo
