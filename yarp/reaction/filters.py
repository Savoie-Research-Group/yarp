import numpy as np
from yarp.yarpecule.distance_metrics import compute_min_distance
from yarp.reaction.enum import validate_reactive_maps_against_starting_species

def filter_enum_candidates(rxns, separate_prods=[], dG_cutoff=1000.0, dG_source=None, netconfig=None, react_atoms=[], verbose=False):
    """
    Parameters:
    -----------
    rxns : dict of reaction objects
        Reactions to extract enumeration candidates from

    separate_prods : str or list of integers
        Control whether enumeration candidates are separated into individual species or not

    dG_cutoff : float
        Upperbound of energy of activation barriers to be considered for enumeration

    dG_source : str
        Level of theory to use for dG_cutoff checks

    netconfig : NetworkConfig object
        Dataclass that holds settings for network exploration mode from input file

    react_atoms : list
        Reactive atom-map ids from the input file. When product separation is
        enabled, these are validated against the unseparated product before
        fragments are generated.

    Returns:
    --------
    candidates : list of yarpecule objects

    """
    if verbose:
        print(f" - Reading in {len(rxns)} total reactions")

    if isinstance(dG_source, str) and verbose:
        print(f" - Barrier filtering selected!")
        print(f"   Reactions with {dG_source} dG above {dG_cutoff} kcal/mol will be excluded from enumeration.")
    elif verbose:
        print(" - No barrier filtering will be performed prior to enumeration")

    r_set = set()
    clean_rxns = dict()
    for count_r, rxn in enumerate(rxns.values()):
        # Throw away all reactions above dG barrier (optionally)
        if isinstance(dG_source, str):
            dG = rxn.barrier.get(dG_source, None)
            if dG is not None and dG > dG_cutoff:
                if verbose:
                    print(f"  + Excluding {rxn.hash} (dG = {dG}) from enumeration")
                continue

        clean_rxns[count_r] = rxn

        # Get a set of all (remaining) reactant yarpecule hashes
        r_set.add(rxn.reactant.hash)

    candidates = []
    if netconfig.target_product is not None:
        if verbose:
            print(f" - Constrained network exploration mode selected!")
        candidates = apply_target_blinders(
            raw_rxns=clean_rxns, target_yp=netconfig.target_product,
            dist=netconfig.distance, mode=netconfig.mode,
            k_nodes=netconfig.n_nodes, tolerance=netconfig.tolerance,
            cap=netconfig.cap
        )
    else:
        for rxn in clean_rxns.values():
            candidates.append(rxn.product.graph)

    if separate_prods == 'all' and verbose:
        print(f" - Performing product separation on all reactions prior to enumeration")
    elif verbose:
        print(" - No product separation will be performed prior to enumeration")

    p_set = set()
    unique_candidates = []
    for mol in candidates:

        # Apply separate product routine to each/select products (optionally)
        if separate_prods == 'all':
            validate_reactive_maps_against_starting_species(mol, react_atoms)
            prod = separate_molecules(mol)
        else:
            prod = [mol]

        # Get a list of all (remaining) product yarpecules
        for p in prod:
            if p.hash in r_set:
                p.get_smiles()
                if verbose:
                    print(f"   + SKIPPING! {p.canon_smi} has already been explored off of as a reactant!")
                continue
            if p.hash in p_set: continue # Throw away all duplicate candidates

            p_set.add(p.hash)
            p.get_inchi()
            unique_candidates.append(p)

    if verbose:
        print(f" - {len(unique_candidates)} unique products identified for enumeration")
        
    return unique_candidates
        
def apply_target_blinders(raw_rxns, target_yp, dist='soergel', mode='beam', k_nodes=1, tolerance=0.0, cap='moderate',verbose=False):
    """
    Parameters:
    -----------
    raw_rxns : dictionary of reaction objects
        Possible reactions to perform network exploration via product enumeration

    target_yp : yarpecule
        The "end-goal" (product side) molecule of interest to network exploration

    dist : str
        Chemical distance metric used to evaluate candidates

    mode : str
        Select candidates based on a beam search ('beam') or distance cap ('capped')
        framework

    k_nodes : int
        Number of candidates to select during beam search mode

    tolerance : float
        Tolerance window for distance tie-breakers in beam search mode

    cap : str
        Protocol for distance cap framework. If 'moderate', all delta distances >= 0.0 will be kept.
        If 'aggressive', only positive delta distances will be kept as enumeration candidates.

    Returns:
    --------
    candidates : list of yarpecule objects
        Eligible species to perform network exploration via product enumeration
    """
    if target_yp.canon_smi is None:
        target_yp.get_smiles()
    target_dist_smi = target_yp.map_smi if dist == "am_ged" else target_yp.canon_smi

    candidates = []
    if mode == 'beam':
        if verbose:
            print(f"  + Selecting {k_nodes} enumeration candidates via beam search (with {tolerance} window)")
            print(f"    Target species: {target_yp.canon_smi}")
            print(f"    Distance metric: {dist}")

        # Compute distances for each reaction product
        mol2dist = dict()
        mol_set = set()
        for rxn in raw_rxns.values():
            mol = rxn.product.graph
            if mol.hash in mol_set: continue # Throw away all duplicate candidates
            mol_set.add(mol.hash)
            mol2dist[mol.hash] = compute_min_distance(mol, target_dist_smi, metric=dist)

        # Sort all hashes by distance (lowest to highest)
        sorted_hashes = sorted(mol2dist, key=mol2dist.get)
        
        # Determine the cutoff threshold
        # If we have fewer candidates than k_nodes, take them all. 
        # Otherwise, find the distance of the k-th item and add the tolerance window.
        if len(sorted_hashes) <= k_nodes:
            cutoff_dist = float('inf')
        else:
            # -1 because list indices are 0-based (e.g. 1st item is at index 0)
            kth_best_dist = mol2dist[sorted_hashes[k_nodes - 1]] 
            cutoff_dist = kth_best_dist + tolerance

        # Filter candidates based on the calculated cutoff
        top_k_mol_hashes = []
        for h in sorted_hashes:
            if mol2dist[h] <= cutoff_dist:
                top_k_mol_hashes.append(h)
            else:
                # Since the list is sorted, we can stop early once we exceed the cutoff
                break
        if verbose:
            print(f"  + Identified {len(top_k_mol_hashes)} candidates within window {tolerance} of top {k_nodes}")

        for rxn in raw_rxns.values():
            mol = rxn.product.graph
            if mol.hash in top_k_mol_hashes:
                if verbose:
                    print(f"  + Selecting {mol.canon_smi} for enumeration (distance = {mol2dist[mol.hash]})")
                candidates.append(mol)
            else:
                if verbose:
                    print(f"  + SKIPPED! {rxn.id} == {mol.canon_smi} (distance = {mol2dist[mol.hash]})")

    elif mode == 'capped':
        if verbose:
            print(f"  + Selecting enumeration candidates via distance capping strategy")
            print(f"    Target species: {target_yp.canon_smi}")
            print(f"    Distance metric: {dist}")
        for rxn in raw_rxns.values():
            r_dist = compute_min_distance(rxn.reactant.graph, target_dist_smi, metric=dist)
            p_dist = compute_min_distance(rxn.product.graph, target_dist_smi, metric=dist)
            diff = r_dist - p_dist
            if cap == 'moderate':
                if diff >= 0.0:
                    if verbose:
                        print(f"  + Selecting {rxn.id} == {rxn.reactant.graph.canon_smi} -> {rxn.product.graph.canon_smi} for enumeration (delta_dist = {diff})")
                    candidates.append(rxn.product.graph)
            elif cap == 'aggressive':
                if diff > 0.0:
                    if verbose:
                        print(f"  + Selecting {rxn.id} == {rxn.reactant.graph.canon_smi} -> {rxn.product.graph.canon_smi} for enumeration (delta_dist = {diff})")
                    candidates.append(rxn.product.graph)
            else:
                raise RuntimeError(f"Cutoff {cap} for capped exploration is not recognized/implemented!")
    else:
        raise RuntimeError(f"Network exploration mode {mode} is not recognized/implemented!")
    if verbose:
        print(f"  + Selected {len(candidates)} out of {len(raw_rxns)} potential candidates")
    return candidates


def filter_enum_products(raw_products, l_cutoff=0.0, fc_cutoff=2.0, ring_filter=False,verbose=False):
    """
    Parameters:
    -----------
    raw_products : list of yarpecule objects
        Products to be filtered

    l_cutoff : float (default = 0.0)
        Threshold used in sequential enumeration to discard unphysical Lewis structures
        with bond-electron matrix scores above this value.

    fc_cutoff : float (default = 2.0)
        Threshold used in sequential enumeration to discard unphysical Lewis structures
        with total formal charges at or above this value.

    ring_filter : bool (default = False)
        Filter out 3 and 4 member rings from enumerated products.

    Returns:
    --------
    clean : list of yarpecule objects
        Filtered list of products!
    """

    # Filter out the garbage potential products according to Lewis threshold
    if verbose:
        print(f"   + Applying Lewis score cutoff of {l_cutoff}")
    clean = [_ for _ in raw_products if _.bond_mat_scores[0] <= l_cutoff]

    # Filter out garbage potential products according to formal charge
    if verbose:
        print(f"   + Applying formal charge cutoff of {fc_cutoff}")
    clean = [_ for _ in clean if sum(np.abs(_.fc)) < fc_cutoff]

    # Filter out 3 and 4 member rings from potential products
    if ring_filter:
        if verbose:
            print(f"   + Removing 3 and 4 member rings")
        product = []
        for _ in clean:
            if _.rings != []:
                if len(_.rings[0]) > 4:
                    product.append(_)
            else:
                product.append(_)
        clean = product

    if verbose:
        print(f"   + Returning {len(clean)} products after filtering")
    return clean



def separate_molecules(node, verbose=False):
    """
    Parameters:
    -----------
    node : yarpecule object

    Returns:
    --------
    sep_mols : list of yarpecule objects
        Will be a single yarpecule if graph is not separable.
        Generates INCHI keys for newly separated yarpecule objects.
    """

    sep_mols = node.separate()
    if len(sep_mols) > 1:
        if verbose:
            print(f"  + Separating {node.inchi} into {len(sep_mols)} nodes")

        for mol in sep_mols:
            mol.get_inchi() # need to generate these for a newly initialized yarpecule object!
            mol.get_smiles()

    return sep_mols
