import numpy as np

def filter_enum_candidates(rxns, separate_prods=[], dG_cutoff=1000.0, dG_source=None, netconfig=None):
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

    Returns:
    --------
    candidates : list of yarpecule objects

    """
    print(f" - Reading in {len(rxns)} total reactions")

    if isinstance(dG_source, str):
        print(f" - Barrier filtering selected!")
        print(f"   Reactions with {dG_source} dG above {dG_cutoff} kcal/mol will be excluded from enumeration.")
    else:
        print(" - No barrier filtering will be performed prior to enumeration")

    r_set = set()
    clean_rxns = dict()
    for count_r, rxn in enumerate(rxns.values()):
        # Throw away all reactions above dG barrier (optionally)
        if isinstance(dG_source, str):
            dG = rxn.barrier.get(dG_source, None)
            if dG is not None and dG > dG_cutoff:
                print(f"  + Excluding {rxn.id} (dG = {dG}) from enumeration")
                continue

        clean_rxns[count_r] = rxn

        # Get a set of all (remaining) reactant yarpecule hashes
        r_set.add(rxn.reactant.hash)

    candidates = []
    if netconfig.target_product is not None:
        candidates = apply_target_blinders(
            clean_rxns, netconfig.target_product, netconfig.distance_metric, netconfig.mode, netconfig.n_nodes
        )
    else:
        for rxn in clean_rxns.values():
            candidates.append(rxn.product.graph)

    if separate_prods == 'all':
        print(f" - Performing product separation on all reactions prior to enumeration")
    else:
        print(" - No product separation will be performed prior to enumeration")

    p_set = set()
    for mol in candidates:

        # Apply separate product routine to each/select products (optionally)
        if separate_prods == 'all':
            prod = separate_molecules(mol)
        else:
            prod = [mol]

        # Get a list of all (remaining) product yarpecules
        for p in prod:
            if p.hash in r_set: continue # Throw away all products which have already been explored as reactants
            if p.hash in p_set: continue # Throw away all duplicate candidates

            p_set.add(p.hash)
            p.get_inchi()
            candidates.append(p)

    print(f" - {len(candidates)} unique products identified for enumeration")
    return candidates
        
def apply_target_blinders(raw_rxns, target_yp, dist='soergel', mode='beam', k_nodes=1):
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

    Returns:
    --------
    candidates : list of yarpecule objects
        Eligible species to perform network exploration via product enumeration
    """
    if target_yp.canon_smi is None:
        target_yp.get_smiles()

    candidates = []
    if mode == 'beam':
        print(f"  + Selecting {k_nodes} enumeration candidates via beam search")
        print(f"    Target species: {target_yp.canon_smi}")
        print(f"    Distance metric: {dist}")

        # Compute distances for each reaction product
        mol2dist = dict()
        mol_set = set()
        for rxn in raw_rxns.values():
            mol = rxn.product.graph
            if mol.hash in mol_set: continue # Throw away all duplicate candidates
            mol_set.add(mol.hash)
            mol2dist[mol.hash] = compute_min_distance(mol, target_yp.canon_smi, metric=dist)

        # Downselect candidates based on number of allowed beams
        top_k_mol_hashes = sorted(mol2dist, key=mol2dist.get)[:k_nodes]
        for rxn in raw_rxns.values():
            mol = rxn.product.graph
            if mol.hash in top_k_mol_hashes:
                print(f"  + Selecting {mol.inchi} for enumeration")
                candidates.append(mol)

    elif mode == 'capped':
        print(f"  + Selecting enumeration candidates via distance capping strategy")
        print(f"    Target species: {target_yp.canon_smi}")
        print(f"    Distance metric: {dist}")
        for rxn in raw_rxns.values():
            r_dist = compute_min_distance(rxn.reactant.graph, target_yp.canon_smi, metric=dist)
            p_dist = compute_min_distance(rxn.product.graph, target_yp.canon_smi, metric=dist)
            diff = p_dist - r_dist
            if diff >= 0.0:
                print(f"  + Selecting {rxn.product.graph.inchi} for enumeration")
                candidates.append(rxn.product.graph)

    else:
        raise RuntimeError(f"Network exploration mode {mode} is not recognized/implemented!")

    print(f"  + Selected {len(candidates)} out of {len(raw_rxns)} potential candidates")
    return candidates


def filter_enum_products(raw_products, l_cutoff=0.0, fc_cutoff=2.0, ring_filter=False):
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
    print(f"   + Applying Lewis score cutoff of {l_cutoff}")
    clean = [_ for _ in raw_products if _.bond_mat_scores[0] <= l_cutoff]

    # Filter out garbage potential products according to formal charge
    print(f"   + Applying formal charge cutoff of {fc_cutoff}")
    clean = [_ for _ in raw_products if sum(np.abs(_.fc)) < fc_cutoff]

    # Filter out 3 and 4 member rings from potential products
    if ring_filter:
        print(f"   + Removing 3 and 4 member rings")
        product = []
        for _ in raw_products:
            if _.rings != []:
                if len(_.rings[0]) > 4:
                    product.append(_)
            else:
                product.append(_)
        clean = product

    print(f"   + Returning {len(clean)} products after filtering")
    return clean



def separate_molecules(node):
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
        print(f"  + Separating {node.inchi} into {len(sep_mols)} nodes")

        for mol in sep_mols:
            mol.get_inchi() # need to generate these for a newly initialized yarpecule object!
            mol.get_smiles()

    return sep_mols

