import numpy as np
from yarp.yarpecule.distance_metrics import compute_min_distance

def filter_enum_candidates(rxns, separate_prods=False, prop_filter=None, netconfig=None, verbose=False):
    """
    Parameters:
    -----------
    rxns : dict of reaction objects
        Reactions to extract enumeration candidates from

    separate_prods : bool
        Control whether enumeration candidates are separated into individual species or not

    prop_filter : PropertyFilterConfig
        Controls whether reactions are examined for certain properties before enumerating

    netconfig : ProductBlindersConfig object
        Dataclass that holds settings for product directed network exploration mode from input file

    Returns:
    --------
    candidates : list of yarpecule objects

    """
    if verbose:
        print(f" - Reading in {len(rxns)} total reactions")

    # Filter #1 (optional): Throw away all reactions above property threshold
    if prop_filter is not None and verbose:
        print(f" - Barrier filtering selected!")
        print(f"   Reactions with {prop_filter.source} {prop_filter.type} above {prop_filter.threshold} kcal/mol will be excluded from enumeration.")
    elif verbose:
        print(" - No barrier filtering will be performed prior to enumeration")

    r_set = set()
    tmp_rxn = dict()
    for count_r, rxn in enumerate(rxns.values()):
        if prop_filter is not None:
            rxn_prop_dict = getattr(rxn, prop_filter.type, None)
            if rxn_prop_dict is not None:
                prop = rxn_prop_dict.get(prop_filter.source, None)
                if prop is not None and prop > prop_filter.threshold:
                    if verbose:
                        print(f"  + Excluding {rxn.hash} ({prop_filter.type} = {prop}) from enumeration")
                    continue

        # Get a set of all (remaining) reactant yarpecule hashes
        r_set.add(rxn.reactant.hash)
        tmp_rxn[count_r] = rxn

    # Filter #2 (always): Throw away all previously explored reactions
    clean_rxns = dict()
    for count_r, rxn in enumerate(tmp_rxn.values()):
        prod = rxn.product.graph
        if prod.hash in r_set:
            if verbose: print(f"   + SKIPPING! {prod.canon_smi} has already been explored off of as a reactant!")
            continue
        clean_rxns[count_r] = rxn

    # Filter #4 (optional): separate multi-molecule product candidates (unimolecular enumeration)
    # TJB NOTE: Apply the existing product-separation behavior without interpreting
    # reactive atom maps here. The important design point is that filtering
    # decides which molecular graphs become enumeration candidates; it does
    # not decide whether a candidate has the requested reactive atoms. That
    # check happens later in enumerate_products, after any split fragment has
    # its final local atom indexing.
    if separate_prods and verbose:
        print(f" - Performing product separation on all reactions prior to enumeration")
    elif verbose:
        print(" - No product separation will be performed prior to enumeration")

    candidates = []
    source2cand = dict()
    p_set = set()
    for rxn in clean_rxns.values():
        mol = rxn.product.graph
        if separate_prods:
            prod = separate_molecules(mol)
        else:
            prod = [mol]
        source2cand[rxn.hash] = prod
        for p in prod:
            if p.hash in p_set: continue # Throw away all duplicate candidates
            if p.hash in r_set:
                print(f"   + SKIPPING! {p.canon_smi} has already been explored off of as a reactant!")
                continue # Second pass to avoid re-exploring (filter #2)
            p_set.add(p.hash)
            candidates.append(p)

    # Filter #4 (optional): Apply product blinders to narrow down candidates
    if netconfig is not None:
        if netconfig.target_product is not None: # ERM: To-do this is fragile!
            if verbose: print(f" - Constrained network exploration mode selected!")
            candidates = apply_product_blinders(
                raw_candidates=candidates, target_yp=netconfig.target_product,
                dist=netconfig.distance_metric, mode=netconfig.mode,
                k_nodes=netconfig.n_nodes, tolerance=netconfig.tie_window,
                verbose=verbose
            )

            # Mark which rxns were selected via product blinders
            final_cand_set = set()
            for mol in candidates:
                final_cand_set.add(mol.hash)
            
            for rxn_hash, cand_list in source2cand.items():
                for cand in cand_list:
                    if cand.hash in final_cand_set:
                        rxns[str(rxn_hash)].network_meta["prod_blind_selected"] = True

    if verbose:
        print(f" - {len(candidates)} unique products identified for enumeration") 
    return candidates

def apply_product_blinders(raw_candidates, target_yp,
                           dist='soergel', mode='beam',
                          k_nodes=1, tolerance=0.0, verbose=False):
    """
    Parameters:
    -----------
    raw_candidates : list of yarpecule objects
        Possible species to perform network exploration via product enumeration

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

        # Compute distances for each input candidate
        mol2dist = dict()
        mol_set = set()
        for mol in raw_candidates:
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

        for mol in raw_candidates:
            if mol.hash in top_k_mol_hashes:
                if verbose: print(f"  + Selecting {mol.canon_smi} for enumeration (distance = {mol2dist[mol.hash]})")
                candidates.append(mol)
            else:
                if verbose: print(f"  + SKIPPED! {mol.canon_smi} (distance = {mol2dist[mol.hash]})")
    else:
        raise RuntimeError(f"Network exploration mode {mode} is not recognized/implemented!")
    if verbose:
        print(f"  + Selected {len(candidates)} out of {len(raw_candidates)} potential candidates")
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
