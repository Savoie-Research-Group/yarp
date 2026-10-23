"""Step 03 trial: an "exhaustive" bnfn that enumerates off every kept BEM.

bnfn_exhaustive() is a verbatim copy of yarp.reaction.enum.bnfn (copied from
yarp/reaction/enum.py at commit 0b87143) with two changes, both marked
"STEP 03 CHANGE":
  1. Radical sites come from every kept BEM, return_radicals(y, all_bmats=True),
     instead of only bond_mats[0].
  2. The bond breaking and forming loop runs once per kept BEM, instead of
     only on the BEM with the lowest sum |formal charge| (fc_ind).
It also tags each product with the BEM it came from (product.source_bem), so
step 03 can report which BEM recovered a reaction. The yarp source is not
changed.

With a single kept BEM, both changes do nothing, so bnfn_exhaustive yields
exactly what bnfn yields.
"""
from copy import copy
from itertools import combinations

import numpy as np

from yarp.yarpecule.yarpecule import yarpecule
from yarp.util.misc import prepare_list
from yarp.reaction.enum_support import (
    apply_legacy_shared_atom_b2f2,
    legacy_shared_atom_b2f2_changes,
    return_radicals,
    return_bondtypes,
    unique_set_partition_generator,
    add_bonds,
)


def bnfn_exhaustive(yarpecules, n, react=[], hashes=None, hash_filter=False, lower_score=False, keep_symmetric=True, verbose=True, debug=False):
    """
    This function provides a shortcut for enumerating "break n form n" products without generating intermediate 
    zwitterionic/dangling bond species

    Still need to implement the keep_symmetric option.

    Parameters
    ----------
    yarpecules: list of yarpecules
                This list holds the yarpecules that should be reacted. 

    react: set, default=None
           When supplied this is used to restrict bond formations only to those atoms in this set. If supplied, then `react` must
           have a searchable list or set (i.e., the function uses an `in` call, so sets are better) per `yarpecule`. An empty list
           is interpreted as all atoms being available to react. 

    hashes: set, default=None
            When supplied, this is used to avoid the generation of products that resolve to the same hash as any that are already
            in this set. This is useful whenever you have a set of products that you have already performed an exploration of and 
            don't want this function to waste time with. For example, if you are performing multiple sequential `form_bonds()` 
            calls, then it is useful to pass the hashes of the genereated products from each call forward to the next to avoid 
            redundant calls. 

    hash_filter: bool, default=False
                 Controls whether the returned products are filtered by uniqueness. Due to symmetry, the same product may be obtained
                 by several distinct bond formations. The default behavior is to avoid returning products that resolve to the same hash.
                 Disabling this option will lead to all distinct mappings being returned (with the associated redundancy). Since isotopomers
                 resolve to distinct hashes, even with this option enabled there may be the appearance of redundant products, but the isotope 
                 placement will be distinct.  

    lower_score: bool, default=False
                 During the enumeration it is common to form species that have poor Lewis structures that cost a time to 
                 perform enumeration on. These are often thrown away after enumeration, but they can cost a lot of time to
                 perform enumeration on if a multi-bond enumeration is being done. When this option is True, structures are 
                 only retained if they result in a bond-electron matrix that has a score that is less than or equal to the 
                 inputted yarpecule. 

    Yields
    ------

    product: yarpecule
             The generator yields a yarpecule object holding the bond_electron matrix and other core yarpecule attributes of
             the product resulting from bond formations.
    """

    # Wrap yarpecules in a list if only one is supplied
    yarpecules = prepare_list(yarpecules)
    
    if verbose:
        print(f"Enumerating break {n} form {n} products for {len(yarpecules)} yarpecules.")
        print(f"Reactive atoms defined as: {react}")

    # Prepare react list if it isn't the same length as the number of yarpecules
    if len(react) != len(yarpecules):
        react = [set(range(len(y))) for y in yarpecules]

    # Prepare empty set if none was supplied
    if hashes is None:
        hashes = set([])

    # Perform all bond breaks over relevant atoms
    for count_y, y in enumerate(yarpecules):

        # STEP 03 CHANGE 1: radical sites from every kept BEM. bnfn uses
        # return_radicals(y), which reads bond_mats[0] only.
        radicals = list(return_radicals(y, all_bmats=True))

        # STEP 03 CHANGE 2: break and form bonds on every kept BEM. bnfn picks only the
        # BEM that minimizes the formal charges:
        #     fc_ind = [sum(abs(x) for x in return_formals(_, y.elements))
        #               for _ in y.lewis.bond_mats]
        #     fc_ind = fc_ind.index(min(fc_ind))
        # The index keeps the name fc_ind so that the code below is bnfn's, indented once.
        for fc_ind in range(len(y.lewis.bond_mats)):

            # Return the bonds available to break (the use of the atom hash is to avoid redundant bond formations)
            # returns all bonds, with their atomic hashes and bond orders
            bonds = return_bondtypes(y, b_inds=[fc_ind])[0]
            # only keep the bonds that involve atoms in the react list
            bonds = [_ for _ in bonds if (
                _[0] in react[count_y] and _[1] in react[count_y])]

            # Loop over combinations of bonds to break (m bonds at a time)
            for b in combinations(list(range(len(bonds))), n):

                # Create set to avoid reforming the exact same bonds we just broke
                # You read this as set(frozenset(bonds we just broke)). We use frozensets so that they are
                # order independent (like sets) but hashable (like tuples) so that they can be used as keys in a set
                # for rapid lookup.
                avoid = set(
                    {frozenset([frozenset([bonds[_][0], bonds[_][1]]) for _ in b])})
                # Assemble list of reactive atoms: atoms from broken bonds + radical sites
                # Get atoms from bonds being broken (first 2 elements of each bond via bonds[j][:2])
                formset = [i for j in b for i in bonds[j][:2]]

                # Add radical atoms that can form new bonds
                formset += radicals

                # Debug output
                if debug:
                    print(f"Breaking bonds at indices: {b}")
                    print(f"Reactive atom set: {formset}")
                    print(f"Bonds to avoid reforming: {[y.describe_bond_pattern(_) for _ in avoid]}")
                    print(f"Number of reactive atoms: {n * 2}")
                    print(f"Breaking bonds: {[y.describe_bond_tuple(bonds[_]) for _ in b]}")

                # Start with copy of original bond matrix
                base_bmat = copy(y.lewis.bond_mats[fc_ind])
                if debug:
                    print("Original bond matrix:")
                    print(base_bmat)

                # Break the selected bonds (subtract 1 from bond order)
                base_bmat = add_bonds(base_bmat, [bonds[_] for _ in b], val=-1)
                if debug:
                    print("Bond matrix after breaking bonds:")
                    print(base_bmat)

                # A repeated B2F2 endpoint can represent the legacy shared-atom
                # bond/electron rearrangement, so identify it before ordinary
                # pairing treats the repeated endpoint as a dangling bond.
                shared_atom_changes = legacy_shared_atom_b2f2_changes(
                    n, formset, radicals, y.lewis.bond_mats[fc_ind], y.elements
                )

                # Loop over all unique ways to pair reactive atoms into new bonds
                if shared_atom_changes is not None:
                    formation_changes = [
                        (change.bonds_to_form, change)
                        for change in shared_atom_changes
                    ]
                else:
                    formation_changes = [
                        (formation, None)
                        for formation in unique_set_partition_generator(formset, 2)
                    ]

                if debug:
                    print(f"this is the formset: {formset}")
                    print(f"these are the bond formations we will test: "
                          f"{[formation for formation, _ in formation_changes]}")

                for g, shared_atom_change in formation_changes:

                    # Skip if we would just reform a bond we broke
                    if frozenset(g) in avoid:
                        if debug:
                            print(f"Skipping - would reform broken bond: {g}")
                        continue

                    # Skip if there will be a dangling bond owing to one of the atoms being involved in multiple bonds that were broken
                    if any([len(_) < 2 for _ in g]):
                        avoid.update(frozenset(g))
                        if debug:
                            print(f"Skipping - would form dangling bond: {g}")
                        continue

                    if debug:
                        print(f"Forming bonds: {[y.describe_atom_pair(_) for _ in g]}")

                    if shared_atom_change is not None:
                        if debug:
                            print(
                                "Legacy shared-atom criterion: "
                                f"{shared_atom_change.criterion}"
                            )
                        # Mirror the complete legacy special case at the BEM level:
                        # account for electrons released/consumed by bond changes,
                        # then move the donor's electron pair to the shared atom.
                        product_bmat = apply_legacy_shared_atom_b2f2(
                            y.lewis.bond_mats[fc_ind],
                            [bonds[_] for _ in b],
                            shared_atom_change,
                        )
                        if debug:
                            print("Bond matrix after legacy electron redistribution:")
                            print(product_bmat)
                    else:
                        product_bmat = copy(base_bmat)
                        product_bmat = add_bonds(
                            product_bmat, [list(_) for _ in g], val=1
                        )

                    # Current YARP constructs a product from connectivity and then
                    # determines its Lewis structures. Convert the redistributed
                    # BEM only after all legacy BEM operations are complete.
                    adj_mat = np.where(product_bmat > 0, 1, 0).astype(int)
                    np.fill_diagonal(adj_mat, 0)

                    # Create new yarpecule product. The np.where is used to convert the bond matrix to an adjacency matrix.
                    product = yarpecule((
                        np.where(adj_mat > 0, 1, 0).astype(int),
                        y.geo.copy(),
                        y.elements,
                        y.q,
                        {
                            i: {
                                **dict(y.atom_info[i]),
                                "formal_charge": None,
                                "stereo": {"atom": None, "bonds": {}},
                            }
                            for i in y.atom_info
                        },
                    ), canon=False)

                    # STEP 03: record which BEM this product came from (not in bnfn)
                    product.source_bem = fc_ind

                    # Debug: show the transformation
                    if debug:
                        print(f"Original adjacency matrix:\n{y._adj_mat}")
                        print(f"New adjacency matrix:\n{product._adj_mat}")

                    # Optional: skip products with higher bond matrix scores (worse quality)
                    # The legacy shared-atom route predates the current Lewis-score gate;
                    # applying that gate here removes the intended CO-containing product.

                    if lower_score and shared_atom_change is None:
                        if product.lewis._scores[0] > y.lewis._scores[0]:
                            if debug:
                                print(f"Skipping - higher score: "
                                      f"{product.lewis._scores[0]} > {y.lewis._scores[0]}")
                            continue

                    # PROPOSAL: if keep_symmetric is True, then we can't just check the hash, because it is mapping independent (by design).
                    # instead, we need to check the bmat hash. I'm not sure if this is currently stored as an attribute of the yarpecule object.

                    # This will avoid duplicates that are symmetrically equivalent (so distinct mappings will get collapsed, which isn't usually what we want)
                    if product._yarpecule_hash not in hashes:
                        if hash_filter:
                            hashes.add(product._yarpecule_hash)

                        # Yield new product
                        if debug:
                            print(f"Yielding new product with hash: "
                                  f"{product._yarpecule_hash}")
                        yield product
                    # KMH: Added error message to let user know why some products were skipped
                    else:
                        if debug:
                            print(f"Skipping - product hash already in set: "
                                  f"{product._yarpecule_hash}")

