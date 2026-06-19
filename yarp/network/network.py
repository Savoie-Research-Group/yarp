"""
Definition of the network object class.
"""

import networkx as nx
from copy import deepcopy
from typing import Iterable, Dict, Any, Optional

from yarp.yarpecule.yarpecule import yarpecule

class network:
    """
    Docs here!
    """

    def __init__(self, rxns, dG_lot=None):
        self.rxns = self._normalize_rxn_dict(rxns)

        considered_rxns = self._get_rxns_by_metadata(rxns.values(), key="prod_blind_selected", value=True)
        self.n_considered_rxns = len(considered_rxns)

        self.crn = self._gen_s2r_bipartite_graph(self.rxns, barrier_lot=dG_lot)

        self.n_species = len([
            n for n, attr in self.crn.nodes(data=True)
            if attr.get('type') == 'species'
        ])

        self.n_rxns = len([
            n for n, attr in self.crn.nodes(data=True)
            if attr.get('type') == 'reaction'
        ])
        

    def _normalize_rxn_dict(self, rxns):
        """
        Normalize incoming reactions to a dictionary keyed by reaction hash.
        Accepts either a dict of reactions or an iterable of reaction objects.
        """
        if isinstance(rxns, dict):
            rxn_iter = rxns.values()
        elif isinstance(rxns, (list, tuple, set)):
            rxn_iter = rxns
        else:
            raise TypeError("rxns must be a dict or iterable of reaction objects")

        normalized = {}
        for rxn in rxn_iter:
            if not hasattr(rxn, "hash"):
                raise AttributeError("All reaction objects must have a `hash` attribute")
            normalized[rxn.hash] = deepcopy(rxn)

        return normalized

    def _get_rxns_by_metadata(self, rxns: Iterable, key: str, value: Optional[Any] = None) -> Dict:
        """
        Return a dict of reactions whose `network_meta` dict contains `key`.
        If `value` is provided, only include reactions where `network_meta[key] == value`.
        """
        out = dict()
        for r in rxns:
            nm = getattr(r, "network_meta", None)
            if isinstance(nm, dict) and key in nm and (value is None or nm.get(key) == value):
                out[r.hash] = r
        return out

    def _gen_s2r_bipartite_graph(self, yp_rxns, barrier_lot=None):
        """
        Convert input reactions into NetworkX DiGraph (directed graph) object
        and return it
        """
        crn = nx.DiGraph()

        for rxn in yp_rxns.values():
            rxn_label = f"Rx_{rxn.hash}"
            crn.add_node(rxn_label, type='reaction')

            for r in rxn.reactant.species:
                reactant_label = f'Sp_{r.hash}'
                crn.add_node(reactant_label, type='species', smi=r.canon_smi)
                crn.add_edge(reactant_label, rxn_label, dG=rxn.barrier.get(barrier_lot, -1000), weight=1)

            for p in rxn.product.species:
                product_label = f'Sp_{p.hash}'
                crn.add_node(product_label, type='species', smi=p.canon_smi)
                crn.add_edge(rxn_label, product_label, dG=0, weight=0)

        return crn

    def calc_min_dist(self, start, end):
        """
        Parameters:
        -----------   
        start : yarpecule
            Starting species node of interest. Must be a single molecule!
        
        end : yarpecule
            Terminal species node of interest. Must be a single molecule!       

        Returns:
        --------
        distance : float
            Minimum distance (reaction steps only) between start and end species nodes
        """

        if not isinstance(start, yarpecule) or not isinstance(end, yarpecule):
            raise TypeError("Requested start and end nodes must be yarpecule objects")

        if len(start.separate()) > 1 or len(end.separate()) > 1:
            raise RuntimeError("Requested start and end nodes must be single molecules")

        start_label = f'Sp_{start.hash}'
        end_label = f'Sp_{end.hash}'

        path = nx.shortest_path(G=self.crn, source=start_label, target=end_label)

        # Only count reaction nodes as "steps" along the path
        dist = sum(1 for s in path if "Rx_" in s)

        return dist

    def calc_eff_branching(self, start, end, tolerance=1e-1, max_iter=100):
        if not isinstance(start, yarpecule) or not isinstance(end, yarpecule):
            raise TypeError("Requested start and end nodes must be yarpecule objects")

        if len(start.separate()) > 1 or len(end.separate()) > 1:
            raise RuntimeError("Requested start and end nodes must be single molecules")

        d = self.calc_min_dist(start=start, end=end)
        if self.n_considered_rxns > 0:
            T = self.n_considered_rxns
        else:
            T = self.n_rxns

        if T < d:
            raise ValueError(f"Total nodes (T={T}) cannot be less than depth (d={d}).")
        if T == d:
            return 1.0

        # Binary search bounds
        low = 1.0

        # Mathematical optimization: For b > 1, b^d is strictly less than
        # the sum of the geometric series (T). Therefore, b < T^(1/d).
        # We add a small buffer (+1.0) to establish a safe upper bound.
        high = (T ** (1.0 / d)) + 1.0

        for _ in range(max_iter):
            mid = (low + high) / 2

            # Calculate the sum of the geometric series for the current mid value
            try:
                # Formula: mid * (mid^d - 1) / (mid - 1)
                val = mid * (mid**d - 1) / (mid - 1)
            except OverflowError:
                # If mid**d overflows, mid is significantly too high
                high = mid
                continue

            # Check if we are within the tolerance or if the bounds have collapsed
            if abs(val - T) < tolerance or (high - low) < tolerance:
                return mid

            if val < T:
                low = mid
            else:
                high = mid

        return (low + high) / 2

    def calc_eff(self, start, end):
        """
        Parameters:
        -----------
        start : yarpecule
            Starting species node of interest. Must be a single molecule!

        end : yarpecule
            Terminal species node of interest. Must be a single molecule!

        Returns:
        --------
        efficiency : float
            Inverse of minimum distance between start and end nodes
        """

        if not isinstance(start, yarpecule) or not isinstance(end, yarpecule):
            raise TypeError("Requested start and end nodes must be yarpecule objects")

        if len(start.separate()) > 1 or len(end.separate()) > 1:
            raise RuntimeError("Requested start and end nodes must be single molecules")

        d = self.calc_min_dist(start=start, end=end)
        return 1 / d

    def calc_wiener_index(self):
        """
        Returns:
        --------
        weiner_index : float
            Sum of shortest-path (weighted) distances between each pair of reachable nodes.
            Reactant species -> reaction nodes are weighted with 1
            Reaction nodes -> product species are weighted with 0
        """
        return nx.wiener_index(G=self.crn, weight='weight')

    def get_simple_paths(self, start, end, cutoff=None, verbose=False):
        """
        Parameters:
        -----------
        start : yarpecule
            Starting species node of interest. Must be a single molecule!

        end : yarpecule
            Terminal species node of interest. Must be a single molecule!
        
        cutoff : integer, optional
            Depth to stop the search. Only paths of length <= cutoff are returned

        Returns:
        --------
        rxn_paths : list of dict
            All simple paths (no repeated nodes) connecting start and end species.
            Each dictionary corresponds to a path and contains step number (key) and reaction object (value) pairs.
        """

        if not isinstance(start, yarpecule) or not isinstance(end, yarpecule):
            raise TypeError("Requested start and end nodes must be yarpecule objects")

        if len(start.separate()) > 1 or len(end.separate()) > 1:
            raise RuntimeError("Requested start and end nodes must be single molecules")

        if verbose:
            print(f"Getting all simple paths from {start.canon_smi} to {end.canon_smi}...")

        start_label = f'Sp_{start.hash}'
        end_label = f'Sp_{end.hash}'
        paths = list(nx.all_simple_paths(self.crn, start_label, end_label, cutoff=cutoff))

        rxn_paths = []
        for path in paths:
            if verbose:
                print(f"\nProcessing path: {path}")
            rxn_steps = path[1::2]  # Extract reaction labels from path
            rxn_path = dict()
            for i, rxn_label in enumerate(rxn_steps):
                # Extract hash from label (e.g., "Rx_123.456" -> 123.456)
                rxn_hash = float(rxn_label.split('_', 1)[1])
                rxn_path[i] = self.rxns.get(rxn_hash, None)

            rxn_paths.append(rxn_path)

        if verbose:
            print(f"Found {len(rxn_paths)} simple paths.")

        return rxn_paths

    def get_terminal_species(self, verbose=False):
        """
        Returns a list of yarpecule objects corresponding to terminal node species,
        which have never been used as reactants for a reaction
        """
        if verbose:
            print(f"Getting terminal species nodes...")
        # Get a list of terminal species labels (not hashes)
        terminal_labels = [
            n for n, attr in self.crn.nodes(data=True)
            if attr.get('type') == 'species'
            and self.crn.out_degree(n) == 0  # no outgoing edges
            and self.crn.in_degree(n) > 0    # connected to graph
        ]
        if verbose: 
            print(f"Found {len(terminal_labels)} terminal species nodes.")

        # Get a set of reaction labels that produced terminal species
        producing_rxn_labels = {
            reaction 
            for species_label in terminal_labels
            for reaction in self.crn.predecessors(species_label)
        }
        if verbose:
            print(f"Found {len(producing_rxn_labels)} reactions that produce terminal species.")

        # Extract hashes from reaction labels and get reaction objects
        terminal_rxns = []
        for rxn_label in producing_rxn_labels:
            rxn_hash = float(rxn_label.split('_', 1)[1])
            if rxn_hash in self.rxns:
                terminal_rxns.append(self.rxns[rxn_hash])

        # Extract species from terminal reactions
        terminal_yp = []
        seen = set()
        terminal_hashes = {float(label.split('_', 1)[1]) for label in terminal_labels}

        for rxn in terminal_rxns:
            if verbose:
                print(f"\nProcessing reaction {rxn.hash} with products {[mol.canon_smi for mol in rxn.product.species]}...")
            for mol in rxn.product.species:
                if verbose:
                    print(f"    Checking if product {mol.canon_smi} with hash {mol.hash} is a terminal species...")
                if mol.hash in terminal_hashes and mol.hash not in seen:
                    if verbose:
                        print(f"    Product {mol.canon_smi} with hash {mol.hash} is a terminal species.")
                    terminal_yp.append(deepcopy(mol))
                    seen.add(mol.hash)

        return terminal_yp

    def get_dijkstra_path(self, start, end, objective='dG'):
        """
        Parameters:
        -----------
        start : yarpecule
            Starting species node of interest. Must be a single molecule!

        end : yarpecule
            Terminal species node of interest. Must be a single molecule!

        objective : str
            Metric of interest to select an optimal path based on.
            Currently, we can only do dG searching.

        Returns:
        --------
        path_rxns : dict
            Optimal path from Dijkstra search.
            Contains step number (key) and reaction object (value) pairs.
        """

        if not isinstance(start, yarpecule) or not isinstance(end, yarpecule):
            raise TypeError("Requested start and end nodes must be yarpecule objects")

        if len(start.separate()) > 1 or len(end.separate()) > 1:
            raise RuntimeError("Requested start and end nodes must be single molecules")

        start_label = f'Sp_{start.hash}'
        end_label = f'Sp_{end.hash}'
        path = nx.dijkstra_path(self.crn, start_label, end_label, weight=objective)

        rxn_steps = path[1::2]  # Extract reaction labels from path

        path_rxns = dict()
        for i, rxn_label in enumerate(rxn_steps):
            # Extract hash from label (e.g., "Rx_123.456" -> 123.456)
            rxn_hash = float(rxn_label.split('_', 1)[1])
            path_rxns[i] = self.rxns.get(rxn_hash, None)

        return path_rxns