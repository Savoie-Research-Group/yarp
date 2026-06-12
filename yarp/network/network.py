"""
Definition of the network object class.
"""

import networkx as nx
from copy import deepcopy

from yarp.yarpecule.distance_metrics import soergel
from yarp.yarpecule.yarpecule import yarpecule

class network:
    """
    Docs here!
    """

    def __init__(self, rxns, dG_lot=None):
        self.rxns = self._normalize_rxn_dict(rxns)

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

    def _gen_s2r_bipartite_graph(self, yp_rxns, barrier_lot=None):
        crn = nx.DiGraph()

        for rxn in yp_rxns.values():
            rxn_label = f"Rx_{rxn.hash}"
            crn.add_node(rxn_label, type='reaction')

            for r in rxn.reactant.species:
                reactant_label = f'Sp_{r.hash}'
                crn.add_node(reactant_label, type='species', smi=r.canon_smi)
                crn.add_edge(reactant_label, rxn_label, dG=rxn.barrier.get(barrier_lot, -1000))

            for p in rxn.product.species:
                product_label = f'Sp_{p.hash}'
                crn.add_node(product_label, type='species', smi=p.canon_smi)
                crn.add_edge(rxn_label, product_label, dG=0)

        return crn
    

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