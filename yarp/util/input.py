"""
Definition of input object class
"""
from dataclasses import dataclass, field
from typing import List, Optional, Union, Any
from yarp.yarpecule.yarpecule import yarpecule


# --- CONFIGURATION OBJECTS ---
# These classes act as simple containers for user provided settings.

@dataclass
class EnumerationConfig:
    """Holds settings specific to the generation of products via enumeration."""
    enumerate: bool = False
    mode: str = "concerted"
    n_break: int = 2
    n_form: int = 2
    react_atoms: List[set] = field(default_factory=list)

@dataclass
class EnumFilterConfig:
    """Holds settings specific to cleaning up enumeration outputs"""
    # Pre-enumeration filters
    dG_cutoff: float = 1000.0
    dG_source: Optional[str] = None
    separate_prods: Union[str, List[int]] = field(default_factory=list)

    # Post-enumeration filters
    l_cutoff: float = 0.0
    fc_cutoff: float = 2.0
    ring_filter: bool = False

@dataclass
class NetworkConfig:
    """Holds settings specific to generating a multi-layered reaction network"""
    target_product: Optional[yarpecule] = None
    distance: str = 'sorgel'
    mode: str = 'capped'
    n_nodes: Optional[int] = 1

# --- MAIN PARSER CLASS ---
# This class handles the messy logic of converting the YAML dict 
# into the clean config objects above.

class InputParser:
    """
    Parses the raw input dictionary and organizes settings into specific Config objects.
    """
    def __init__(self, file_dict: dict):
        # 1. Basic validation
        initnode = file_dict.get('initialize', None)
        if not initnode:
            raise RuntimeError("Hey bro beans, I need some molecules or reactions to work with. "
                               "Missing `initialize` node in YAML file.")

        # 2. Extract top-level attributes
        self.d0_node = initnode.get("initial species", None)
        if not self.d0_node:
            raise RuntimeError("Please provide an initial species for enumeration.")

        self.out_file = initnode.get("output", "reactions.pkl")

        # 3. Create Sub-Configuration Objects
        # We delegate the grouping of parameters to private helper methods
        self.enum = self._parse_enum_config(initnode)
        self.enum_filters = self._parse_enum_filters(initnode)
        self.net_explore = self._parse_network_config(initnode)

    def _parse_separate_prods(self, raw_value) -> Union[str, List[int]]:
        """Handles the logic for the 'separate products' input."""
        if raw_value is None:
            return []
        if isinstance(raw_value, str) and raw_value.lower() == 'all':
            return 'all'
        if isinstance(raw_value, int):
            return [raw_value]
        if isinstance(raw_value, list):
            return raw_value

        raise RuntimeError(f"Invalid value for separate products: {raw_value}")

    def _parse_enum_config(self, initnode: dict) -> EnumerationConfig:
        """Extracts enumeration settings and returns a clean EnumerationConfig object."""

        # Handle the reactive atoms list-to-set conversion
        raw_react = initnode.get("reactive atoms", None)
        react_atoms_processed = []
        if raw_react:
            react_atoms_processed = [set(raw_react)]

        return EnumerationConfig(
            enumerate=initnode.get("enumerate", False),
            mode=initnode.get("mode", "concerted"),
            n_break=initnode.get("bonds to break", 2),
            n_form=initnode.get("bonds to form", 2),
            react_atoms=react_atoms_processed,
        )

    def _parse_enum_filters(self, initnode: dict) -> EnumFilterConfig:
        """Extracts enumeration filtering settings and returns a clean EnumFilterConfig object."""

        # Handle complex "separate products" logic using a helper method
        separate_prods = self._parse_separate_prods(initnode.get("separate products"))

        # Handle nested filters
        filters = initnode.get('enumeration filters', {})
        # If filters is None (yaml key exists but is empty), treat as empty dict
        if filters is None: 
            filters = {}

        return EnumFilterConfig(
            l_cutoff=filters.get('lewis score', 0.0),
            fc_cutoff=filters.get('formal charge', 2.0),
            ring_filter=filters.get('discard strained rings', False),
            dG_cutoff=filters.get('barrier cutoff', -100.00),
            dG_source=filters.get('barrier source', None),
            separate_prods=separate_prods
        )

    def _parse_network_config(self, initnode: dict) -> NetworkConfig:
        """Extracts network exploration settings and returns a clean NetworkConfig object."""

        # Handle nested filters
        netconfig = initnode.get('network exploration', {})
        # If netconfig is None (yaml key exists but is empty), treat as empty dict
        if netconfig is None: 
            netconfig = {}

        target = netconfig.get("target product", None)
        if target is not None:
            target_yp = yp.yarpecule(target)
            target_yp.get_inchi()
            target_yp.get_smiles()
        else:
            target_yp = None

        return NetworkConfig(
            target_product=target_yp,
            distance=netconfig.get("distance metric", 'soergel'),
            mode=netconfig.get("mode", 'capped'),
            n_nodes=netconfig.get("n_nodes", 1)
        )
