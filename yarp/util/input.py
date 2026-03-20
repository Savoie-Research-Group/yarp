"""
Definition of input object class
"""
from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
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
    tolerance: float = 0.0
    cap: str = 'moderate'

@dataclass
class ConformerConfig:
    """Single Source of Truth for Conformer Generation parameter defaults!"""
    software: str = "crest"
    lot: str = "gfn2"
    n_cpus: int = 4
    mem_per_cpu: int = 1000
    energy_window: float = 6.0
    solvent: Optional[Dict[str, str]] = None
    charge: int = 0
    n_unpaired_electrons: int = 0

@dataclass
class GSMConfig:
    """Single Source of Truth for Growing String Method parameter defaults!"""
    software: str = "pysisyphus"
    selector: str = "rich"
    joint_opt: str = "dual"
    n_conf: int = 1
    n_cpus: int = 4
    mem_per_cpu: int = 1000
    bias_lot: str = "uff"
    gsm_lot: str = "xtb"
    max_gsm_nodes: int = 30
    charge: int = 0
    multiplicity: int = 1

@dataclass
class RPOptConfig:
    software: str = "pysisyphus"
    lot: str = "xtb"
    n_cpus: int = 1

@dataclass
class TSOptConfig:
    software: str = "pysisyphus"
    lot: str = "xtb"
    n_cpus: int = 1

@dataclass
class IRCValConfig:
    software: str = "pysisyphus"
    lot: str = "xtb"
    n_cpus: int = 1

@dataclass
class TaskDef:
    """A specific executable step within a stage and its execution prerequisites."""
    task_id: str             # Unique identifier (e.g., 'll_path.gsm')
    task_type: str           # The YARP functional type (e.g., 'gsm', 'reactant_conformer')
    parent_stage: str        # The arbitrary user-defined stage name (e.g., 'll_path')
    depends_on: List[str]    # List of prerequisite task_ids that must complete first
    config: Any              # The specific dataclass configuration for this task
    requires_data: List[str] = field(default_factory=list) # e.g., ["ts_guess"]
    provides_data: List[str] = field(default_factory=list) # e.g., ["ts_guess", "ts_opt"]

@dataclass
class StageConfig:
    name: str
    method: str
    tasks: Dict[str, TaskDef] = field(default_factory=dict)

# --- MAIN PARSER CLASS ---
# This class handles the messy logic of converting the YAML dict 
# into the clean config objects above.

class InputParser:
    """
    Parses the raw input dictionary and organizes settings into specific Config objects.
    """
    def __init__(self, file_dict: dict):
        # ---------------------------------------------------------
        # 1. Parse Initialize Node
        # ---------------------------------------------------------
        initnode = file_dict.get('initialize', None)
        if not initnode:
            raise RuntimeError("Hey bro beans, I need some molecules or reactions to work with. "
                               "Missing `initialize` node in YAML file.")

        # Core level inputs
        self.d0_node = initnode.get("initial species", None)
        if not self.d0_node:
            raise RuntimeError("Please provide an initial species for enumeration.")
        self.out_file = initnode.get("output", "YARP_RXNS.pkl")

        # Job manager configuration
        jm_node = initnode.get("job manager", {})
        self.scheduler = jm_node.get("scheduler", "slurm")
        self.container = jm_node.get("container", "docker")
        self.max_active_jobs = jm_node.get("max active jobs", 100) # Default to a safe large number

        # Enumeration configs
        enum_node = initnode.get("enumeration", {})
        self.enum = self._parse_enum_config(enum_node)
        self.enum_filters = self._parse_enum_filters(enum_node)
        self.net_explore = self._parse_network_config(enum_node)

        # Process stages node
        self.stage_names = file_dict.get('stages', [])
        self.stage_configs: Dict[str, StageConfig] = {}

        # A global, flat dictionary of all executable tasks across all stages.
        # This acts as the master "To-Do List" template for any reaction.
        self.pipeline_tasks: Dict[str, TaskDef] = {}

        # A running ledger of what data will be generated by the pipeline, 
        # and which task_id provides it.
        promised_data = {}

        for name in self.stage_names:
            stage_data = file_dict.get(name, {})
            stage_config = self._parse_stage(name, stage_data)
            self.stage_configs[name] = stage_config

            for task_id, task_def in stage_config.tasks.items():
                # DYNAMIC LINKING: Check if this task requires data promised by an earlier task
                for req in task_def.requires_data:
                    if req in promised_data:
                        # Add the execution dependency automatically!
                        task_def.depends_on.append(promised_data[req])

                # Register the data this task will provide to downstream tasks
                for prov in task_def.provides_data:
                    promised_data[prov] = task_id

                self.pipeline_tasks[task_id] = task_def

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

    def _parse_enum_config(self, enum_node: dict) -> EnumerationConfig:
        """Extracts enumeration settings and returns a clean EnumerationConfig object."""

        # Handle the reactive atoms list-to-set conversion
        raw_react = enum_node.get("reactive atoms", None)
        react_atoms_processed = []
        if raw_react:
            react_atoms_processed = [set(raw_react)]

        return EnumerationConfig(
            enumerate=enum_node.get("enumerate", False),
            mode=enum_node.get("mode", "concerted"),
            n_break=enum_node.get("bonds to break", 2),
            n_form=enum_node.get("bonds to form", 2),
            react_atoms=react_atoms_processed,
        )

    def _parse_enum_filters(self, enum_node: dict) -> EnumFilterConfig:
        """Extracts enumeration filtering settings and returns a clean EnumFilterConfig object."""

        # Handle complex "separate products" logic using a helper method
        separate_prods = self._parse_separate_prods(enum_node.get("separate products"))

        # Handle nested filters
        filters = enum_node.get('enumeration filters', {})
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

    def _parse_network_config(self, enum_node: dict) -> NetworkConfig:
        """Extracts network exploration settings and returns a clean NetworkConfig object."""

        # Handle nested filters
        netconfig = enum_node.get('network exploration', {})
        # If netconfig is None (yaml key exists but is empty), treat as empty dict
        if netconfig is None: 
            netconfig = {}

        target = netconfig.get("target product", None)
        if target is not None:
            target_yp = yarpecule(target)
            target_yp.get_inchi()
            target_yp.get_smiles()
        else:
            target_yp = None

        return NetworkConfig(
            target_product=target_yp,
            distance=netconfig.get("distance metric", 'soergel'),
            mode=netconfig.get("mode", 'capped'),
            n_nodes=netconfig.get("n_nodes", 1),
            tolerance=netconfig.get("tie window", 0.0),
            cap=netconfig.get("cutoff", "moderate")
        )

    def _parse_stage(self, name: str, data: dict) -> StageConfig:
        method = data.get('method')
        if not method:
            raise ValueError(f"Stage '{name}' is missing the required 'method' key.")

        config = StageConfig(name=name, method=method)

        if method == 'init_rxn_path':
            conf_data = data.get('conformers', {})
            conf_cfg = ConformerConfig(**{k: v for k, v in conf_data.items() if k in ConformerConfig.__dataclass_fields__})

            gsm_data = data.get('gsm', {})
            gsm_cfg = GSMConfig(**{k: v for k, v in gsm_data.items() if k in GSMConfig.__dataclass_fields__})

            # Define Unique Task IDs
            r_conf_id = f"{name}.reactant_conformer"
            p_conf_id = f"{name}.product_conformer"
            gsm_id = f"{name}.gsm"

            # Create Tasks and Map Dependencies
            config.tasks[r_conf_id] = TaskDef(
                task_id=r_conf_id,
                task_type="reactant_conformer",
                parent_stage=name,
                depends_on=[],
                config=conf_cfg,
                provides_data=["reactant_conf"]
            )

            config.tasks[p_conf_id] = TaskDef(
                task_id=p_conf_id,
                task_type="product_conformer",
                parent_stage=name,
                depends_on=[],
                config=conf_cfg,
                provides_data=["product_conf"]
            )

            config.tasks[gsm_id] = TaskDef(
                task_id=gsm_id, 
                task_type="gsm", 
                parent_stage=name, 
                depends_on=[r_conf_id, p_conf_id],
                config=gsm_cfg,
                requires_data=["reactant_conf", "product_conf"],
                provides_data=["ts_guess"]
            )

        elif method == 'refine_rxn_path':
            rp_data = data.get('rp_opt', {})
            rp_cfg = RPOptConfig(software=rp_data.get('software', 'pysisyphus'), lot=rp_data.get('lot', 'xtb'), n_cpus=rp_data.get('n_cpus', 1))

            ts_data = data.get('ts_opt', {})
            ts_cfg = TSOptConfig(software=ts_data.get('software', 'pysisyphus'), lot=ts_data.get('lot', 'xtb'), n_cpus=ts_data.get('n_cpus', 1))

            irc_data = data.get('irc_val', {})
            irc_cfg = IRCValConfig(software=irc_data.get('software', 'pysisyphus'), lot=irc_data.get('lot', 'xtb'), n_cpus=irc_data.get('n_cpus', 1))

            # Define Unique Task IDs
            r_opt_id = f"{name}.reactant_optimization"
            p_opt_id = f"{name}.product_optimization"
            ts_opt_id = f"{name}.transition_state_optimization"
            irc_id = f"{name}.irc_validation"

            # Create Tasks and Map Dependencies
            config.tasks[r_opt_id] = TaskDef(
                task_id=r_opt_id,
                task_type="reactant_optimization",
                parent_stage=name,
                depends_on=[],
                config=rp_cfg,
                requires_data=["reactant_conf"], # Needs a guess to run!
            )

            config.tasks[p_opt_id] = TaskDef(
                task_id=p_opt_id,
                task_type="product_optimization",
                parent_stage=name,
                depends_on=[],
                config=rp_cfg,
                requires_data=["product_conf"], # Needs a guess to run!
            )

            config.tasks[ts_opt_id] = TaskDef(
                task_id=ts_opt_id, 
                task_type="transition_state_optimization", 
                parent_stage=name, 
                depends_on=[], # Might be populated dynamically in __init__
                config=ts_cfg,
                requires_data=["ts_guess"], # Needs a guess to run!
                provides_data=["ts_opt"]
            )

            config.tasks[irc_id] = TaskDef(
                task_id=irc_id, 
                task_type="irc_validation", 
                parent_stage=name, 
                depends_on=[ts_opt_id], 
                config=irc_cfg,
                requires_data=["ts_opt"] # Needs an optimized TS to run!
            )

        elif method == 'ml_predict':
            # ML acts as a single standalone block without dependencies
            ml_id = f"{name}.ml_predict"
            config.tasks[ml_id] = TaskDef(task_id=ml_id, task_type="ml_predict", parent_stage=name, depends_on=[], config={"model": data.get('model', 'egat_pretrain')})

        return config
