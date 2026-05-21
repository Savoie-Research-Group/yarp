"""
Definition of input object class
"""
import os

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
from yarp.yarpecule.yarpecule import yarpecule


# --- CONFIGURATION OBJECTS ---
# These classes act as simple containers for user provided settings.
@dataclass
class InitalStructConfig:
    """Holds settings for reading in initial species/reactions data."""
    source: str = None
    type: str = None
    mode: str = None

    def __post_init__(self):
        if self.type not in ["smiles", "xyz", "yarp_pickle"]:
            raise ValueError(f"Invalid 'type' provided in 'initial_species': {self.type}. Valid options are 'smiles', 'xyz', or 'yarp_pickle'")
        if self.mode not in ["species", "reaction"]:
            raise ValueError(f"Invalid 'mode' provided in 'initial_species': {self.mode}. Valid options are 'species' or 'reaction'")

        if self.type == 'yarp_pickle' and self.mode != 'reaction':
            raise ValueError(f"Only 'mode = reaction' is valid for 'type = yarp_pickle'")

@dataclass
class JobManagerConfig:
    """Holds settings for job scheduling and container execution."""
    scheduler: str = "local"
    container: str = "docker"
    sif_location: Optional[str] = None
    module_container: Optional[str] = None
    max_active_jobs: int = 100
    queue: Optional[str] = None
    job_name: str = "yarp"
    account: Optional[str] = None  # Slurm/SGE billing account (e.g. #SBATCH -A on Anvil)

    def __post_init__(self):
        # Normalize inputs for easier checking
        if self.scheduler is not None:
            self.scheduler = self.scheduler.lower()
        if self.container is not None:
            self.container = self.container.lower()

        # Sanity Check 1: Schedulers that require a queue
        if self.scheduler in ["sge", "slurm"] and not self.queue:
            raise ValueError(f"Sanity Check Failed: A 'queue' must be specified when using the '{self.scheduler}' scheduler.")

        # Sanity Check 2: Apptainer/Singularity environments
        if self.container == "apptainer" and not self.sif_location:
            # Dynamically resolve the path relative to this file
            # __file__       == base_git_repo/yarp/util/input.py
            # .resolve()     == converts to absolute path resolving any symlinks
            # .parents[0]    == base_git_repo/yarp/util
            # .parents[1]    == base_git_repo/yarp
            # .parents[2]    == base_git_repo
            base_repo_path = Path(__file__).resolve().parents[2]
            
            self.sif_location = str(base_repo_path / "containers")

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
class PreProcessFilterConfig:
    """Holds settings for filtering out reactions before downstream reaction characterization steps."""
    property: str
    source: str
    threshold: float

@dataclass
class GeomSourceConfig:
    label: str
    lot: str
    software: str

@dataclass
class InitialGeomConfig:
    reactant: GeomSourceConfig
    product: GeomSourceConfig
    transition_state: GeomSourceConfig

@dataclass
class MLPropConfig:
    """Holds settings for global ML reaction property predictions."""
    model: str
    n_cpus: int = 1
    mem_per_cpu: int = 1000
    max_runtime: str = "01:00:00"

@dataclass
class ConformerConfig:
    """Holds settings specific to generating a reactant/product conformers"""
    software: str = "crest"
    lot: str = "gfn2"
    n_cpus: int = 1
    mem_per_cpu: int = 1000
    max_runtime: str = "01:00:00"
    energy_window: float = 6.0
    solvent: Optional[Dict[str, str]] = None
    charge: int = 0
    n_unpaired_electrons: int = 0

@dataclass
class GSMConfig:
    """Holds settings specific to generating transition state guesses via GSM"""
    software: str = "pysisyphus"
    selector: str = "rich"
    joint_opt: str = "dual"
    n_conf: int = 1
    n_cpus: int = 1
    mem_per_cpu: int = 1000
    max_runtime: str = "01:00:00"
    bias_lot: str = "uff"
    gsm_lot: str = "xtb"
    max_gsm_nodes: int = 30
    charge: int = 0
    multiplicity: int = 1

@dataclass
class RPOptConfig:
    """Holds settings specific to optimizing reactant/product conformers"""
    software: str = "pysisyphus"
    lot: str = "xtb"
    n_cpus: int = 1
    mem_per_cpu: int = 1000
    max_runtime: str = "01:00:00"
    hessian_recalc: int = 3
    max_cycles: int = 300
    charge: int = 0
    multiplicity: int = 1
    initial_geom: Optional[InitialGeomConfig] = None

@dataclass
class TSOptConfig:
    """Holds settings specific to optimizing transition state conformers"""
    software: str = "pysisyphus"
    lot: str = "xtb"
    n_cpus: int = 1
    mem_per_cpu: int = 1000
    max_runtime: str = "01:00:00"
    hessian_recalc: int = 3
    max_cycles: int = 300
    conv_thresh: str = 'gau'
    charge: int = 0
    multiplicity: int = 1
    initial_geom: Optional[InitialGeomConfig] = None

@dataclass
class IRCValConfig:
    """Holds settings specific to validating transition states with IRC"""
    software: str = "pysisyphus"
    lot: str = "xtb"
    n_cpus: int = 1
    mem_per_cpu: int = 1000
    max_runtime: str = "01:00:00"
    max_cycles: int = 300
    conv_thresh: str = 'gau'
    charge: int = 0
    multiplicity: int = 1

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

        # Control of output generation
        self.out_file = initnode.get("output", "YARP_RXNS.pkl")
        if os.path.splitext(self.out_file)[1].lower() != '.pkl':
            raise ValueError(f"'output' must be a .pkl file, got: '{self.out_file}'")

        self.status_file = initnode.get("status", "STATUS.json")
        if os.path.splitext(self.status_file)[1].lower() != '.json':
            raise ValueError(f"'status' must be a .json file, got: '{self.status_file}'")

        self.verbose = initnode.get("verbose", False) # bool, initialize_yarp only
        if self.verbose not in [True, False]:
            raise ValueError(f"Acceptable inputs for 'verbose' are 'True' or 'False', got: '{self.verbose}'")

        # Reading in intial structure controls
        init_struct_node = initnode.get("initial_structure", {})
        if init_struct_node == {}:
            raise RuntimeError("Missing required block! 'initial_structure' must be provided!")
        self.init_struct = self._parse_init_struct(init_struct_node)

        # Job manager configuration
        jm_node = initnode.get("job_manager", {})
        self.job_manager = self._parse_job_manager(jm_node)

        # Enumeration configs
        enum_node = initnode.get("enumeration", {})
        if self.init_struct.mode == 'species' and enum_node == {}:
            raise RuntimeError("Invalid input configuration! Enumeration must be turned on if starting from a 'species' rather than a 'reaction'!")
        self.enum = self._parse_enum_config(enum_node)
        self.enum_filters = self._parse_enum_filters(enum_node)
        self.net_explore = self._parse_network_config(enum_node)

        # ---------------------------------------------------------
        # 2. Process Stages Node(s)
        # ---------------------------------------------------------
        self.stage_names = file_dict.get('stages', [])
        self.stage_configs: Dict[str, StageConfig] = {}

        # A global, flat dictionary of all executable tasks across all stages.
        # This acts as the master "To-Do List" template for any reaction.
        self.pipeline_tasks: Dict[str, TaskDef] = {}
        self.global_tasks: Dict[str, TaskDef] = {}

        # Track the global ML task ID across stages
        # Also track requested pre-processing filters before a given stage
        self.ml_task_id = None
        self.stage_filters = {}

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
        
        # Dynamically link inter-stage dependencies for path refinement
        self._link_refine_dependencies()

    def _parse_init_struct(self, init_struct: dict) -> InitalStructConfig:
         # 1. Normalize keys (spaces to underscores) and drop None values.
        # Dropping None ensures we don't accidentally overwrite a dataclass default.
        kwargs = {
            key.replace(" ", "_"): value
            for key, value in init_struct.items()
            if value is not None
        }
        # 2. Unpack the clean dictionary into the dataclass
        try:
            return InitalStructConfig(**kwargs)
        except TypeError as e:
            # Python's dataclass automatically raises a TypeError for two reasons:
            # A) A required field (one without a default) is missing.
            # B) An unexpected/unrecognized key was provided (e.g., a typo in the YAML).
            raise ValueError(f"Invalid 'job_manager' configuration in YAML: {e}")

    def _parse_job_manager(self, jm_node: dict) -> JobManagerConfig:
        """Extracts job manager settings and returns a clean JobManagerConfig object."""
        # If the user omitted the block entirely, use all dataclass defaults
        if not jm_node:
            return JobManagerConfig()

        # 1. Normalize keys (spaces to underscores) and drop None values.
        # Dropping None ensures we don't accidentally overwrite a dataclass default.
        kwargs = {
            key.replace(" ", "_"): value
            for key, value in jm_node.items()
            if value is not None
        }

        # 2. Unpack the clean dictionary into the dataclass
        try:
            return JobManagerConfig(**kwargs)
        except TypeError as e:
            # Python's dataclass automatically raises a TypeError for two reasons:
            # A) A required field (one without a default) is missing.
            # B) An unexpected/unrecognized key was provided (e.g., a typo in the YAML).
            raise ValueError(f"Invalid 'job_manager' configuration in YAML: {e}")

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

        if method == "ml_rxn_prop":
            # Save the ID to the instance so downstream stages can see it
            self.ml_task_id = f"{name}.ml_predict" 
            
            ml_cfg = MLPropConfig(
                model=data.get("model"),
                n_cpus=data.get("n_cpus", 1),
                mem_per_cpu=data.get("mem_per_cpu", 1000),
                max_runtime=data.get("max_runtime", "01:00:00")
            )
            
            # Add to global_tasks instead of pipeline_tasks
            self.global_tasks[self.ml_task_id] = TaskDef(
                task_id=self.ml_task_id, 
                task_type="ml_predict", 
                parent_stage=name, 
                depends_on=[], 
                config=ml_cfg,
                requires_data=[]
            )

        elif method == 'init_rxn_path':
            pre_filter_node = data.get("pre_process_filtering")
            pre_filter_node = data.get("pre_process_filtering")
            if pre_filter_node:
                self.stage_filters[name] = PreProcessFilterConfig(
                    property=pre_filter_node.get("property"),
                    source=pre_filter_node.get("source"),
                    threshold=float(pre_filter_node.get("threshold"))
                )

            conf_data = data.get('conformers', {})
            conf_cfg = ConformerConfig(**{k: v for k, v in conf_data.items() if k in ConformerConfig.__dataclass_fields__})

            gsm_data = data.get('gsm', {})
            gsm_cfg = GSMConfig(**{k: v for k, v in gsm_data.items() if k in GSMConfig.__dataclass_fields__})

            # Define Unique Task IDs
            r_conf_id = f"{name}.reactant_conformer"
            p_conf_id = f"{name}.product_conformer"
            gsm_id = f"{name}.gsm"

            # Determine initial dependencies: Wait for ML if it exists!
            initial_deps = [self.ml_task_id] if self.ml_task_id else []

            # Create Tasks and Map Dependencies
            config.tasks[r_conf_id] = TaskDef(
                task_id=r_conf_id,
                task_type="reactant_conformer",
                parent_stage=name,
                depends_on=initial_deps,
                config=conf_cfg,
                provides_data=["reactant_conf"]
            )

            config.tasks[p_conf_id] = TaskDef(
                task_id=p_conf_id,
                task_type="product_conformer",
                parent_stage=name,
                depends_on=initial_deps,
                config=conf_cfg,
                provides_data=["product_conf"]
            )

            config.tasks[gsm_id] = TaskDef(
                task_id=gsm_id, 
                task_type="gsm", 
                parent_stage=name, 
                depends_on=[r_conf_id, p_conf_id],
                config=gsm_cfg,
                requires_data=["reactant_conf", "product_conf"], # Needs 2 starting nodes to run!
                provides_data=["ts_guess"]
            )

        elif method == 'refine_rxn_path':
            ig_node = data.get("initial_geom")
            if not ig_node:
                raise ValueError(f"Stage '{name}' uses 'refine_rxn_path' but is missing the required 'initial_geom' block.")

            ig_config = InitialGeomConfig(
                reactant=GeomSourceConfig(**ig_node.get("reactant", {})),
                product=GeomSourceConfig(**ig_node.get("product", {})),
                transition_state=GeomSourceConfig(**ig_node.get("transition_state", {}))
            )

            rp_data = data.get('rp_opt', {})
            rp_cfg = RPOptConfig(**{k: v for k, v in rp_data.items() if k in RPOptConfig.__dataclass_fields__})
            rp_cfg.initial_geom = ig_config

            ts_data = data.get('ts_opt', {})
            ts_cfg = TSOptConfig(**{k: v for k, v in ts_data.items() if k in TSOptConfig.__dataclass_fields__})
            ts_cfg.initial_geom = ig_config

            irc_data = data.get('irc_val', {})
            irc_cfg = IRCValConfig(**{k: v for k, v in irc_data.items() if k in IRCValConfig.__dataclass_fields__})

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
                depends_on=[], # Populated dynamically in __init__ via _link_refine_dependencies
                config=rp_cfg,
                requires_data=[], # Explicitly removed! (ERM: Wonder if this will break "refine only" workflows...)
                provides_data=["reactant_opt"]
            )

            config.tasks[p_opt_id] = TaskDef(
                task_id=p_opt_id,
                task_type="product_optimization",
                parent_stage=name,
                depends_on=[], # Populated dynamically in __init__ via _link_refine_dependencies
                config=rp_cfg,
                requires_data=[], # Explicitly removed! (ERM: Wonder if this will break "refine only" workflows...)
                provides_data=["product_opt"]
            )

            config.tasks[ts_opt_id] = TaskDef(
                task_id=ts_opt_id, 
                task_type="transition_state_optimization", 
                parent_stage=name,
                depends_on=[], # Populated dynamically in __init__ via _link_refine_dependencies
                config=ts_cfg,
                requires_data=[], # Explicitly removed! (ERM: Wonder if this will break "refine only" workflows...)
                provides_data=["ts_opt"]
            )

            config.tasks[irc_id] = TaskDef(
                task_id=irc_id, 
                task_type="irc_validation", 
                parent_stage=name, 
                depends_on=[ts_opt_id],
                config=irc_cfg,
                requires_data=["ts_opt", "reactant_opt", "product_opt"] # IRC does barrier calculations, so needs all 3 prior stages
            )

        return config

    def _link_refine_dependencies(self):
        """
        Scans pipeline tasks for 'refine_rxn_path' tasks. Looks backward to find the 
        exact task that generated the requested initial_geom, and injects the dependency.
        Raises a fail-fast error if the requested geometry source cannot be found.
        """
        for task_id, task_def in self.pipeline_tasks.items():
            
            # Check if this task has an initial_geom requirement attached
            ig = getattr(task_def.config, "initial_geom", None)
            if not ig:
                continue

            # Determine what type of task we are hunting for
            if task_def.task_type == "reactant_optimization":
                source = ig.reactant
                target_type = "reactant_conformer" if source.label == "conf_gen" else "reactant_optimization"
                
            elif task_def.task_type == "product_optimization":
                source = ig.product
                target_type = "product_conformer" if source.label == "conf_gen" else "product_optimization"
                
            elif task_def.task_type == "transition_state_optimization":
                source = ig.transition_state
                # CRITICAL: If the user requests 'ts_opt', we must wait for the IRC task 
                # of that refinement layer, because IRC produces the validated_ts dict!
                target_type = "gsm" if source.label == "ts_guess" else "irc_validation"
                
            else:
                continue # Skip IRC tasks (they link statically to ts_opt)

            # Search BACKWARD through the pipeline to find the matching task
            match_found = False
            for prev_id, prev_task in self.pipeline_tasks.items():
                if prev_id == task_id:
                    break # Stop looking once we reach ourselves

                if prev_task.task_type == target_type:
                    # Check LOT and Software match
                    if target_type == "gsm":
                        prev_lot = getattr(prev_task.config, "gsm_lot", "").lower()
                    else:
                        prev_lot = getattr(prev_task.config, "lot", "").lower()
                    prev_soft = getattr(prev_task.config, "software", "").lower()

                    if prev_lot == source.lot.lower() and prev_soft == source.software.lower():
                        task_def.depends_on.append(prev_id)
                        match_found = True
                        break # Successfully linked!

            if not match_found:
                raise ValueError(
                    f"\n[FAIL-FAST] Incompatible Workflow Detected!\n"
                    f"Task '{task_id}' requested an initial geometry with:\n"
                    f"  - Label:    {source.label}\n"
                    f"  - LOT:      {source.lot}\n"
                    f"  - Software: {source.software}\n"
                    f"...but no preceding task in the pipeline matches these criteria."
                )