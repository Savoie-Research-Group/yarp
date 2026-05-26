"""
Definition of input object class
"""
import os
from datetime import datetime

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Any
from pathlib import Path
from yarp.yarpecule.yarpecule import yarpecule

def is_valid_time_format(time_str):
    try:
        # %H = 24-hour hour, %M = minute, %S = second
        datetime.strptime(time_str, "%H:%M:%S")
        return True
    except ValueError:
        return False

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

        if self.type == 'yarp_pickle':
            if os.path.splitext(self.source)[1].lower() != '.pkl':
                raise ValueError(f"'source' must be a .pkl file, got: '{self.source}'")
        # TO-DO: Put in checks for XYZ and SMILES, which are compatible with Tanveer's changes

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

        # Get the proper location of apptainer containers
        if self.container == "apptainer" and not self.sif_location:
            # Dynamically resolve the path relative to this file
            # __file__       == base_git_repo/yarp/util/input.py
            # .resolve()     == converts to absolute path resolving any symlinks
            # .parents[0]    == base_git_repo/yarp/util
            # .parents[1]    == base_git_repo/yarp
            # .parents[2]    == base_git_repo
            base_repo_path = Path(__file__).resolve().parents[2]
            
            self.sif_location = str(base_repo_path / "containers")

        # Sanity Checks
        if self.scheduler not in ["local", "sge", "slurm"]:
            raise ValueError(f"Invalid 'scheduler' entered: '{self.scheduler}'. Valid options are 'local', 'sge', and 'slurm'")
        if self.container not in ["apptainer", "docker", 'singularity']:
            raise ValueError(f"Invalid 'container' entered: '{self.container}'. Valid options are 'apptainer', 'singularity', and 'docker'")
        if self.scheduler in ["sge", "slurm"] and not self.queue:
            raise ValueError(f"Sanity Check Failed: A 'queue' must be specified when using the '{self.scheduler}' scheduler.")

        if not isinstance(self.max_active_jobs, int):
            raise ValueError("Please provide an integer value to 'max_active_jobs'")
        if self.sif_location and not isinstance(self.sif_location, str):
            raise ValueError("Please provide a valid string value to 'sif_location'")
        if self.module_container and not isinstance(self.sif_location, str):
            raise ValueError("Please provide a valid string value to 'module_container'")

@dataclass
class PropertyFilterConfig:
    type: str
    source: str
    threshold: float

    def __post_init__(self):
        if self.type not in ['barrier', 'reverse_barrier']:
            raise ValueError(f"Invalid property selected: '{self.type}'. Valid options are 'barrier', 'reverse_barrier'")

@dataclass
class ProductBlindersConfig:
    target_product: str = None
    distance_metric: str = "soergel"
    mode: str = "beam"
    n_nodes: int = 1

    def __post_init__(self):
        if not self.target_product:
            raise ValueError("If using product blinders, you must provide a 'target_product'!!!")
        
        try:
            target = yarpecule(self.target_product)
            target.get_inchi()
            target.get_smiles()
            self.target_product = target
        except TypeError as e:
            raise ValueError(f"Invalid entry for 'target_product': {e}")

@dataclass
class PreEnumFilters:
    separate_prods: bool = False
    property_filter: Optional[PropertyFilterConfig] = None
    product_blinders: Optional[ProductBlindersConfig] = None

@dataclass
class PostEnumFilters:
    lewis_score: float = 0.0
    formal_charge: float = 2.0
    ring_filter: bool = False

@dataclass
class EnumerationConfig:
    ON: bool = False
    mode: str = "concerted"
    n_break: int = 2
    n_form: int = 2
    pre_enum_filters: PreEnumFilters = field(default_factory=PreEnumFilters)
    post_enum_filters: PostEnumFilters = field(default_factory=PostEnumFilters)
    react_atoms: List[set] = field(default_factory=list)

    def __post_init__(self):
        # Clean up reactive atoms input for uniqueness
        react_atoms_processed = []
        if self.react_atoms:
            react_atoms_processed = [set(self.react_atoms)]
            self.react_atoms = react_atoms_processed

        # Sanity checks
        if self.mode not in ["concerted", "sequential"]:
            raise ValueError(f"Invalid enumeration 'mode' entered: '{self.mode}'. Valid options are 'concerted' and 'sequential'")
        if not isinstance(self.n_break, int):
            raise ValueError("Please provide an integer value to 'n_break'")
        if not isinstance(self.n_form, int):
            raise ValueError("Please provide an integer value to 'n_form'")
        # TO-DO: Put in sanity checks related to reactive atoms!

        # User warnings
        if self.mode == 'concerted' and self.n_break != self.n_form:
            print(f"WARNING! Concerted enumeration requires n_break = n_form! Setting n_break = n_form = {self.n_break}")
            self.n_form = self.n_break

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

    n_cpus: int = 8
    mem_per_cpu: int = 1000
    max_runtime: str = "01:00:00"

    def __post_init__(self):
        if not self.model:
            raise ValueError("Missing required key! Please provide 'model' when using ml_rxn_prop method!")
        if self.model not in ['egat_rgd1']:
            raise ValueError(f"Invalid 'model' provided: '{self.model}' Currently, only valid option is 'egat_rdg1'")
        if not isinstance(self.n_cpus, int):
            raise ValueError("Please provide an integer value to ml_rxn_prop -> 'n_cpus'")
        if not isinstance(self.mem_per_cpu, int):
            raise ValueError("Please provide an integer value (in MB) to ml_rxn_prop -> 'mem_per_cpu'")
        if not is_valid_time_format(self.max_runtime):
            raise ValueError("Please provide ml_rxn_prop -> 'max_runtime' time in HH:MM:SS!")

@dataclass
class ConformerConfig:
    """Holds settings specific to generating a reactant/product conformers"""
    software: str = None
    lot: str = None
    charge: int = None
    n_unpaired_electrons: int = None
    energy_window: float = 6.0
    solvent: Optional[Dict[str, str]] = None

    n_cpus: int = 1
    mem_per_cpu: int = 1000
    max_runtime: str = "01:00:00"


    def __post_init__(self):
        if self.charge == None:
            raise ValueError("Missing required key! Please provide 'charge' in conf_gen block!")
        if not self.software:
            raise ValueError("Missing required key! Please provide 'software' in conf_gen block!")
        if self.software not in ['crest']:
            raise ValueError(f"Invalid 'software' provided: '{self.software}' Currently, only option is 'crest'")
        if self.software == 'crest' and self.lot not in ['gfn2', 'gfn1', 'gfnff', 'gfn2//gfnff']:
            raise ValueError(f"Invalid 'lot' for CREST software! Valid options: 'gfn2', 'gfn1', 'gfnff', 'gfn2//gfnff'")
        if self.software == 'crest' and self.n_unpaired_electrons == None:
            raise ValueError(f"Missing required key! 'n_unpaired_electrons' field is required for conf_gen with CREST!")

        if not isinstance(self.charge, int):
            raise ValueError("Please provide an integer value to conf_gen: 'charge'")
        if self.n_unpaired_electrons != None and not isinstance(self.n_unpaired_electrons, int):
            raise ValueError("Please provide an integer value to conf_gen: 'n_unpaired_electrons'")
        if not isinstance(self.energy_window, float):
            raise ValueError("Please provide a float value to conf_gen: 'energy_window'")
        if not isinstance(self.n_cpus, int):
            raise ValueError("Please provide an integer value to conf_gen: 'n_cpus'")
        if not isinstance(self.mem_per_cpu, int):
            raise ValueError("Please provide an integer value (in MB) to conf_gen: 'mem_per_cpu'")
        if not is_valid_time_format(self.max_runtime):
            raise ValueError("Please provide conf_gen: 'max_runtime' time in HH:MM:SS!")


@dataclass
class TSGuessConfig:
    """Holds settings specific to generating transition state guesses via GSM"""
    software: str = None
    gsm_lot: str = None
    charge: int = None
    multiplicity: int = None
    n_conf: int = 1
    max_gsm_nodes: int = 30
    bias_lot: str = "uff"
    joint_opt: str = "dual"

    n_cpus: int = 1
    mem_per_cpu: int = 1000
    max_runtime: str = "01:00:00"

    def __post_init__(self):
        if self.charge == None:
            raise ValueError("Missing required key! Please provide 'charge' in ts_guess block!")
        if self.multiplicity == None:
            raise ValueError("Missing required key! Please provide 'multiplicity' in ts_guess block!")
        if not self.software:
            raise ValueError("Missing required key! Please provide 'software' in ts_guess block!")
        if self.software not in ['pysisyphus']:
            raise ValueError(f"Invalid 'software' provided: '{self.software}' Currently, only option is 'pysisyphus'")
        if self.software == 'pysisyphus' and self.gsm_lot not in ['xtb']:
            raise ValueError(f"Invalid 'lot' for Pysisyphus software! Valid options: 'xtb'")

        if self.bias_lot not in ['uff', 'Ghemical', 'MMFF94']:
            raise ValueError(f"Invalid force field selected for 'bias_lot': '{self.bias_lot}' Valid options are: 'uff', 'Ghemical', 'MMFF94'")
        if self.joint_opt not in ['dual', 'r_only', 'p_only', 'off']:
            raise ValueError(f"Invalid 'joint_opt' entry: '{self.joint_opt}' Valid options are: 'dual', 'r_only', 'p_only', 'off'")

        if not isinstance(self.n_conf, int):
            raise ValueError("Please provide an integer value to ts_guess: 'n_conf'")
        if not isinstance(self.max_gsm_nodes, int):
            raise ValueError("Please provide an integer value to ts_guess: 'max_gsm_nodes'")
        if not isinstance(self.n_cpus, int):
            raise ValueError("Please provide an integer value to ts_guess: 'n_cpus'")
        if not isinstance(self.mem_per_cpu, int):
            raise ValueError("Please provide an integer value (in MB) to ts_guess: 'mem_per_cpu'")
        if not is_valid_time_format(self.max_runtime):
            raise ValueError("Please provide ts_guess: 'max_runtime' time in HH:MM:SS!")


@dataclass
class RPOptConfig:
    """Holds settings specific to optimizing reactant/product conformers"""
    software: str = "pysisyphus"
    lot: str = "xtb"
    hessian_recalc: int = 3
    max_cycles: int = 300
    charge: int = 0
    multiplicity: int = 1
    initial_geom: Optional[InitialGeomConfig] = None

    n_cpus: int = 1
    mem_per_cpu: int = 1000
    max_runtime: str = "01:00:00"


@dataclass
class TSOptConfig:
    """Holds settings specific to optimizing transition state conformers"""
    software: str = "pysisyphus"
    lot: str = "xtb"
    hessian_recalc: int = 3
    max_cycles: int = 300
    conv_thresh: str = 'gau'
    charge: int = 0
    multiplicity: int = 1
    initial_geom: Optional[InitialGeomConfig] = None

    n_cpus: int = 1
    mem_per_cpu: int = 1000
    max_runtime: str = "01:00:00"


@dataclass
class IRCValConfig:
    """Holds settings specific to validating transition states with IRC"""
    software: str = "pysisyphus"
    lot: str = "xtb"
    max_cycles: int = 300
    conv_thresh: str = 'gau'
    charge: int = 0
    multiplicity: int = 1

    n_cpus: int = 1
    mem_per_cpu: int = 1000
    max_runtime: str = "01:00:00"


@dataclass
class TaskDef:
    """A specific executable step within a stage and its execution prerequisites."""
    task_id: str             # Unique identifier (e.g., 'll_path.ts_guess')
    task_type: str           # The YARP functional type (e.g., 'ts_guess', 'reactant_conformer')
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
        self.init_struct = self._parse_init_struct(init_struct_node)

        # Job manager configuration
        jm_node = initnode.get("job_manager", {})
        self.job_manager = self._parse_job_manager(jm_node)

        # Enumeration configs
        enum_node = initnode.get("enumeration", {})
        if enum_node != {}:
            enum_node['ON'] = True
        self.enum = self._parse_enum_config(enum_node)

        if self.init_struct.mode == 'species' and not self.enum.ON:
            raise ValueError("Invalid input configuration! Enumeration must be turned on if starting from a 'species' rather than a 'reaction'!")

        pre_enum_filters_provided = "pre_enum_filters" in enum_node
        if self.init_struct.mode == 'species' and pre_enum_filters_provided:
            print(f"WARNING: When starting from single species, all 'pre_enum_filters' are non-applicable!")

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
        if not init_struct or init_struct == {}:
            raise ValueError("Missing required block! 'initial_structure' must be provided!")

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
            raise ValueError(f"Invalid 'initial_structure' configuration in YAML: {e}")

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
            return JobManagerConfig(**{k: v for k, v in kwargs.items() if k in JobManagerConfig.__dataclass_fields__})
        except TypeError as e:
            # Python's dataclass automatically raises a TypeError for two reasons:
            # A) A required field (one without a default) is missing.
            # B) An unexpected/unrecognized key was provided (e.g., a typo in the YAML).
            raise ValueError(f"Invalid 'job_manager' configuration in YAML: {e}")

    def _parse_enum_config(self, enum_node: dict) -> EnumerationConfig:
        kwargs = {
            key.replace(" ", "_"): value
            for key, value in enum_node.items()
            if value is not None
        }

        pre_node = kwargs.pop("pre_enum_filters", {}) or {}
        post_node = kwargs.pop("post_enum_filters", {}) or {}

        pre_filters = self._parse_pre_enum_filters(pre_node)
        post_filters = self._parse_post_enum_filters(post_node)

        try:
            return EnumerationConfig(
                pre_enum_filters=pre_filters,
                post_enum_filters=post_filters,
                **{k: v for k, v in kwargs.items() if k in EnumerationConfig.__dataclass_fields__}
            )
        except TypeError as e:
            raise ValueError(f"Invalid 'enumeration' configuration in YAML: {e}")

    def _parse_pre_enum_filters(self, pre_node: dict) -> PreEnumFilters:
        if pre_node is None:
            pre_node = {}

        kwargs = {
            key.replace(" ", "_"): value
            for key, value in pre_node.items()
            if value is not None
        }

        property_data = kwargs.pop("property", {}) or {}
        product_blinders_data = kwargs.pop("product_blinders", {}) or {}

        return PreEnumFilters(
            property_filter=PropertyFilterConfig(**property_data) if property_data else None,
            product_blinders=ProductBlindersConfig(**product_blinders_data) if product_blinders_data else None,
            **{k: v for k, v in kwargs.items() if k in PreEnumFilters.__dataclass_fields__}
        )

    def _parse_post_enum_filters(self, post_node: dict) -> PostEnumFilters:
        if post_node is None:
            post_node = {}

        kwargs = {
            key.replace(" ", "_"): value
            for key, value in post_node.items()
            if value is not None
        }

        return PostEnumFilters(**{k: v for k, v in kwargs.items() if k in PostEnumFilters.__dataclass_fields__})

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
            if pre_filter_node:
                self.stage_filters[name] = PropertyFilterConfig(**{k: v for k, v in pre_filter_node.items() if k in PropertyFilterConfig.__dataclass_fields__})

            conf_data = data.get('conf_gen', {})
            conf_cfg = ConformerConfig(**{k: v for k, v in conf_data.items() if k in ConformerConfig.__dataclass_fields__})

            tsg_data = data.get('ts_guess', {})
            tsg_cfg = TSGuessConfig(**{k: v for k, v in tsg_data.items() if k in TSGuessConfig.__dataclass_fields__})

            # Define Unique Task IDs
            r_conf_id = f"{name}.reactant_conformer"
            p_conf_id = f"{name}.product_conformer"
            tsg_id = f"{name}.ts_guess"

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

            config.tasks[tsg_id] = TaskDef(
                task_id=tsg_id, 
                task_type="ts_guess", 
                parent_stage=name, 
                depends_on=[r_conf_id, p_conf_id],
                config=tsg_cfg,
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
                target_type = "ts_guess" if source.label == "ts_guess" else "irc_validation"
                
            else:
                continue # Skip IRC tasks (they link statically to ts_opt)

            # Search BACKWARD through the pipeline to find the matching task
            match_found = False
            for prev_id, prev_task in self.pipeline_tasks.items():
                if prev_id == task_id:
                    break # Stop looking once we reach ourselves

                if prev_task.task_type == target_type:
                    # Check LOT and Software match
                    if target_type == "ts_guess":
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