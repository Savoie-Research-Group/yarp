"""
Definition of input object class
"""
import os
import re

from dataclasses import asdict, dataclass, field
from pprint import pformat
from typing import List, Dict, Any

from yarp.util.config import InitalStructConfig, JobManagerConfig, EnumerationConfig, PreEnumFilters, PropertyFilterConfig, ProductBlindersConfig, PostEnumFilters, MLPropConfig, ConformerConfig, RPOptConfig, TSOptConfig, TSGuessConfig, IRCValConfig, InitialGeomConfig, GeomSourceConfig

def has_atom_maps(smiles: str) -> bool:
    """
    Checks if a SMILES string contains atom mapping.

    Atom maps are formatted as a colon followed by numbers inside 
    square brackets, for example: [C:1] or [O:12].

    Parameters:
    smiles (str): The SMILES string to check.

    Returns:
    bool: True if atom maps are found, False otherwise.
    """
    # Regex breakdown:
    # \[     -> matches the opening square bracket
    # [^\]]+ -> matches one or more characters that are NOT a closing bracket (the element/isotopes)
    # :      -> matches the literal colon used for mapping
    # \d+    -> matches one or more digits (the map number)
    # \]     -> matches the closing square bracket
    atom_map_pattern = r'\[[^\]]+:\d+\]'

    return bool(re.search(atom_map_pattern, smiles))

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
        # Normalize keys before parsing so all downstream code can assume underscores.
        file_dict = self._normalize_keys(file_dict)

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
        if "job_manager" not in initnode:
            raise ValueError("Missing required block! 'job_manager' must be provided!")
        jm_node = initnode.get("job_manager", {})
        self.job_manager = self._parse_job_manager(jm_node)

        # Enumeration configs
        enum_node = initnode.get("enumeration", {})
        if enum_node != {}:
            enum_node['ON'] = True
        self.enum = self._parse_enum_config(enum_node)

        if self.init_struct.mode == 'species' and not self.enum.ON:
            raise ValueError("Invalid input configuration! Enumeration must be turned on if starting from a 'species' rather than a 'reaction'!")

        if self.enum.react_atoms != [] and self.init_struct.mode =='species' and self.init_struct.type == 'smiles' and not has_atom_maps(self.init_struct.source):
            raise ValueError("Invalid input configuration! Reactive atoms require user to provide atom mapped SMILES!")

        pre_enum_filters_provided = "pre_enum_filters" in enum_node
        if self.init_struct.mode == 'species' and pre_enum_filters_provided:
            print(f"WARNING: When starting from single species, all 'pre_enum_filters' are non-applicable!")

        # ---------------------------------------------------------
        # 2. Process Stages Node(s)
        # ---------------------------------------------------------
        self.stage_names = [
            self._normalize_key(name)
            for name in file_dict.get('stages', [])
        ]
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

    def __repr__(self):
        return pformat({
            "output": self.out_file,
            "status": self.status_file,
            "verbose": self.verbose,
            "initial_structure": asdict(self.init_struct),
            "job_manager": asdict(self.job_manager),
            "enumeration": asdict(self.enum),
            "stages": self.stage_names,
            "pipeline_tasks": {
                task_id: self._task_asdict(task_def)
                for task_id, task_def in self.pipeline_tasks.items()
            },
            "global_tasks": {
                task_id: self._task_asdict(task_def)
                for task_id, task_def in self.global_tasks.items()
            },
        })

    def _parse_init_struct(self, init_struct: dict) -> InitalStructConfig:
        if not init_struct or init_struct == {}:
            raise ValueError("Missing required block! 'initial_structure' must be provided!")

        kwargs = init_struct
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

        kwargs = jm_node

        # 2. Unpack the clean dictionary into the dataclass
        try:
            return JobManagerConfig(**{k: v for k, v in kwargs.items() if k in JobManagerConfig.__dataclass_fields__})
        except TypeError as e:
            # Python's dataclass automatically raises a TypeError for two reasons:
            # A) A required field (one without a default) is missing.
            # B) An unexpected/unrecognized key was provided (e.g., a typo in the YAML).
            raise ValueError(f"Invalid 'job_manager' configuration in YAML: {e}")

    def _parse_enum_config(self, enum_node: dict) -> EnumerationConfig:
        kwargs = dict(enum_node)

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

        kwargs = dict(pre_node)

        property_data = kwargs.pop("property_filter", {}) or {}
        product_blinders_data = kwargs.pop("product_blinders", {}) or {}

        return PreEnumFilters(
            property_filter=PropertyFilterConfig(**property_data) if property_data else None,
            product_blinders=ProductBlindersConfig(**product_blinders_data) if product_blinders_data else None,
            **{k: v for k, v in kwargs.items() if k in PreEnumFilters.__dataclass_fields__}
        )

    def _parse_post_enum_filters(self, post_node: dict) -> PostEnumFilters:
        if post_node is None:
            post_node = {}

        return PostEnumFilters(**{k: v for k, v in post_node.items() if k in PostEnumFilters.__dataclass_fields__})

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
            pre_filter_node = data.get("pre_characterize_filters")
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

            ig_config = self._parse_initial_geom(ig_node)

            missing_blocks = [
                block for block in ("rp_opt", "ts_opt", "irc_val")
                if not data.get(block)
            ]
            if missing_blocks:
                raise ValueError(
                    f"Stage '{name}' uses 'refine_rxn_path' but is missing required block(s): "
                    f"{', '.join(missing_blocks)}"
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

    def _parse_initial_geom(self, ig_node: dict) -> InitialGeomConfig:
        if not isinstance(ig_node, dict):
            raise ValueError("'initial_geom' must be a mapping/dictionary")
        required = ["reactant", "product", "transition_state"]
        missing = [k for k in required if k not in ig_node]
        if missing:
            raise ValueError(f"'initial_geom' is missing required entries: {missing}")
        extra = [k for k in ig_node if k not in required]
        if extra:
            raise ValueError(f"Unexpected keys in 'initial_geom': {extra}")

        return InitialGeomConfig(
            reactant=self._parse_geom_source(ig_node["reactant"], "reactant"),
            product=self._parse_geom_source(ig_node["product"], "product"),
            transition_state=self._parse_geom_source(ig_node["transition_state"], "transition_state")
        )

    def _parse_geom_source(self, section: dict, name: str) -> GeomSourceConfig:
        if not isinstance(section, dict):
            raise ValueError(f"'initial_geom.{name}' must be a dictionary")
        try:
            return GeomSourceConfig(**{k: v for k, v in section.items() if k in GeomSourceConfig.__dataclass_fields__})
        except TypeError as e:
            raise ValueError(f"Invalid 'initial_geom.{name}' configuration: {e}")

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

    def _normalize_keys(self, d: dict) -> dict:
        """Helper function to convert keys with spaces to underscores, and drop None values"""
        if isinstance(d, list):
            return [self._normalize_keys(item) for item in d]
        if not isinstance(d, dict):
            return d

        self._duplicate_key_check(d, normalize=True, recurse=False)
        return {
            self._normalize_key(key): self._normalize_keys(value)
            for key, value in d.items()
            if value is not None
        }

    def _normalize_key(self, key):
        return key.replace(" ", "_") if isinstance(key, str) else key

    def _config_asdict(self, config):
        return asdict(config) if hasattr(config, "__dataclass_fields__") else config

    def _task_asdict(self, task_def: TaskDef) -> dict:
        return {
            "task_type": task_def.task_type,
            "parent_stage": task_def.parent_stage,
            "depends_on": task_def.depends_on,
            "config": self._config_asdict(task_def.config),
        }

    def _duplicate_key_check(self, d: dict, path="", normalize=False, recurse=True):
        """Recursively checks for duplicate keys in the input dictionary."""
        if not isinstance(d, dict):
            return d
        seen = set()
        for key, value in d.items():
            check_key = self._normalize_key(key) if normalize else key
            if check_key in seen:
                raise ValueError(f"Duplicate key detected: '{check_key}' at path '{path}'")
            seen.add(check_key)
            if recurse:
                self._duplicate_key_check(value, path + f".{check_key}", normalize=normalize)
        return d
