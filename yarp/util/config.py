
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict
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

    def __post_init__(self):
        if not self.label or not isinstance(self.label, str):
            raise ValueError("initial_geom source must include a non-empty 'label'")
        if not self.lot or not isinstance(self.lot, str):
            raise ValueError("initial_geom source must include a non-empty 'lot'")
        if not self.software or not isinstance(self.software, str):
            raise ValueError("initial_geom source must include a non-empty 'software'")

@dataclass
class InitialGeomConfig:
    reactant: GeomSourceConfig
    product: GeomSourceConfig
    transition_state: GeomSourceConfig

    def __post_init__(self):
        if self.reactant.label not in ['conf_gen', 'rp_opt']:
            raise ValueError(f"Invalid 'label' for initial_geom.reactant: '{self.reactant.label}'; valid options are 'conf_gen', 'rp_opt'")
        if self.product.label not in ['conf_gen', 'rp_opt']:
            raise ValueError(f"Invalid 'label' for initial_geom.product: '{self.product.label}'; valid options are 'conf_gen', 'rp_opt'")
        if self.transition_state.label not in ['ts_guess', 'ts_opt']:
            raise ValueError(f"Invalid 'label' for initial_geom.transition_state: '{self.transition_state.label}'; valid options are 'ts_guess', 'ts_opt'")

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
        if self.n_cpus < 8:
            raise ValueError(f"EGAT requires 8 CPUs in order to run safely! Number selected: {self.n_cpus}")
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
    software: str = None
    lot: str = None
    charge: int = None
    multiplicity: int = None
    hessian_recalc: int = 3
    max_cycles: int = 300
    initial_geom: Optional[InitialGeomConfig] = None
    # ERM: To-do -> put in convergence threshold and solvent options?

    n_cpus: int = 1
    mem_per_cpu: int = 1000
    max_runtime: str = "01:00:00"

    def __post_init__(self):
        if self.charge == None:
            raise ValueError("Missing required key! Please provide 'charge' in rp_opt block!")
        if self.multiplicity == None:
            raise ValueError("Missing required key! Please provide 'multiplicity' in rp_opt block!")
        if self.software not in ['pysisyphus', 'orca']:
            raise ValueError(f"Invalid rp_opt.'software' provided: '{self.software}'; valid options: 'pysisyphus', 'orca'")
        if self.software == 'pysisyphus' and self.lot not in ['xtb']:
            raise ValueError(f"Invalid rp_opt.'lot' for Pysisyphus software! Valid options: 'xtb'")
        # ERM: To-do -> add a valid input check function for ORCA keyword block?

        if not isinstance(self.hessian_recalc, int):
            raise ValueError("Please provide an integer value to rp_opt: 'hessian_recalc'")
        if not isinstance(self.max_cycles, int):
            raise ValueError("Please provide an integer value to rp_opt: 'max_cycles'")
        if not isinstance(self.n_cpus, int):
            raise ValueError("Please provide an integer value to rp_opt: 'n_cpus'")
        if not isinstance(self.mem_per_cpu, int):
            raise ValueError("Please provide an integer value (in MB) to rp_opt: 'mem_per_cpu'")
        if not is_valid_time_format(self.max_runtime):
            raise ValueError("Please provide rp_opt: 'max_runtime' time in HH:MM:SS!")


@dataclass
class TSOptConfig:
    """Holds settings specific to optimizing transition state conformers"""
    software: str = None
    lot: str = None
    charge: int = None
    multiplicity: int = None
    hessian_recalc: int = 3
    max_cycles: int = 300
    conv_thresh: str = 'gau' # ERM: only used for xTB right now...
    initial_geom: Optional[InitialGeomConfig] = None

    n_cpus: int = 1
    mem_per_cpu: int = 1000
    max_runtime: str = "01:00:00"

    def __post_init__(self):
        if self.charge == None:
            raise ValueError("Missing required key! Please provide 'charge' in ts_opt block!")
        if self.multiplicity == None:
            raise ValueError("Missing required key! Please provide 'multiplicity' in ts_opt block!")
        if self.software not in ['pysisyphus', 'orca']:
            raise ValueError(f"Invalid ts_opt.'software' provided: '{self.software}'; valid options: 'pysisyphus', 'orca'")
        if self.software == 'pysisyphus' and self.lot not in ['xtb']:
            raise ValueError(f"Invalid ts_opt.'lot' for Pysisyphus software! Valid options: 'xtb'")
        # ERM: To-do -> add a valid input check function for ORCA keyword block?
        # ERM: To-do -> look up valid inputs for conv_thresh in xtb!

        if not isinstance(self.hessian_recalc, int):
            raise ValueError("Please provide an integer value to ts_opt: 'hessian_recalc'")
        if not isinstance(self.max_cycles, int):
            raise ValueError("Please provide an integer value to ts_opt: 'max_cycles'")
        if not isinstance(self.n_cpus, int):
            raise ValueError("Please provide an integer value to ts_opt: 'n_cpus'")
        if not isinstance(self.mem_per_cpu, int):
            raise ValueError("Please provide an integer value (in MB) to ts_opt: 'mem_per_cpu'")
        if not is_valid_time_format(self.max_runtime):
            raise ValueError("Please provide ts_opt: 'max_runtime' time in HH:MM:SS!")


@dataclass
class IRCValConfig:
    """Holds settings specific to validating transition states with IRC"""
    software: str = None
    lot: str = None
    charge: int = None
    multiplicity: int = None
    max_cycles: int = 300
    conv_thresh: str = 'gau' # ERM: only used for xTB right now...

    n_cpus: int = 1
    mem_per_cpu: int = 1000
    max_runtime: str = "01:00:00"

    def __post_init__(self):
        if self.charge == None:
            raise ValueError("Missing required key! Please provide 'charge' in irc_val block!")
        if self.multiplicity == None:
            raise ValueError("Missing required key! Please provide 'multiplicity' in irc_val block!")
        if self.software not in ['pysisyphus', 'orca']:
            raise ValueError(f"Invalid irc_val.'software' provided: '{self.software}'; valid options: 'pysisyphus', 'orca'")
        if self.software == 'pysisyphus' and self.lot not in ['xtb']:
            raise ValueError(f"Invalid irc_val.'lot' for Pysisyphus software! Valid options: 'xtb'")
        # ERM: To-do -> add a valid input check function for ORCA keyword block?
        # ERM: To-do -> look up valid inputs for conv_thresh in xtb!

        if not isinstance(self.max_cycles, int):
            raise ValueError("Please provide an integer value to irc_val: 'max_cycles'")
        if not isinstance(self.n_cpus, int):
            raise ValueError("Please provide an integer value to irc_val: 'n_cpus'")
        if not isinstance(self.mem_per_cpu, int):
            raise ValueError("Please provide an integer value (in MB) to irc_val: 'mem_per_cpu'")
        if not is_valid_time_format(self.max_runtime):
            raise ValueError("Please provide irc_val: 'max_runtime' time in HH:MM:SS!")