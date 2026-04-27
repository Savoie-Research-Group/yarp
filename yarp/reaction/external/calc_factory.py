from yarp.reaction.external.calc_base import AsyncYarpCalculator
from yarp.reaction.external.ml_predict import EgatMLPredict
from yarp.reaction.external.conf_gen import CrestConfCalculator
from yarp.reaction.external.ts_guess import PysisyphusTSGuessCalculator
from yarp.reaction.external.min_opt import PysisyphusMinOptCalculator, OrcaMinOptCalculator
from yarp.reaction.external.ts_opt import PysisyphusTSOptCalculator, OrcaTSOptCalculator
from yarp.reaction.external.irc_val import PysisyphusIRCValCalculator, OrcaIRCValCalculator

def get_calculator(task_def, rxn_data, job_config) -> AsyncYarpCalculator:
    """
    Routes the task to the specific combination of Task Type and Software.
    """
    t_type = task_def.task_type
    software = getattr(task_def.config, 'software', 'unknown').lower()
    model = getattr(task_def.config, 'model', 'unknown').lower()

    # Task 1: ML Predict
    if t_type == "ml_predict":
        if model == "egat_rgd1":
            return EgatMLPredict(task_def, rxn_data, job_config)

    # Task 2: Conformers
    elif t_type in ["reactant_conformer", "product_conformer"]:
        if software == "crest":
            return CrestConfCalculator(task_def, rxn_data, job_config)

    # Task 3: TS Guess
    elif t_type == "gsm":
        if software == "pysisyphus":
            return PysisyphusTSGuessCalculator(task_def, rxn_data, job_config)

    # Tasks 4 & 5: Optimizations
    elif t_type in ["reactant_optimization", "product_optimization", "transition_state_optimization"]:
        if software == "pysisyphus":
            if t_type in ["reactant_optimization", "product_optimization"]:
                return PysisyphusMinOptCalculator(task_def, rxn_data, job_config)
            elif t_type == "transition_state_optimization":
                return PysisyphusTSOptCalculator(task_def, rxn_data, job_config)
        elif software == "orca":
            if t_type in ["reactant_optimization", "product_optimization"]:
                return OrcaMinOptCalculator(task_def, rxn_data, job_config)
            elif t_type == "transition_state_optimization":
                return OrcaTSOptCalculator(task_def, rxn_data, job_config)
        else:
            pass

    # Task 6: IRC Validation
    elif t_type == "irc_validation":
        if software == "pysisyphus":
            return PysisyphusIRCValCalculator(task_def, rxn_data, job_config)
        elif software == "orca":
            return OrcaIRCValCalculator(task_def, rxn_data, job_config)
        else:
            pass

    raise ValueError(f"No calculator implemented for Task: '{t_type}' with Software: '{software}'")