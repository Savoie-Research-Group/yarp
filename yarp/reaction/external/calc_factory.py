from yarp.reaction.external.calc_base import AsyncYarpCalculator
from yarp.reaction.external.ml_predict import EgatMLPredict
from yarp.reaction.external.conf_gen import CrestConfCalculator
from yarp.reaction.external.ts_guess import PysisyphusTSGuessCalculator
from yarp.reaction.external.min_opt import PysisyphusMinOptCalculator
from yarp.reaction.external.ts_opt import PysisyphusTSOptCalculator

def get_calculator(task_def, rxn_obj, container_runner="docker") -> AsyncYarpCalculator:
    """
    Routes the task to the specific combination of Task Type and Software.
    """
    t_type = task_def.task_type
    software = getattr(task_def.config, 'software', 'unknown').lower()

    # Task 1: ML Predict
    if t_type == "ml_predict":
        if software == "egat" or software == "unknown": # Fallback for now
            return EgatMLPredict(task_def, rxn_obj)

    # Task 2: Conformers
    elif t_type in ["reactant_conformer", "product_conformer"]:
        if software == "crest":
            return CrestConfCalculator(task_def, rxn_obj, container_runner)

    # Task 3: TS Guess
    elif t_type == "gsm":
        if software == "pysisyphus":
            return PysisyphusTSGuessCalculator(task_def, rxn_obj)

    # Tasks 4 & 5: Optimizations
    elif t_type in ["reactant_optimization", "product_optimization", "transition_state_optimization"]:
        if software == "pysisyphus":
            if t_type in ["reactant_optimization", "product_optimization"]:
                return PysisyphusMinOptCalculator(task_def, rxn_obj)
            elif t_type == "transition_state_optimization":
                return PysisyphusTSOptCalculator(task_def, rxn_obj)
        else:
            pass

    # Task 6: IRC Validation
    elif t_type == "irc_validation":
        # Would return your IRC calculator
        pass

    raise ValueError(f"No calculator implemented for Task: '{t_type}' with Software: '{software}'")