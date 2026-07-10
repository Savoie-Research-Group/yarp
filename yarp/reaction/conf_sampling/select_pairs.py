"""
Compatibility wrapper for GSM conformer pair selection.
"""


def select_gsm_pairs(rxn, config, scratch_dir=None):
    """
    Select reactant/product conformer pairs for GSM.

    The old direct host helper has been removed from this module. The
    Pysisyphus TS guess workflow owns conformer selection and joint
    optimization; only sklearn model scoring is delegated to the model_scorer
    container.
    """
    raise RuntimeError(
        "Direct GSM conformer pair selection from yarp.reaction.conf_sampling "
        "has been removed. Use the Pysisyphus TS guess workflow; it delegates "
        "only sklearn model scoring to the model_scorer container."
    )
