"""
Compatibility wrapper for GSM conformer pair selection.
"""


def select_gsm_pairs(rxn, config, scratch_dir=None):
    """
    Select reactant/product conformer pairs for GSM.

    Host-side ML pair selection has been removed. The Pysisyphus TS guess
    workflow now runs conformer selection, joint optimization, ML scoring, and
    GSM inside the jo_opt container.
    """
    raise RuntimeError(
        "Host-side GSM conformer pair selection has been removed. "
        "Use the Pysisyphus TS guess workflow, which runs pre-GSM selection "
        "inside the jo_opt container."
    )
