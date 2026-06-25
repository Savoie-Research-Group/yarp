"""
Compatibility wrapper for GSM conformer pair selection.
"""

from yarp.reaction.external.conformer_select import ConformerPairSelector


def select_gsm_pairs(rxn, config, scratch_dir=None):
    """
    Select reactant/product conformer pairs for GSM.
    """
    return ConformerPairSelector(rxn, config, scratch_dir=scratch_dir).select()
