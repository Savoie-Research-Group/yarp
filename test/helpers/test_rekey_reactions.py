"""Tests for the optional, offline reaction-pickle rekey utility."""

from rekey_reactions import rekey_reactions

from yarp.reaction.reaction import reaction
from yarp.yarpecule.yarpecule import yarpecule


def test_rekey_retains_first_reaction_and_reports_deduplication(capsys):
    reactant = yarpecule(
        "[C:0]([C:1](=[O:2])[H:3])([H:4])([H:5])[H:6]",
        canon=False,
    )
    product = yarpecule(
        "[C:0](=[C:1]([O:2][H:4])[H:3])([H:5])[H:6]",
        canon=False,
    )
    first = reaction(reactant, product)
    second = reaction(product, reactant)
    first.network_meta["retained"] = True

    result = rekey_reactions({"first": first, "second": second})

    assert len(result) == 1
    assert next(iter(result.values())) is first
    assert next(iter(result.values())).network_meta["retained"] is True
    assert "Deduplicated reaction" in capsys.readouterr().out
