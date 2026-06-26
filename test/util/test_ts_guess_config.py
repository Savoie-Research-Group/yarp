import pytest

pytest.importorskip("openbabel")

from yarp.util.config import TSGuessConfig
from yarp.reaction.external.conformer_select import ConformerPairSelector


def test_ts_guess_config_defaults_to_openbabel_joint_opt():
    cfg = TSGuessConfig(
        software="pysisyphus",
        gsm_lot="xtb",
        charge=0,
        multiplicity=1,
    )

    assert cfg.joint_opt_engine == "ob"
    assert cfg.xtb_joint_lot == "gfn2"


def test_ts_guess_config_accepts_xtb_joint_opt_options():
    cfg = TSGuessConfig(
        software="pysisyphus",
        gsm_lot="xtb",
        charge=0,
        multiplicity=1,
        joint_opt_engine="xtb",
        xtb_joint_force_constant=1,
    )

    assert cfg.joint_opt_engine == "xtb"
    assert cfg.xtb_joint_force_constant == 1.0


def test_ts_guess_config_rejects_unknown_joint_opt_engine():
    with pytest.raises(ValueError, match="joint_opt_engine"):
        TSGuessConfig(
            software="pysisyphus",
            gsm_lot="xtb",
            charge=0,
            multiplicity=1,
            joint_opt_engine="bad",
        )


def test_conformer_pair_selection_requires_container(monkeypatch):
    monkeypatch.delenv("YARP_PREGSM_CONTAINER", raising=False)

    with pytest.raises(RuntimeError, match="jo_opt pre-GSM container"):
        ConformerPairSelector.__new__(ConformerPairSelector).select()
