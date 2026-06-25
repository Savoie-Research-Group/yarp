import pytest

pytest.importorskip("openbabel")

from yarp.util.config import TSGuessConfig


def test_ts_guess_config_defaults_to_openbabel_joint_opt():
    cfg = TSGuessConfig(
        software="pysisyphus",
        gsm_lot="xtb",
        charge=0,
        multiplicity=1,
    )

    assert cfg.joint_opt_engine == "ob"
    assert cfg.xtb_joint_lot == "gfn2"
    assert cfg.containerize_pre_gsm is False


def test_ts_guess_config_accepts_xtb_joint_opt_options():
    cfg = TSGuessConfig(
        software="pysisyphus",
        gsm_lot="xtb",
        charge=0,
        multiplicity=1,
        joint_opt_engine="xtb",
        xtb_joint_force_constant=1,
        containerize_pre_gsm=True,
    )

    assert cfg.joint_opt_engine == "xtb"
    assert cfg.xtb_joint_force_constant == 1.0
    assert cfg.containerize_pre_gsm is True


def test_ts_guess_config_rejects_unknown_joint_opt_engine():
    with pytest.raises(ValueError, match="joint_opt_engine"):
        TSGuessConfig(
            software="pysisyphus",
            gsm_lot="xtb",
            charge=0,
            multiplicity=1,
            joint_opt_engine="bad",
        )
