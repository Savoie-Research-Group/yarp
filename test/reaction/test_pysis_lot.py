"""Level-of-theory routing for Pysisyphus tasks: container image and calc block."""
from types import SimpleNamespace

import pytest

from yarp.reaction.external.calc_base import (
    PYSIS_GXTB_IMAGE,
    PYSIS_XTB_IMAGE,
    pysis_image,
    pysis_xtb_calc_lines,
)

CONFIG = SimpleNamespace(n_cpus=2, mem_per_cpu=4000, charge=0, multiplicity=1)


def test_xtb_uses_stock_image():
    assert pysis_image("xtb") == PYSIS_XTB_IMAGE == "erm42/yarp:pysis_xtb"


def test_gxtb_uses_gxtb_image():
    assert pysis_image("gxtb") == PYSIS_GXTB_IMAGE
    assert PYSIS_GXTB_IMAGE != PYSIS_XTB_IMAGE


def test_image_lookup_is_case_insensitive():
    assert pysis_image("GXTB") == PYSIS_GXTB_IMAGE


def test_xtb_calc_block_has_no_gxtb_keyword():
    # Stock pysisyphus (erm42/yarp:pysis_xtb) does not accept a `gxtb` argument.
    block = "".join(pysis_xtb_calc_lines("xtb", CONFIG))
    assert "type: xtb" in block
    assert "gxtb" not in block
    assert "pal: 2" in block and "mem: 4000" in block


def test_gxtb_calc_block_requests_gxtb():
    block = "".join(pysis_xtb_calc_lines("gxtb", CONFIG))
    assert "type: xtb" in block
    assert "gxtb: True" in block
    assert "type: gxtb" not in block


@pytest.mark.parametrize("func", [pysis_image, lambda lot: pysis_xtb_calc_lines(lot, CONFIG)])
def test_unknown_lot_is_rejected(func):
    with pytest.raises(ValueError, match="Unsupported Pysisyphus level of theory"):
        func("orca")
