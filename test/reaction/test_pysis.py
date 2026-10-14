"""Level-of-theory routing for Pysisyphus tasks: container image and calc block."""
from types import SimpleNamespace

import pytest

from yarp.reaction.external.pysis import (
    PYSIS_GXTB_IMAGE,
    PYSIS_XTB_IMAGE,
    pysis_image,
    pysis_xtb_calc_lines,
)


class TestPysisLevelOfTheory:
    """
    `lot: xtb` must keep the stock image and stock Pysisyphus input; `lot: gxtb`
    must get the g-xTB image and the `gxtb: True` flag that only its patched
    Pysisyphus understands.
    """

    CONFIG = SimpleNamespace(n_cpus=2, mem_per_cpu=4000, charge=0, multiplicity=1)

    def test_xtb_uses_stock_image(self):
        assert pysis_image("xtb") == PYSIS_XTB_IMAGE == "erm42/yarp:pysis_xtb"

    def test_gxtb_uses_gxtb_image(self):
        assert pysis_image("gxtb") == PYSIS_GXTB_IMAGE
        assert PYSIS_GXTB_IMAGE != PYSIS_XTB_IMAGE

    def test_image_lookup_is_case_insensitive(self):
        assert pysis_image("GXTB") == PYSIS_GXTB_IMAGE

    def test_xtb_calc_block_has_no_gxtb_keyword(self):
        # Stock pysisyphus (erm42/yarp:pysis_xtb) does not accept a `gxtb` argument.
        block = "".join(pysis_xtb_calc_lines("xtb", self.CONFIG))
        assert "type: xtb" in block
        assert "gxtb" not in block

    def test_gxtb_calc_block_requests_gxtb(self):
        block = "".join(pysis_xtb_calc_lines("gxtb", self.CONFIG))
        assert "type: xtb" in block
        assert "gxtb: True" in block
        assert "type: gxtb" not in block

    def test_unknown_lot_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported Pysisyphus level of theory"):
            pysis_image("orca")
        with pytest.raises(ValueError, match="Unsupported Pysisyphus level of theory"):
            pysis_xtb_calc_lines("orca", self.CONFIG)
