"""progress_yarp fast-forward: stored data keys must match the level of theory exactly."""
import pytest

from yarp.progress_yarp import _matches_lot


@pytest.mark.parametrize("key, desired, expected", [
    # "xtb_pysisyphus" is a substring of every g-xTB key; it must not match them
    ("preopt_gxtb_pysisyphus", "xtb_pysisyphus", False),
    ("1_tsopt_gxtb_pysisyphus", "xtb_pysisyphus", False),
    ("rpopt_gxtb_pysisyphus", "xtb_pysisyphus", False),
    ("ts_guess_2_gxtb_pysisyphus", "xtb_pysisyphus", False),
    ("gxtb_pysisyphus", "xtb_pysisyphus", False),          # barrier dict key
    # genuine matches
    ("preopt_xtb_pysisyphus", "xtb_pysisyphus", True),
    ("1_tsopt_gxtb_pysisyphus", "gxtb_pysisyphus", True),
    ("conf_gen_rank3_gfn2_crest", "gfn2_crest", True),
    ("xtb_pysisyphus", "xtb_pysisyphus", True),            # barrier dict key
])
def test_matches_lot(key, desired, expected):
    assert _matches_lot(key, desired) is expected
