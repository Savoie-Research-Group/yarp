"""
Pysisyphus-specific helpers shared by the Pysisyphus-backed tasks
(ts_guess, min_opt, ts_opt, irc_val): which container image to run for a
given level of theory, and the matching `calc:` block for the input YAML.
"""


# Pysisyphus container per level of theory. `xtb` (GFN2) keeps the stock image:
# upstream pysisyphus + conda xtb. `gxtb` uses the g-xTB image built from
# containers/yarp_gxtb (patched pysisyphus + the g-xTB build of xtb).
# Bump the gxtb tag whenever that image is rebuilt: YARP only pulls an image
# (or an Apptainer .sif) that is missing locally, so re-pushing an existing tag
# never reaches users who already have it.
PYSIS_XTB_IMAGE = "erm42/yarp:pysis_xtb"
PYSIS_GXTB_IMAGE = "erm42/yarp_gxtb:v1"
PYSIS_IMAGES = {"xtb": PYSIS_XTB_IMAGE, "gxtb": PYSIS_GXTB_IMAGE}


def pysis_image(lot):
    """Container image for a Pysisyphus task run at level of theory `lot`."""
    key = lot.lower()
    if key not in PYSIS_IMAGES:
        raise ValueError(f"Unsupported Pysisyphus level of theory: {lot}")
    return PYSIS_IMAGES[key]


def pysis_xtb_calc_lines(lot, config):
    """
    Return the Pysisyphus `calc:` block for xTB (GFN2) or g-xTB.

    Both use pysisyphus' XTB calculator. The `gxtb: True` flag (patched
    pysisyphus in the yarp_gxtb image) selects --gxtb instead of --gfn 2. It is
    written only for g-xTB, so `lot: xtb` input stays valid for stock pysisyphus,
    which does not know the keyword.
    """
    key = lot.lower()
    if key not in PYSIS_IMAGES:
        raise ValueError(f"Unsupported Pysisyphus level of theory: {lot}")

    lines = [
        "calc:\n",
        " type: xtb\n",
        f" pal: {config.n_cpus}\n",
        f" mem: {config.mem_per_cpu}\n",
        f" charge: {config.charge}\n",
        f" mult: {config.multiplicity}\n",
    ]
    if key == "gxtb":
        lines.append(" gxtb: True\n")
    return lines
