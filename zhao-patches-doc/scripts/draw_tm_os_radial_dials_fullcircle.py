#!/usr/bin/env python3
# draw_tm_os_radial_dials_fullcircle.py
#
# Usage:
#   python draw_tm_os_radial_dials_fullcircle.py tm_os_matrix_SUM.csv -o tm_os_dials.svg --png
#
# Optional:
#   python draw_tm_os_radial_dials_fullcircle.py tm_os_matrix_SUM.csv \
#       -o tm_os_dials.svg \
#       --png \
#       --hide-legend \
#       --ignore-os-beyond-valence
#
# Input CSV format:
#   Metal,-3,-2,-1,0,1,2,3,4,5,6,Total
#   Sc,0,0,0,10,4,2,0,0,0,0,16
#   ...
#
# Meaning:
#   - Each transition-metal cell contains a radial oxidation-state dial.
#   - Nonzero OS values are spokes around the circle.
#   - Spoke length = per-metal fraction of that OS.
#   - Spoke color = sign of OS.
#   - Center puck area = fraction of OS 0.
#   - Ring opacity = oxidation-state diversity.
#   - Optional flag can remove OS values with |OS| larger than neutral valence count.

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle


# ============================================================
# Colors
# ============================================================

POS_COLOR = "#D4AF37"   # positive OS, gold
NEG_COLOR = "#23364A"   # negative OS, deep navy
ZERO_COLOR = "#C9D1D9"  # OS 0 / neutral
RING_COLOR = "#B0B7BF"  # reference ring
CELL_EDGE  = "#9AA4AE"
BG_COLOR   = "white"


# ============================================================
# Dial geometry
# ============================================================

# Full-circle layout:
# spokes are evenly spaced around 360 degrees.
START_DEG = 90.0          # first spoke at 12 o'clock, then clockwise
WEDGE_GAP_FRAC = 0.30     # fraction of each angular sector kept empty

# Fraction -> radial length mapping for nonzero OS spokes
INNER_R = 0.18
OUTER_R = 0.46
MIN_FLOOR = 0.08          # minimum visible spoke length for rare states

# Center puck: OS 0 encoded by AREA, so radius ~ sqrt(f0)
PUCK_R_MIN = 0.06
PUCK_R_MAX = 0.16         # keep below INNER_R
PUCK_FILL  = ZERO_COLOR
PUCK_EDGE  = "none"
PUCK_ALPHA = 1.0

PUCK_ABSENT_HOLLOW = True
PUCK_ABSENT_EDGE   = ZERO_COLOR
PUCK_ABSENT_LW     = 0.8

# Optional numeric label for OS 0 fraction
LABEL_ZERO       = False
LABEL_ZERO_FMT   = "{:.0%}"
LABEL_ZERO_DY    = 0.03
LABEL_ZERO_SIZE  = 5
LABEL_ZERO_COLOR = "#444444"

# Entropy ring
ENTROPY_RING = True
ENTROPY_RING_R = 0.50
ENTROPY_ALPHA_MIN = 0.15
ENTROPY_ALPHA_MAX = 0.85


# ============================================================
# Oxidation-state binning
# ============================================================

NEG_MIN = -3       # merge all OS <= -3 into this bin
POS_MAX = +6       # merge all OS >= +6 into this bin

SHOW_EXTREME_LABELS = True

# Label all nonzero OS spokes
LABEL_ALL_SPOKES = True
LABEL_FONT_SIZE = 5
LABEL_FONT_WEIGHT = "bold"
LABEL_OFFSET = 0.10
LABEL_COLOR = "#444444"


# ============================================================
# Grid geometry
# ============================================================

CELL_W = 1.2
CELL_H = 1.32      # mild bump to fit the literature-OS annotation below the dial
PAD_X  = 0.5
PAD_Y  = 0.5
LABEL_MARGIN = 0.07
COMMON_OS_DY = 0.55   # how far below dial center to place the 'lit:' label


# ============================================================
# Neutral valence electron dictionary
# ============================================================

EL_VALENCE = {
    'h':1, 'he':2,
    'li':1, 'be':2,
    'b':3,  'c':4,  'n':5,  'o':6,  'f':7,  'ne':8,
    'na':1, 'mg':2,
    'al':3, 'si':4, 'p':5,  's':6,  'cl':7, 'ar':8,
    'k':1, 'ca':2,
    'sc':3, 'ti':4, 'v':5, 'cr':6, 'mn':7, 'fe':8,
    'co':9, 'ni':10, 'cu':11, 'zn':12,
    'ga':3, 'ge':4, 'as':5, 'se':6, 'br':7, 'kr':8,
    'rb':1, 'sr':2,
    'y':3, 'zr':4, 'nb':5, 'mo':6, 'tc':7, 'ru':8,
    'rh':9, 'pd':10, 'ag':11, 'cd':12,
    'in':3, 'sn':4, 'sb':5, 'te':6, 'i':7, 'xe':8,
    'cs':1, 'ba':2,
    'la':3, 'hf':4, 'ta':5, 'w':6, 're':7, 'os':8,
    'ir':9, 'pt':10, 'au':11, 'hg':12,
    'tl':3, 'pb':4, 'bi':5, 'po':6, 'at':7, 'rn':8
}


# ============================================================
# Transition metal layout
# Groups: G3 G4 G5 G6 G7 G8 G9 G10 G11 G12
# ============================================================

# Most common (textbook) oxidation states per element.
# Source: Cotton/Greenwood inorganic chemistry + LibreTexts; for organometallic-
# heavy contexts we list the 1-2 catalytically dominant values.
COMMON_OS = {
    'Sc': '+3',
    'Ti': '+4', 'V': '+4,+5', 'Cr': '+3,+6', 'Mn': '+2,+7',
    'Fe': '+2,+3', 'Co': '+2,+3', 'Ni': '+2', 'Cu': '+1,+2', 'Zn': '+2',
    'Y':  '+3',
    'Zr': '+4', 'Nb': '+5', 'Mo': '+4,+6', 'Tc': '+4,+7',
    'Ru': '+2,+3', 'Rh': '+1,+3', 'Pd': '0,+2', 'Ag': '+1', 'Cd': '+2',
    'La': '+3',
    'Hf': '+4', 'Ta': '+5', 'W':  '+4,+6', 'Re': '+5,+7',
    'Os': '+4,+8', 'Ir': '+1,+3', 'Pt': '+2,+4', 'Au': '+1,+3', 'Hg': '+2',
}
COMMON_OS_FONT_SIZE = 6
COMMON_OS_COLOR = "#7d4a1f"   # warm brown — visually distinct from spoke gold/navy


ROW_LABELS = ["3d", "4d", "5d"]

ORDER_3D = ['Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
ORDER_4D = ['Y',  'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']

# Important:
# 5d row has a blank Group-3 placeholder so Hf aligns under Ti,
# and Au aligns with Cu/Ag.
ORDER_5D = [None, 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']

TM_ROWS = [ORDER_3D, ORDER_4D, ORDER_5D]


# ============================================================
# Helper functions
# ============================================================

def is_int_str(s: str) -> bool:
    try:
        int(s)
        return True
    except Exception:
        return False


def clamp_os_bin(k: int) -> int:
    """Merge oxidation states into terminal bins."""
    if k <= NEG_MIN:
        return NEG_MIN
    if k >= POS_MAX:
        return POS_MAX
    return k


def entropy_from_counts(counts: dict) -> float:
    """Shannon entropy over nonzero oxidation-state counts."""
    total = sum(counts.values())
    if total <= 0:
        return 0.0

    p = np.array([c / total for c in counts.values() if c > 0], dtype=float)
    return float(-np.sum(p * np.log2(p)))


def fractions(counts: dict) -> dict:
    """Convert count dictionary to per-metal fractions."""
    total = sum(counts.values())
    if total <= 0:
        return {}

    return {k: v / total for k, v in counts.items() if v > 0}


def spoke_color(k: int) -> str:
    if k > 0:
        return POS_COLOR
    if k < 0:
        return NEG_COLOR
    return ZERO_COLOR


def order_bins_fullcircle(merged_counts: dict):
    """
    Return ordered nonzero OS bins for full-circle placement.

    The order interleaves signs:
        +1, -1, +2, -2, +3, -3, ...

    Only bins with nonzero counts are included.
    """
    present = [
        k for k, v in merged_counts.items()
        if v > 0 and k != 0
    ]

    pos = sorted([k for k in present if k > 0], key=lambda k: abs(k))
    neg = sorted([k for k in present if k < 0], key=lambda k: abs(k))

    inter = []
    i = 0
    j = 0

    while i < len(pos) or j < len(neg):
        if i < len(pos):
            inter.append(pos[i])
            i += 1
        if j < len(neg):
            inter.append(neg[j])
            j += 1

    return inter


# ============================================================
# Drawing
# ============================================================

def draw_dial(ax, cx, cy, metal_counts: dict, metal_label: str):
    """
    Draw one oxidation-state dial at cell center (cx, cy).
    """

    # Merge extreme oxidation states
    merged = {}
    for k, v in metal_counts.items():
        kk = clamp_os_bin(k)
        merged[kk] = merged.get(kk, 0) + v

    fr = fractions(merged)

    # Reference ring
    ax.add_patch(
        Circle(
            (cx, cy),
            ENTROPY_RING_R,
            fill=False,
            ec=RING_COLOR,
            lw=0.6,
            zorder=1
        )
    )

    # Entropy ring
    if ENTROPY_RING:
        observed_bins = [k for k, v in merged.items() if v > 0]
        n_bins = max(1, len(observed_bins))

        e = entropy_from_counts(merged)
        e_max = math.log2(n_bins) if n_bins > 1 else 0.0

        if e_max <= 1e-12:
            alpha = ENTROPY_ALPHA_MIN
        else:
            alpha = ENTROPY_ALPHA_MIN + (
                ENTROPY_ALPHA_MAX - ENTROPY_ALPHA_MIN
            ) * (e / e_max)

        ax.add_patch(
            Circle(
                (cx, cy),
                ENTROPY_RING_R,
                fill=False,
                ec=NEG_COLOR,
                lw=1.2,
                alpha=alpha,
                zorder=2
            )
        )

    # Center puck for OS 0: area encodes f0
    f0 = fr.get(0, 0.0)

    if f0 > 0.0:
        r0 = PUCK_R_MIN + (f0 ** 0.5) * (PUCK_R_MAX - PUCK_R_MIN)

        ax.add_patch(
            Circle(
                (cx, cy),
                r0,
                fc=PUCK_FILL,
                ec=PUCK_EDGE,
                alpha=PUCK_ALPHA,
                zorder=3
            )
        )

        if LABEL_ZERO:
            ax.text(
                cx,
                cy - (r0 + LABEL_ZERO_DY),
                LABEL_ZERO_FMT.format(f0),
                fontsize=LABEL_ZERO_SIZE,
                ha="center",
                va="top",
                color=LABEL_ZERO_COLOR
            )
    else:
        if PUCK_ABSENT_HOLLOW:
            ax.add_patch(
                Circle(
                    (cx, cy),
                    PUCK_R_MIN,
                    fill=False,
                    ec=PUCK_ABSENT_EDGE,
                    lw=PUCK_ABSENT_LW,
                    zorder=3
                )
            )

    # Nonzero spokes
    bins_order = order_bins_fullcircle(merged)
    n = max(1, len(bins_order))

    sector = 360.0 / n
    wedge_width = sector * (1.0 - WEDGE_GAP_FRAC)

    for idx, k in enumerate(bins_order):
        f = fr.get(k, 0.0)
        if f <= 0.0:
            continue

        f_eff = max(f, MIN_FLOOR)

        r_inner = INNER_R
        r_outer = INNER_R + f_eff * (OUTER_R - INNER_R)

        # Clockwise angular placement
        angle = START_DEG - idx * sector

        theta1 = angle - wedge_width / 2.0
        theta2 = angle + wedge_width / 2.0

        color = spoke_color(k)

        wedge = Wedge(
            center=(cx, cy),
            r=r_outer,
            theta1=theta1,
            theta2=theta2,
            width=(r_outer - r_inner),
            facecolor=color,
            edgecolor="none",
            zorder=4
        )

        ax.add_patch(wedge)

        # Labels beside every spoke tip
        if LABEL_ALL_SPOKES:
            if k == NEG_MIN:
                label = f"≤{NEG_MIN}"
            elif k == POS_MAX:
                label = f"≥{POS_MAX}"
            else:
                label = str(k)

            rad = math.radians(angle)

            lx = cx + (r_outer + LABEL_OFFSET) * math.cos(rad)
            ly = cy + (r_outer + LABEL_OFFSET) * math.sin(rad)

            ax.text(
                lx,
                ly,
                label,
                fontsize=LABEL_FONT_SIZE,
                ha="center",
                va="center",
                color=LABEL_COLOR,
                fontweight=LABEL_FONT_WEIGHT
            )

        elif SHOW_EXTREME_LABELS and k in (NEG_MIN, POS_MAX):
            label = f"≤{NEG_MIN}" if k == NEG_MIN else f"≥{POS_MAX}"

            rad = math.radians(angle)

            lx = cx + (r_outer + LABEL_OFFSET) * math.cos(rad)
            ly = cy + (r_outer + LABEL_OFFSET) * math.sin(rad)

            ax.text(
                lx,
                ly,
                label,
                fontsize=LABEL_FONT_SIZE,
                ha="center",
                va="center",
                color=LABEL_COLOR,
                fontweight=LABEL_FONT_WEIGHT
            )

    # Element label at top-left
    tlx = cx - CELL_W / 2.0 + LABEL_MARGIN
    tly = cy + CELL_H / 2.0 - LABEL_MARGIN

    ax.text(
        tlx,
        tly,
        metal_label,
        fontsize=8,
        ha="left",
        va="top",
        color="#111111"
    )

    # Common-OS annotation below the dial (literature reference values).
    # Placed at fixed offset COMMON_OS_DY below the dial center so it sits in
    # the extended bottom strip of the cell and never overlaps spoke labels.
    common = COMMON_OS.get(metal_label)
    if common:
        ax.text(
            cx,
            cy - COMMON_OS_DY,
            f"lit: {common}",
            fontsize=COMMON_OS_FONT_SIZE,
            ha="center",
            va="top",
            color=COMMON_OS_COLOR,
            style="italic",
        )


# ============================================================
# Data loading
# ============================================================

def load_matrix(csv_path: Path, ignore_beyond_valence: bool = False):
    """
    Load Metal x OS matrix.

    Returns:
        metals[Metal] = {OS: count}

    If ignore_beyond_valence=True:
        remove any OS k where |k| > neutral valence electrons for that element.
    """
    with csv_path.open("r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr)

        idx = {h: i for i, h in enumerate(header)}

        if "Metal" not in idx:
            raise ValueError("CSV must have a 'Metal' column.")

        metals = {}

        for row in rdr:
            if not row:
                continue

            m = row[idx["Metal"]].strip()

            if not m or m == "AllMetals":
                continue

            counts = {}

            for h, i in idx.items():
                if is_int_str(h):
                    try:
                        raw = row[i].strip()
                        v = int(raw) if raw else 0
                    except Exception:
                        v = 0

                    if v != 0:
                        counts[int(h)] = v

            if not counts:
                continue

            if ignore_beyond_valence:
                val = EL_VALENCE.get(m.lower())

                if val is not None:
                    counts = {
                        k: v for k, v in counts.items()
                        if abs(k) <= val
                    }

            if counts:
                metals[m] = counts

    return metals


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Draw full-circle radial oxidation-state dials "
            "for transition metals."
        )
    )

    ap.add_argument(
        "csv",
        type=Path,
        help="Summed Metal x OS CSV."
    )

    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("tm_os_dials.svg"),
        help="Output SVG path."
    )

    ap.add_argument(
        "--png",
        action="store_true",
        help="Also write a PNG alongside the SVG."
    )

    ap.add_argument(
        "--ignore-os-beyond-valence",
        action="store_true",
        help=(
            "Drop any OS k where |k| is greater than the element's "
            "neutral valence-electron count."
        )
    )

    ap.add_argument(
        "--hide-legend",
        action="store_true",
        help="Do not render the legend block."
    )

    args = ap.parse_args()

    metals = load_matrix(
        args.csv,
        ignore_beyond_valence=args.ignore_os_beyond_valence
    )

    n_rows = 3
    n_cols = 10

    # Add a little extra width only when legend is shown
    legend_extra = 2.5 if not args.hide_legend else 0.0

    fig_w = PAD_X * 2 + n_cols * CELL_W + legend_extra
    fig_h = PAD_Y * 2 + n_rows * CELL_H

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    # Draw cells and dials
    for r, row in enumerate(TM_ROWS):
        y = PAD_Y + (n_rows - 1 - r) * CELL_H + CELL_H / 2.0

        for c, metal in enumerate(row):
            x = PAD_X + c * CELL_W + CELL_W / 2.0

            # Cell border
            ax.add_patch(
                Rectangle(
                    (x - CELL_W / 2.0, y - CELL_H / 2.0),
                    CELL_W,
                    CELL_H,
                    fill=False,
                    ec=CELL_EDGE,
                    lw=0.7,
                    zorder=0
                )
            )

            # Placeholder for 5d group-3 alignment
            if metal is None:
                continue

            counts = metals.get(metal)

            if counts:
                draw_dial(ax, x, y, counts, metal)
            else:
                # Empty-data hint
                tlx = x - CELL_W / 2.0 + LABEL_MARGIN
                tly = y + CELL_H / 2.0 - LABEL_MARGIN

                ax.text(
                    tlx,
                    tly,
                    metal,
                    fontsize=8,
                    ha="left",
                    va="top",
                    color="#888888"
                )

    # Row labels
    for r, label in enumerate(ROW_LABELS):
        ylab = PAD_Y + (n_rows - 1 - r) * CELL_H + CELL_H / 2.0

        ax.text(
            PAD_X - 0.28,
            ylab,
            label,
            fontsize=8,
            ha="right",
            va="center",
            color="#444444"
        )

    # Optional legend
    if not args.hide_legend:
        legend_x = PAD_X + n_cols * CELL_W + 0.25
        legend_y = PAD_Y + n_rows * CELL_H - 0.2

        ax.text(
            legend_x,
            legend_y,
            "Full-circle spokes\n"
            "Color = sign: − navy, + gold\n"
            "Length = per-metal fraction\n"
            "Center disk area = fraction at OS 0\n"
            "Labels = oxidation state\n"
            "≤ / ≥ = merged extreme bins\n"
            "Ring opacity = OS diversity",
            fontsize=7,
            ha="left",
            va="top",
            color="#222222"
        )

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        args.out,
        bbox_inches="tight",
        pad_inches=0.05
    )

    if args.png:
        png_path = args.out.with_suffix(".png")
        fig.savefig(
            png_path,
            bbox_inches="tight",
            pad_inches=0.05
        )

    plt.close(fig)

    print(f"✓ Wrote {args.out.resolve()}")

    if args.png:
        print(f"✓ Wrote {png_path.resolve()}")


if __name__ == "__main__":
    main()
