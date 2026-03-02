#!/usr/bin/env python3
"""Generate bulk per-product plots and summary CSV for one network output directory."""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "mpl_cache_subnetwork_kinetics"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)
if "XDG_CACHE_HOME" not in os.environ:
    xdg_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "xdg_cache_subnetwork_kinetics"
    xdg_cache.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache)

import matplotlib.pyplot as plt


def default_config_path():
    cwd_cfg = Path("pipeline/configs/pipeline_config.yaml")
    if cwd_cfg.exists():
        return cwd_cfg
    return SCRIPT_DIR / "configs" / "pipeline_config.yaml"


def resolve_path(path_text, config_dir):
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (config_dir / path).resolve()


def load_config(config_path):
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be a mapping.")
    return cfg, config_path.parent.resolve()


def read_parquet_with_arrow_retry(path):
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        if "pandas.period already defined" not in str(exc):
            raise
        import pyarrow as pa

        for ext_name in ("pandas.period", "pandas.interval"):
            try:
                pa.unregister_extension_type(ext_name)
            except Exception:
                pass
        sys.modules.pop("pandas.core.arrays.arrow.extension_types", None)
        return pd.read_parquet(path)


def load_table(base_path):
    base_path = Path(base_path)
    if base_path.exists():
        if base_path.suffix == ".parquet":
            return read_parquet_with_arrow_retry(base_path), base_path
        return pd.read_pickle(base_path), base_path
    pkl = base_path.with_suffix(".pkl")
    if pkl.exists():
        return pd.read_pickle(pkl), pkl
    return pd.DataFrame(), None


def load_personal_colors(csv_path):
    pal_df = pd.read_csv(csv_path)
    col_map = {str(c).strip().lower(): c for c in pal_df.columns}
    name_col = col_map.get("color name")
    hex_col = col_map.get("hex code")
    if not name_col or not hex_col:
        raise RuntimeError(f"Palette CSV must include columns: Color Name, Hex Code. Found: {list(pal_df.columns)}")
    out = {}
    for _, row in pal_df.iterrows():
        name = str(row[name_col]).strip()
        hex_code = str(row[hex_col]).strip()
        if not name or not hex_code or hex_code.lower() == "nan":
            continue
        if not hex_code.startswith("#"):
            hex_code = f"#{hex_code.lstrip('#')}"
        out[name] = hex_code.upper()
    return out


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))[:120]


def parse_terminal_token_counts(terminal_states_text, product_smiles):
    states = [s.strip() for s in str(terminal_states_text or "").split(";") if s.strip()]
    if not states:
        return Counter([product_smiles]) if product_smiles else Counter()
    chosen = None
    for st in states:
        tokens = [t.strip() for t in st.split(".") if t.strip()]
        if product_smiles in tokens:
            chosen = tokens
            break
    if chosen is None:
        chosen = [t.strip() for t in states[0].split(".") if t.strip()]
    return Counter(chosen)


def role_for_species(smi, role_sets, product_smiles, reagent_smiles, coproduct_set, role_priority, role_colors):
    if smi == product_smiles:
        return "product"
    if smi == reagent_smiles:
        return "reagent"
    if smi in coproduct_set:
        return "co_product"
    roles = role_sets.get(smi, set())
    if not roles:
        return "unknown"
    best = sorted(roles, key=lambda r: role_priority.get(r, 99))[0]
    return best if best in role_colors else "unknown"


def add_initial_concentration_anchor(conc_df, all_species, reagent_smiles):
    if conc_df.empty:
        base = pd.DataFrame(columns=["time_s", "species_smiles", "concentration_x"])
    else:
        base = conc_df[conc_df["time_s"] > 0.0].copy()
    species = [str(s) for s in all_species if str(s)]
    if reagent_smiles and reagent_smiles not in species:
        species.append(reagent_smiles)
    anchor = pd.DataFrame(
        {
            "time_s": 0.0,
            "species_smiles": species,
            "concentration_x": [1.0 if s == reagent_smiles else 0.0 for s in species],
        }
    )
    out = pd.concat([base, anchor], ignore_index=True)
    return out.groupby(["time_s", "species_smiles"], as_index=False)["concentration_x"].mean()


def add_initial_flux_anchor(flux_df, all_species):
    if flux_df.empty:
        base = pd.DataFrame(columns=["time_s", "species_smiles", "cumulative_in_flux", "cumulative_out_flux"])
    else:
        base = flux_df[flux_df["time_s"] > 0.0].copy()
    species = [str(s) for s in all_species if str(s)]
    anchor = pd.DataFrame(
        {
            "time_s": 0.0,
            "species_smiles": species,
            "cumulative_in_flux": 0.0,
            "cumulative_out_flux": 0.0,
        }
    )
    out = pd.concat([base, anchor], ignore_index=True)
    return out.groupby(["time_s", "species_smiles"], as_index=False)[["cumulative_in_flux", "cumulative_out_flux"]].mean()


def split_state_parts(state_text):
    return {p.strip() for p in str(state_text or "").split(".") if p and str(p).lower() != "nan"}


def classify_reaction_to_product(row, reagent_smiles, product_smiles):
    from_parts = split_state_parts(row.get("from_smiles", ""))
    to_parts = split_state_parts(row.get("to_smiles", ""))
    source_type = str(row.get("source_type", "") or "").strip()
    if not product_smiles or product_smiles not in to_parts:
        return ""
    if (reagent_smiles and reagent_smiles in from_parts) or source_type == "R":
        return "R->P"
    if source_type == "I" or (reagent_smiles and reagent_smiles not in from_parts):
        return "I->P"
    return ""


def decorate_reaction_label(label, class_map):
    tag = class_map.get(str(label), "")
    return f"{label} [{tag}]" if tag else str(label)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate per-product profile plots for a network output directory.")
    parser.add_argument("--config", default=None, help="Path to pipeline config YAML.")
    parser.add_argument("--network-out-dir", required=True, help="Path to one networks__* output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_path = Path(args.config) if args.config else default_config_path()
    cfg, cfg_dir = load_config(cfg_path)
    plot_cfg = cfg.get("plot_export", {}) or {}
    if not bool(plot_cfg.get("enabled", False)):
        print("plot_export.enabled=false; skipping plot generation.")
        return

    network_out = Path(args.network_out_dir).expanduser().resolve()
    if not network_out.exists():
        raise FileNotFoundError(f"Network output directory not found: {network_out}")

    profile_pattern = str(plot_cfg.get("profile_pattern", "random_flux_timeseries*.parquet"))
    product_pattern = str(plot_cfg.get("product_pattern", "product_*.parquet"))
    output_subdir = str(plot_cfg.get("output_subdir", "bulk_profile_plots"))
    summary_filename = str(plot_cfg.get("summary_csv_name", "bulk_profile_summary.csv"))
    max_conc_species = int(plot_cfg.get("max_conc_species", 12))
    max_flux_species = int(plot_cfg.get("max_flux_species", 10))
    max_reaction_lines = int(plot_cfg.get("max_reaction_lines", 12))
    fig_dpi = int(plot_cfg.get("fig_dpi", 170))
    palette_csv = resolve_path(str(plot_cfg.get("palette_csv", "../../color_palettes/colors_personal.csv")), cfg_dir)

    out_root = network_out / output_subdir
    summary_csv = out_root / summary_filename
    out_root.mkdir(parents=True, exist_ok=True)

    personal_colors = load_personal_colors(palette_csv)

    def cp(name, fallback):
        return personal_colors.get(name, fallback)

    bulk_color_cycle = [
        cp("Midnight Gridiron", "#11214F"),
        cp("Ion Blue", "#1EA7FF"),
        cp("Electric Emerald", "#00C853"),
        cp("Gridiron Violet", "#6A37C8"),
        cp("Edge Pink", "#FF5FA2"),
        cp("Slate Line", "#3A4450"),
        cp("Steel Grey", "#5A606B"),
        cp("Flat Black", "#0D0D0D"),
        cp("Iron Grey", "#2B2F33"),
        cp("Brass", "#D4B56E"),
    ]
    role_colors = {
        "reagent": cp("Ion Blue", "#1EA7FF"),
        "intermediate": cp("Electric Emerald", "#00C853"),
        "co_product": cp("Gridiron Violet", "#6A37C8"),
        "product": cp("Edge Pink", "#FF5FA2"),
        "unknown": cp("Slate Line", "#3A4450"),
    }
    role_priority = {"product": 0, "reagent": 1, "co_product": 2, "intermediate": 3, "unknown": 4}

    profile_map = {}
    for path in sorted(network_out.glob(profile_pattern)):
        df, loaded = load_table(path)
        if loaded is None or df.empty or "product_id" not in df.columns:
            continue
        ids = df["product_id"].dropna().astype(str)
        if not ids.empty:
            profile_map[ids.iloc[0]] = loaded

    product_tables = sorted(network_out.glob(product_pattern))
    if not product_tables:
        product_tables = sorted(network_out.glob(product_pattern.replace(".parquet", ".pkl")))

    summary_rows = []
    for product_path in product_tables:
        tdf, tdf_path = load_table(product_path)
        if tdf_path is None or tdf.empty or "row_role" not in tdf.columns:
            continue
        pid = product_path.stem.replace("product_", "")
        product_row = tdf[tdf["row_role"] == "product"].head(1)
        reagent_row = tdf[tdf["row_role"] == "reagent"].head(1)
        if product_row.empty:
            continue

        product_smiles = str(product_row["species_smiles"].iloc[0])
        reagent_smiles = str(reagent_row["species_smiles"].iloc[0]) if not reagent_row.empty else ""
        product_x_final = float(product_row["final_concentration"].iloc[0]) if "final_concentration" in product_row.columns else np.nan
        terminal_states = (
            str(tdf["terminal_product_states"].dropna().iloc[0])
            if "terminal_product_states" in tdf.columns and not tdf["terminal_product_states"].dropna().empty
            else product_smiles
        )
        completion_terminal_conc = (
            float(tdf["completion_terminal_concentration"].dropna().iloc[0])
            if "completion_terminal_concentration" in tdf.columns and not tdf["completion_terminal_concentration"].dropna().empty
            else np.nan
        )
        token_counts = parse_terminal_token_counts(terminal_states, product_smiles)
        expected_scale = 0.0
        if token_counts:
            total_tokens = sum(token_counts.values())
            expected_scale = token_counts.get(product_smiles, 0) / total_tokens if total_tokens > 0 else 0.0

        nonprod = tdf[tdf["row_role"] != "product"][["species_smiles", "final_concentration"]].copy() if "final_concentration" in tdf.columns else pd.DataFrame()
        nonprod = nonprod.sort_values("final_concentration", ascending=False) if not nonprod.empty else nonprod
        top_nonproduct_smiles = str(nonprod.iloc[0]["species_smiles"]) if len(nonprod) else ""
        top_nonproduct_x = float(nonprod.iloc[0]["final_concentration"]) if len(nonprod) else np.nan

        profile_path = profile_map.get(pid)
        plot_dir = out_root / f"product_{pid}__{safe_name(product_smiles)}"
        plot_dir.mkdir(parents=True, exist_ok=True)

        conc_plot = plot_dir / "concentration_species_vs_time.png"
        p_ps_plot = plot_dir / "concentration_P_and_PS_vs_time.png"
        flux_plot = plot_dir / "flux_in_out_vs_time.png"
        rxn_flux_plot = plot_dir / "reaction_top_cumulative_flux_vs_time.png"
        rxn_current_plot = plot_dir / "reaction_top_current_flux_vs_time.png"

        if profile_path is None:
            summary_rows.append(
                {
                    "product_id": pid,
                    "product_smiles": product_smiles,
                    "profile_found": False,
                    "profile_path": "",
                    "profile_size_mb": np.nan,
                    "profile_rows": np.nan,
                    "timepoints": np.nan,
                    "terminal_product_states": terminal_states,
                    "expected_product_scale": expected_scale,
                    "product_x_final": product_x_final,
                    "completion_terminal_concentration": completion_terminal_conc,
                    "top_nonproduct_smiles": top_nonproduct_smiles,
                    "top_nonproduct_x": top_nonproduct_x,
                    "plot_species_concentration": "",
                    "plot_p_and_ps": "",
                    "plot_flux_in_out": "",
                    "plot_reaction_cumulative_flux": "",
                    "plot_reaction_current_flux": "",
                    "n_reaction_labels_r_to_p": 0,
                    "n_reaction_labels_i_to_p": 0,
                }
            )
            continue

        rdf, _ = load_table(profile_path)
        if rdf.empty:
            continue

        conc = rdf[rdf.get("row_kind", "") == "species_concentration"][["time_s", "species_smiles", "concentration_x"]].copy() if "row_kind" in rdf.columns else pd.DataFrame()
        if not conc.empty:
            conc["time_s"] = pd.to_numeric(conc["time_s"], errors="coerce")
            conc = conc.dropna(subset=["time_s", "species_smiles", "concentration_x"])
            conc = conc.groupby(["time_s", "species_smiles"], as_index=False)["concentration_x"].mean()

        flux = rdf[rdf.get("row_kind", "") == "species_flux"][["time_s", "species_smiles", "cumulative_in_flux", "cumulative_out_flux"]].copy() if "row_kind" in rdf.columns else pd.DataFrame()
        if not flux.empty:
            flux["time_s"] = pd.to_numeric(flux["time_s"], errors="coerce")
            flux = flux.dropna(subset=["time_s", "species_smiles"])
            flux = flux.groupby(["time_s", "species_smiles"], as_index=False)[["cumulative_in_flux", "cumulative_out_flux"]].mean()

        role_sets = (
            tdf.groupby("species_smiles")["row_role"].apply(lambda rows: set(str(x) for x in rows.dropna().tolist())).to_dict()
            if "row_role" in tdf.columns
            else {}
        )
        coproduct_set = (
            set(str(x) for x in tdf.loc[tdf["row_role"] == "co_product", "species_smiles"].dropna().tolist())
            if "row_role" in tdf.columns
            else set()
        )

        all_species = sorted(
            set(str(x) for x in tdf.get("species_smiles", pd.Series(dtype=str)).dropna().tolist())
            | set(str(x) for x in conc.get("species_smiles", pd.Series(dtype=str)).dropna().tolist())
            | set(str(x) for x in flux.get("species_smiles", pd.Series(dtype=str)).dropna().tolist())
        )
        for keep_smi in [reagent_smiles, product_smiles]:
            if keep_smi and keep_smi not in all_species:
                all_species.append(keep_smi)

        conc = add_initial_concentration_anchor(conc, all_species, reagent_smiles)
        flux = add_initial_flux_anchor(flux, all_species)

        conc_max = conc.groupby("species_smiles")["concentration_x"].max().sort_values(ascending=False)
        conc_species = list(conc_max.head(max_conc_species).index)
        for keep_smi in [reagent_smiles, product_smiles]:
            if keep_smi and keep_smi not in conc_species and keep_smi in conc_max.index:
                conc_species.append(keep_smi)

        fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=fig_dpi)
        inter_idx = 0
        for smi in conc_species:
            d = conc[conc["species_smiles"] == smi].sort_values("time_s")
            if d.empty:
                continue
            role = role_for_species(smi, role_sets, product_smiles, reagent_smiles, coproduct_set, role_priority, role_colors)
            if role == "intermediate":
                color = bulk_color_cycle[inter_idx % len(bulk_color_cycle)]
                inter_idx += 1
            else:
                color = role_colors.get(role, role_colors["unknown"])
            label = f"[P] {smi}" if smi == product_smiles else smi
            lw = 2.4 if smi == product_smiles else (2.0 if smi == reagent_smiles else 1.3)
            ax.plot(d["time_s"], d["concentration_x"], linewidth=lw, color=color, label=label)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Concentration X")
        ax.set_title(f"Species Concentration vs Time | product {pid}")
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.grid(False)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(conc_plot)
        plt.close(fig)

        pivot = conc.pivot_table(index="time_s", columns="species_smiles", values="concentration_x", aggfunc="mean").sort_index().fillna(0.0)
        if product_smiles not in pivot.columns:
            pivot[product_smiles] = 0.0
        p_series = pivot[product_smiles].astype(float)
        ps_series = pd.Series(0.0, index=pivot.index)
        for token, count in token_counts.items():
            if token in pivot.columns:
                ps_series = ps_series + float(count) * pivot[token].astype(float)

        fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=fig_dpi)
        ax.plot(pivot.index, p_series.values, linewidth=2.6, color=role_colors["product"], label=f"[P] {product_smiles}")
        ax.plot(
            pivot.index,
            ps_series.values,
            linewidth=2.1,
            linestyle="--",
            color=cp("Slate Line", "#3A4450"),
            label="[PS] product-state sum",
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Concentration X")
        ax.set_title(f"Product vs Product-State Sum | product {pid}")
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.grid(False)
        ax.legend(loc="best", fontsize=9, frameon=False)
        fig.tight_layout()
        fig.savefig(p_ps_plot)
        plt.close(fig)

        flux_strength = (
            flux.groupby("species_smiles")[["cumulative_in_flux", "cumulative_out_flux"]].max().max(axis=1).sort_values(ascending=False)
        )
        flux_species = list(flux_strength.head(max_flux_species).index)
        coproduct_species = [str(x) for x in tdf.loc[tdf["row_role"] == "co_product", "species_smiles"].dropna().tolist()] if "row_role" in tdf.columns else []
        for keep_smi in [reagent_smiles, product_smiles] + coproduct_species:
            if keep_smi and keep_smi not in flux_species and keep_smi in flux_strength.index:
                flux_species.append(keep_smi)

        fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=fig_dpi)
        inter_idx = 0
        for smi in flux_species:
            d = flux[flux["species_smiles"] == smi].sort_values("time_s")
            if d.empty:
                continue
            role = role_for_species(smi, role_sets, product_smiles, reagent_smiles, coproduct_set, role_priority, role_colors)
            if role == "intermediate":
                color = bulk_color_cycle[inter_idx % len(bulk_color_cycle)]
                inter_idx += 1
            else:
                color = role_colors.get(role, role_colors["unknown"])
            tag = "[P]" if smi == product_smiles else ("[R]" if smi == reagent_smiles else ("[CoP]" if smi in coproduct_set else ""))
            label_base = f"{tag} {smi}".strip()
            ax.plot(d["time_s"], d["cumulative_in_flux"], color=color, linewidth=1.9, linestyle="-", label=f"in: {label_base}")
            ax.plot(d["time_s"], d["cumulative_out_flux"], color=color, linewidth=1.3, linestyle="--", label=f"out: {label_base}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cumulative Flux")
        ax.set_title(f"Species Flux In/Out vs Time | product {pid}")
        ax.set_xlim(left=0.0)
        ax.set_ylim(bottom=0.0)
        ax.grid(False)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False, ncol=1)
        fig.tight_layout()
        fig.savefig(flux_plot)
        plt.close(fig)

        reaction_plot_written = False
        reaction_current_plot_written = False
        n_reaction_labels_r_to_p = 0
        n_reaction_labels_i_to_p = 0

        reaction_flux = rdf[rdf["row_kind"] == "reaction_flux"].copy() if "row_kind" in rdf.columns else pd.DataFrame()
        if not reaction_flux.empty and {"time_s", "cumulative_abs_flux"}.issubset(reaction_flux.columns):
            if "reaction_label" not in reaction_flux.columns:
                if {"from_smiles", "to_smiles"}.issubset(reaction_flux.columns):
                    reaction_flux["reaction_label"] = reaction_flux["from_smiles"].fillna("?") + " -> " + reaction_flux["to_smiles"].fillna("?")
                else:
                    reaction_flux["reaction_label"] = reaction_flux.get("orig_key", "reaction").astype(str)
            reaction_flux["time_s"] = pd.to_numeric(reaction_flux["time_s"], errors="coerce")
            reaction_flux["cumulative_abs_flux"] = pd.to_numeric(reaction_flux["cumulative_abs_flux"], errors="coerce")
            reaction_flux = reaction_flux.dropna(subset=["time_s", "cumulative_abs_flux", "reaction_label"])
            if not reaction_flux.empty:
                reaction_class_by_label = {}
                if {"reaction_label", "from_smiles", "to_smiles"}.issubset(reaction_flux.columns):
                    meta_cols = ["reaction_label", "from_smiles", "to_smiles"]
                    if "source_type" in reaction_flux.columns:
                        meta_cols.append("source_type")
                    rxn_meta = reaction_flux[meta_cols].drop_duplicates(subset=["reaction_label", "from_smiles", "to_smiles"], keep="last")
                    for _, row in rxn_meta.iterrows():
                        lbl = str(row.get("reaction_label", ""))
                        tag = classify_reaction_to_product(row, reagent_smiles, product_smiles)
                        if tag:
                            reaction_class_by_label[lbl] = tag
                n_reaction_labels_r_to_p = int(sum(1 for x in reaction_class_by_label.values() if x == "R->P"))
                n_reaction_labels_i_to_p = int(sum(1 for x in reaction_class_by_label.values() if x == "I->P"))

                reaction_plot_df = reaction_flux.groupby(["reaction_label", "time_s"], as_index=False)["cumulative_abs_flux"].mean().sort_values(["reaction_label", "time_s"])
                labels_all = sorted(reaction_plot_df["reaction_label"].dropna().astype(str).unique().tolist())
                base = reaction_plot_df[reaction_plot_df["time_s"] > 0.0].copy()
                anchor = pd.DataFrame({"reaction_label": labels_all, "time_s": 0.0, "cumulative_abs_flux": 0.0})
                reaction_plot_df = pd.concat([base, anchor], ignore_index=True)
                reaction_plot_df = reaction_plot_df.groupby(["reaction_label", "time_s"], as_index=False)["cumulative_abs_flux"].mean()
                final_by_rxn = (
                    reaction_plot_df.sort_values(["reaction_label", "time_s"])
                    .groupby("reaction_label", as_index=False)
                    .tail(1)
                    .sort_values("cumulative_abs_flux", ascending=False)
                )
                top_labels = final_by_rxn.head(max_reaction_lines)["reaction_label"].tolist()
                if top_labels:
                    style_map = {}
                    for i, lbl in enumerate(top_labels):
                        color = bulk_color_cycle[i % len(bulk_color_cycle)]
                        linestyle = ["-", "--", "-.", ":"][(i // len(bulk_color_cycle)) % 4]
                        style_map[lbl] = (color, linestyle)
                    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=fig_dpi)
                    for lbl in top_labels:
                        d = reaction_plot_df[reaction_plot_df["reaction_label"] == lbl].sort_values("time_s")
                        color, linestyle = style_map[lbl]
                        ax.plot(
                            d["time_s"],
                            d["cumulative_abs_flux"].clip(lower=0.0),
                            linewidth=2.0,
                            color=color,
                            linestyle=linestyle,
                            label=decorate_reaction_label(lbl, reaction_class_by_label),
                        )
                    ax.set_xlabel("Time (s)")
                    ax.set_ylabel("Cumulative abs flux")
                    ax.set_title(f"Top Reaction Cumulative Flux vs Time | product {pid}")
                    ax.set_xlim(left=0.0)
                    ax.set_ylim(bottom=0.0)
                    ax.grid(False)
                    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)
                    fig.tight_layout()
                    fig.savefig(rxn_flux_plot)
                    plt.close(fig)
                    reaction_plot_written = True

                    inst_rows = []
                    for lbl, grp in reaction_plot_df.groupby("reaction_label"):
                        g = grp.sort_values("time_s")[["time_s", "cumulative_abs_flux"]]
                        t = g["time_s"].to_numpy(dtype=float)
                        cvals = g["cumulative_abs_flux"].to_numpy(dtype=float)
                        if len(t) < 2:
                            continue
                        dt = np.diff(t)
                        dc = np.diff(cvals)
                        valid = dt > 0
                        curr = np.zeros_like(dc)
                        curr[valid] = dc[valid] / dt[valid]
                        curr = np.clip(curr, 0.0, None)
                        for ti, vi in zip(t[1:], curr):
                            inst_rows.append({"time_s": float(ti), "reaction_label": str(lbl), "current_abs_flux": float(vi)})
                    inst_df = pd.DataFrame(inst_rows)
                    if not inst_df.empty:
                        top_current = inst_df.groupby("reaction_label", as_index=False)["current_abs_flux"].max().sort_values("current_abs_flux", ascending=False)
                        top_current_labels = top_current.head(max_reaction_lines)["reaction_label"].tolist()
                        fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=fig_dpi)
                        for i, lbl in enumerate(top_current_labels):
                            d = inst_df[inst_df["reaction_label"] == lbl].sort_values("time_s")
                            if lbl in style_map:
                                color, linestyle = style_map[lbl]
                            else:
                                color = bulk_color_cycle[i % len(bulk_color_cycle)]
                                linestyle = ["-", "--", "-.", ":"][(i // len(bulk_color_cycle)) % 4]
                            ax.plot(
                                d["time_s"],
                                d["current_abs_flux"].clip(lower=0.0),
                                linewidth=2.0,
                                color=color,
                                linestyle=linestyle,
                                label=decorate_reaction_label(lbl, reaction_class_by_label),
                            )
                        ax.set_xlabel("Time (s)")
                        ax.set_ylabel("Current abs flux")
                        ax.set_title(f"Top Reaction Current Flux vs Time | product {pid}")
                        ax.set_xlim(left=0.0)
                        ax.set_ylim(bottom=0.0)
                        ax.grid(False)
                        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)
                        fig.tight_layout()
                        fig.savefig(rxn_current_plot)
                        plt.close(fig)
                        reaction_current_plot_written = True

        summary_rows.append(
            {
                "product_id": pid,
                "product_smiles": product_smiles,
                "profile_found": True,
                "profile_path": str(profile_path),
                "profile_size_mb": profile_path.stat().st_size / (1024 * 1024),
                "profile_rows": int(len(rdf)),
                "timepoints": int(conc["time_s"].nunique()) if not conc.empty else 0,
                "terminal_product_states": terminal_states,
                "expected_product_scale": expected_scale,
                "product_x_final": product_x_final,
                "completion_terminal_concentration": completion_terminal_conc,
                "top_nonproduct_smiles": top_nonproduct_smiles,
                "top_nonproduct_x": top_nonproduct_x,
                "plot_species_concentration": str(conc_plot),
                "plot_p_and_ps": str(p_ps_plot),
                "plot_flux_in_out": str(flux_plot),
                "plot_reaction_cumulative_flux": str(rxn_flux_plot) if reaction_plot_written else "",
                "plot_reaction_current_flux": str(rxn_current_plot) if reaction_current_plot_written else "",
                "n_reaction_labels_r_to_p": int(n_reaction_labels_r_to_p),
                "n_reaction_labels_i_to_p": int(n_reaction_labels_i_to_p),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("product_id") if summary_rows else pd.DataFrame()
    summary_df.to_csv(summary_csv, index=False)
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Products processed: {len(summary_df)}")
    if not summary_df.empty and "profile_found" in summary_df.columns:
        print(f"Profiles found: {int(summary_df['profile_found'].sum())}")
    print(f"Plot root: {out_root}")
    if not summary_df.empty:
        print(summary_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
