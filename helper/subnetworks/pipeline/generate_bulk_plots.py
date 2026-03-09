#!/usr/bin/env python3
"""Generate bulk per-product plots and summary CSV for one network output directory."""

import argparse
import os
import re
import subprocess
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
    """Return the default plotting config path."""
    cwd_cfg = Path("pipeline/configs/pipeline_config.yaml")
    if cwd_cfg.exists():
        return cwd_cfg
    return SCRIPT_DIR / "configs" / "pipeline_config.yaml"


def resolve_path(path_text, config_dir):
    """Resolve config-relative paths."""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (config_dir / path).resolve()


def load_config(config_path):
    """Load the pipeline config and config directory."""
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg, config_path.parent.resolve()


def read_parquet_with_arrow_retry(path):
    """Read parquet with an Arrow extension reset fallback for notebook kernels."""
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
    """Load parquet or pickle table path with pickle suffix fallback."""
    base_path = Path(base_path)
    if base_path.exists():
        if base_path.suffix == ".parquet":
            return read_parquet_with_arrow_retry(base_path), base_path
        return pd.read_pickle(base_path), base_path
    pkl = base_path.with_suffix(".pkl")
    if pkl.exists():
        return pd.read_pickle(pkl), pkl
    return pd.DataFrame(), None


def first_match_recursive(root, pattern):
    """Return the first recursive glob match or None."""
    root = Path(root)
    hits = sorted(root.rglob(pattern))
    return hits[0] if hits else None


def first_cantera_input_yaml(root):
    """Return cantera model YAML, skipping sidecar metadata YAML files."""
    root = Path(root)
    for path in sorted(root.rglob("*.yaml")):
        name = path.name
        if name.endswith(".species_map.yaml"):
            continue
        if name.endswith(".reaction_map.yaml"):
            continue
        if name.endswith(".run.yaml"):
            continue
        if name.endswith(".reactor_debug.yaml"):
            continue
        return path
    return None


def resolve_network_pickle_from_runtime(runtime_cfg_path, network_out):
    """Resolve the source network pickle path for a temp product work directory."""
    runtime_cfg_path = Path(runtime_cfg_path)
    network_out = Path(network_out)
    try:
        runtime_cfg = yaml.safe_load(runtime_cfg_path.read_text()) or {}
    except Exception:
        runtime_cfg = {}
    if isinstance(runtime_cfg, dict):
        sub_input = ((runtime_cfg.get("subnetwork_gen") or {}).get("input") or {})
        candidate = sub_input.get("pickle")
        if candidate:
            candidate_path = Path(candidate).expanduser().resolve()
            if candidate_path.exists():
                return candidate_path

    network_name = str(network_out.name)
    if network_name.startswith("networks__"):
        network_id = network_name.replace("networks__", "", 1)
        fallback = (REPO_ROOT / "networks" / f"{network_id}.pkl").resolve()
        if fallback.exists():
            return fallback
    return None


def build_profile_from_saved_tmp(
    *,
    network_out,
    product_id,
    profile_root,
    rebuild,
    verbose,
):
    """Build a per-product profile table from saved temp artifacts for debug plotting."""
    network_out = Path(network_out)
    profile_root = Path(profile_root)
    pid = str(product_id)
    profile_out = profile_root / f"debug_flux_timeseries__{pid}.parquet"
    if profile_out.exists() and not rebuild:
        return profile_out

    work_dir = network_out / ".tmp" / pid
    if not work_dir.exists():
        return None

    runtime_cfg = work_dir / "pipeline_runtime.yaml"
    if not runtime_cfg.exists():
        return None

    subnetwork_root = work_dir / "subnetworks"
    cantera_root = work_dir / "subnetwork_cantera_yaml"
    subnetwork_pickle = first_match_recursive(subnetwork_root, "*.pkl")
    cantera_yaml = first_cantera_input_yaml(cantera_root)
    to_final_csv = first_match_recursive(cantera_root, "*.to_final.csv")
    flux_ts_csv = first_match_recursive(cantera_root, "*.flux_timeseries.csv")
    network_pickle = resolve_network_pickle_from_runtime(runtime_cfg, network_out)

    required = {
        "runtime_cfg": runtime_cfg,
        "subnetwork_pickle": subnetwork_pickle,
        "cantera_yaml": cantera_yaml,
        "to_final_csv": to_final_csv,
        "flux_timeseries_csv": flux_ts_csv,
        "network_pickle": network_pickle,
    }
    missing = [name for name, value in required.items() if value is None or not Path(value).exists()]
    if missing:
        if verbose:
            print(f"[plot_warn] product {pid}: missing temp inputs ({', '.join(missing)}); cannot build debug profile.")
        return None

    table_root = profile_root / "_debug_product_tables"
    table_root.mkdir(parents=True, exist_ok=True)
    table_out = table_root / f"debug_product__{pid}.parquet"

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_product_table.py"),
        "--config",
        str(runtime_cfg),
        "--subnetwork-pickle",
        str(subnetwork_pickle),
        "--cantera-yaml",
        str(cantera_yaml),
        "--to-final-csv",
        str(to_final_csv),
        "--flux-timeseries-csv",
        str(flux_ts_csv),
        "--network-pickle",
        str(network_pickle),
        "--output",
        str(table_out),
        "--flux-output",
        str(profile_out),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Failed to build debug profile from saved temp files.\n"
            f"product_id={pid}\n"
            f"cmd={' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    if not profile_out.exists():
        raise RuntimeError(f"Expected debug profile not written for product_id={pid}: {profile_out}")
    if verbose:
        print(f"[plot_info] product {pid}: built temp-derived profile {profile_out.name}")
    return profile_out


def load_personal_colors(csv_path):
    """Load named hex colors from the personal palette CSV."""
    pal_df = pd.read_csv(csv_path)
    col_map = {str(c).strip().lower().lstrip("\ufeff"): c for c in pal_df.columns}
    name_col = col_map.get("color name") or col_map.get("name")
    hex_col = col_map.get("hex code") or col_map.get("hex")
    if not name_col or not hex_col:
        raise RuntimeError(
            "Palette CSV must include either (Color Name, Hex Code) or (name, hex). "
            f"Found: {list(pal_df.columns)}"
        )
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
    """Create a compact filesystem-safe name fragment."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))[:120]


def column_first_nonempty(df, column, default=""):
    if df is None or df.empty or column not in df.columns:
        return default
    series = df[column].dropna().astype(str).str.strip()
    series = series[(series != "") & (series.str.lower() != "nan")]
    return str(series.iloc[0]) if not series.empty else default


def column_first_numeric(df, column, default=np.nan):
    if df is None or df.empty or column not in df.columns:
        return float(default)
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    return float(values.iloc[0]) if not values.empty else float(default)


def save_placeholder_plot(path, *, title, message, dpi):
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=dpi)
    ax.axis("off")
    ax.text(0.5, 0.62, str(title), ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(0.5, 0.45, str(message), ha="center", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


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


def finite_positive_min(values):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if arr.size == 0:
        return np.nan
    return float(arr.min())


def plot_time_series(ax, x_values, y_values, *, log_time_axis, **plot_kwargs):
    x_arr = pd.to_numeric(pd.Series(x_values), errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(pd.Series(y_values), errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if log_time_axis:
        mask &= x_arr > 0.0
    if not np.any(mask):
        return np.nan
    ax.plot(x_arr[mask], y_arr[mask], **plot_kwargs)
    return finite_positive_min(x_arr[mask])


def apply_time_axis(ax, *, log_time_axis, min_positive_time):
    ax.set_xlabel("Time (s)")
    if log_time_axis and np.isfinite(min_positive_time) and min_positive_time > 0.0:
        ax.set_xscale("log")
        ax.set_xlim(left=float(min_positive_time))
    else:
        ax.set_xlim(left=0.0)


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
    plot_enabled = bool(plot_cfg.get("enabled", cfg.get("plot_all_network_flux_outputs", False)))
    if not plot_enabled:
        print("plot_export.enabled=false; skipping plot generation.")
        return

    network_out = Path(args.network_out_dir).expanduser().resolve()
    if not network_out.exists():
        raise FileNotFoundError(f"Network output directory not found: {network_out}")

    profile_pattern = str(plot_cfg.get("profile_pattern", "flux_timeseries*.parquet"))
    product_pattern = str(plot_cfg.get("product_pattern", "product_*.parquet"))
    output_subdir = str(plot_cfg.get("output_subdir", "bulk_profile_plots"))
    summary_filename = str(plot_cfg.get("summary_csv_name", "bulk_profile_summary.csv"))
    profile_source_mode = str(plot_cfg.get("profile_source_mode", "saved_flux_tables")).strip().lower()
    tmp_profile_subdir = str(plot_cfg.get("tmp_profile_subdir", "_debug_profiles_from_tmp"))
    rebuild_tmp_profiles = bool(plot_cfg.get("rebuild_tmp_profiles", False))
    tmp_profile_verbose = bool(plot_cfg.get("tmp_profile_verbose", bool(cfg.get("verbose", False))))
    max_conc_species = int(plot_cfg.get("max_conc_species", 12))
    max_flux_species = int(plot_cfg.get("max_flux_species", 10))
    max_reaction_lines = int(plot_cfg.get("max_reaction_lines", 12))
    fig_dpi = int(plot_cfg.get("fig_dpi", 170))
    reaction_cumulative_log_y = bool(plot_cfg.get("reaction_cumulative_log_y", True))
    reaction_current_log_y = bool(plot_cfg.get("reaction_current_log_y", True))
    reaction_log_floor = float(plot_cfg.get("reaction_log_floor", 1.0e-30))
    reaction_time_round_decimals = int(plot_cfg.get("reaction_time_round_decimals", 9))
    reaction_dt_fraction_floor = float(plot_cfg.get("reaction_dt_fraction_floor", 1.0e-3))
    reaction_abs_dt_floor_s = float(plot_cfg.get("reaction_abs_dt_floor_s", 1.0e-9))
    time_axis_log_raw = plot_cfg.get("time_axis_log", False)
    if isinstance(time_axis_log_raw, str):
        time_axis_log = time_axis_log_raw.strip().lower() in {"1", "true", "yes", "on"}
    else:
        time_axis_log = bool(time_axis_log_raw)
    palette_csv = resolve_path(str(plot_cfg.get("palette_csv", "../../color_palettes/color_wheel.csv")), cfg_dir)

    out_root = network_out / output_subdir
    summary_csv = out_root / summary_filename
    out_root.mkdir(parents=True, exist_ok=True)

    personal_colors = load_personal_colors(palette_csv)
    personal_colors_lower = {str(k).strip().lower(): v for k, v in personal_colors.items()}

    def cp(name, fallback):
        return personal_colors.get(name, personal_colors_lower.get(str(name).strip().lower(), fallback))

    edge_pink = cp("Edge Pink", "#FF5FA2")
    non_product_cycle = [
        cp("Ion Blue", "#1EA7FF"),
        cp("Neon Mint", "#5FF2D2"),
        cp("Electric Emerald", "#00C853"),
        cp("Gridiron Violet", "#6A37C8"),
        cp("Brass", "#D4B56E"),
    ]

    def style_for_rank(rank_idx):
        if rank_idx == 0:
            return edge_pink, "-"
        shifted = rank_idx - 1
        color = non_product_cycle[shifted % len(non_product_cycle)]
        linestyle = ["-", "--", "-.", ":"][(shifted // len(non_product_cycle)) % 4]
        return color, linestyle

    product_tables = sorted(network_out.glob(product_pattern))
    if not product_tables:
        product_tables = sorted(network_out.glob(product_pattern.replace(".parquet", ".pkl")))

    profile_map = {}
    if profile_source_mode in {
        "saved_flux_tables",
        "flux_tables",
        "flux_outputs",
        "retained_flux",
    }:
        for path in sorted(network_out.glob(profile_pattern)):
            df, loaded = load_table(path)
            if loaded is None or df.empty or "product_id" not in df.columns:
                continue
            ids = df["product_id"].dropna().astype(str)
            if not ids.empty:
                profile_map[ids.iloc[0]] = loaded
    elif profile_source_mode in {"tmp_saved_files", "tmp_files", "saved_tmp"}:
        tmp_profile_root = network_out / tmp_profile_subdir
        tmp_profile_root.mkdir(parents=True, exist_ok=True)
        for product_path in product_tables:
            pid = product_path.stem.replace("product_", "")
            try:
                built = build_profile_from_saved_tmp(
                    network_out=network_out,
                    product_id=pid,
                    profile_root=tmp_profile_root,
                    rebuild=rebuild_tmp_profiles,
                    verbose=tmp_profile_verbose,
                )
            except Exception as exc:
                print(f"[plot_warn] product {pid}: failed to build temp-derived profile ({exc})")
                built = None
            if built is not None:
                profile_map[str(pid)] = built
    else:
        raise ValueError(
            f"Unsupported plot_export.profile_source_mode={profile_source_mode!r}. "
            "Use saved_flux_tables or tmp_saved_files."
        )

    if profile_source_mode in {"tmp_saved_files", "tmp_files", "saved_tmp"}:
        print(f"Profile source mode: {profile_source_mode} (saved temp files)")
    else:
        print(f"Profile source mode: {profile_source_mode} ({profile_pattern})")

    summary_rows = []
    for product_path in product_tables:
        tdf, tdf_path = load_table(product_path)
        if tdf_path is None or tdf.empty:
            continue

        pid = product_path.stem.replace("product_", "")
        profile_path = profile_map.get(pid)
        rdf, _ = load_table(profile_path) if profile_path is not None else (pd.DataFrame(), None)
        missing_profile_message = (
            "No temp-derived profile found under .tmp for this product."
            if profile_source_mode in {"tmp_saved_files", "tmp_files", "saved_tmp"}
            else "No flux_timeseries profile found for this product."
        )

        product_smiles = column_first_nonempty(tdf, "product_smiles", default="")
        reagent_smiles = column_first_nonempty(tdf, "reagent_smiles", default="")
        if not product_smiles:
            product_smiles = column_first_nonempty(rdf, "product_smiles", default="")
        if not reagent_smiles:
            reagent_smiles = column_first_nonempty(rdf, "reagent_smiles", default="")
        if not product_smiles:
            product_smiles = pid

        terminal_states = column_first_nonempty(tdf, "terminal_product_states", default="")
        if not terminal_states:
            terminal_states = column_first_nonempty(rdf, "terminal_product_states", default="")
        if not terminal_states:
            terminal_states = product_smiles

        completion_terminal_conc = column_first_numeric(tdf, "completion_terminal_concentration", default=np.nan)
        if not np.isfinite(completion_terminal_conc):
            completion_terminal_conc = column_first_numeric(rdf, "completion_terminal_concentration", default=np.nan)

        token_counts = parse_terminal_token_counts(terminal_states, product_smiles)
        expected_scale = 0.0
        if token_counts:
            total_tokens = sum(token_counts.values())
            expected_scale = token_counts.get(product_smiles, 0) / total_tokens if total_tokens > 0 else 0.0

        plot_dir = out_root / f"product_{pid}__{safe_name(product_smiles)}"
        plot_dir.mkdir(parents=True, exist_ok=True)

        conc_plot = plot_dir / "concentration_species_vs_time.png"
        p_ps_plot = plot_dir / "concentration_P_and_PS_vs_time.png"
        flux_plot = plot_dir / "flux_in_out_vs_time.png"
        rxn_flux_plot = plot_dir / "reaction_top_cumulative_flux_vs_time.png"
        rxn_current_plot = plot_dir / "reaction_top_current_flux_vs_time.png"
        rxn_fraction_plot = plot_dir / "reaction_terminal_fractional_flux_vs_time.png"

        if rdf.empty:
            save_placeholder_plot(
                conc_plot,
                title=f"product {pid}: species concentration",
                message=missing_profile_message,
                dpi=fig_dpi,
            )
            save_placeholder_plot(
                p_ps_plot,
                title=f"product {pid}: [P] and [PS]",
                message=missing_profile_message,
                dpi=fig_dpi,
            )
            save_placeholder_plot(
                flux_plot,
                title=f"product {pid}: species flux in/out",
                message=missing_profile_message,
                dpi=fig_dpi,
            )
            save_placeholder_plot(
                rxn_flux_plot,
                title=f"product {pid}: reaction cumulative flux",
                message=missing_profile_message,
                dpi=fig_dpi,
            )
            save_placeholder_plot(
                rxn_current_plot,
                title=f"product {pid}: reaction current flux",
                message=missing_profile_message,
                dpi=fig_dpi,
            )
            save_placeholder_plot(
                rxn_fraction_plot,
                title=f"product {pid}: terminal reaction fractional flux",
                message=missing_profile_message,
                dpi=fig_dpi,
            )
            summary_rows.append(
                {
                    "product_id": pid,
                    "product_smiles": product_smiles,
                    "profile_source_mode": profile_source_mode,
                    "profile_found": False,
                    "profile_path": "",
                    "profile_size_mb": np.nan,
                    "profile_rows": 0,
                    "timepoints": 0,
                    "terminal_product_states": terminal_states,
                    "expected_product_scale": expected_scale,
                    "product_x_final": np.nan,
                    "completion_terminal_concentration": completion_terminal_conc,
                    "top_nonproduct_smiles": "",
                    "top_nonproduct_x": np.nan,
                    "plot_species_concentration": str(conc_plot),
                    "plot_p_and_ps": str(p_ps_plot),
                    "plot_flux_in_out": str(flux_plot),
                    "plot_reaction_cumulative_flux": str(rxn_flux_plot),
                    "plot_reaction_current_flux": str(rxn_current_plot),
                    "plot_reaction_terminal_fraction": str(rxn_fraction_plot),
                    "plot_time_axis_log": bool(time_axis_log),
                    "n_reaction_labels_r_to_p": 0,
                    "n_reaction_labels_i_to_p": 0,
                }
            )
            continue

        conc = (
            rdf[rdf.get("row_kind", "") == "species_concentration"][["time_s", "species_smiles", "concentration_x"]].copy()
            if "row_kind" in rdf.columns
            else pd.DataFrame()
        )
        if not conc.empty:
            conc["time_s"] = pd.to_numeric(conc["time_s"], errors="coerce")
            conc["concentration_x"] = pd.to_numeric(conc["concentration_x"], errors="coerce")
            conc = conc.dropna(subset=["time_s", "species_smiles", "concentration_x"])
            conc = conc.groupby(["time_s", "species_smiles"], as_index=False)["concentration_x"].mean()

        flux = (
            rdf[rdf.get("row_kind", "") == "species_flux"][["time_s", "species_smiles", "cumulative_in_flux", "cumulative_out_flux"]].copy()
            if "row_kind" in rdf.columns
            else pd.DataFrame()
        )
        if not flux.empty:
            flux["time_s"] = pd.to_numeric(flux["time_s"], errors="coerce")
            flux["cumulative_in_flux"] = pd.to_numeric(flux["cumulative_in_flux"], errors="coerce")
            flux["cumulative_out_flux"] = pd.to_numeric(flux["cumulative_out_flux"], errors="coerce")
            flux = flux.dropna(subset=["time_s", "species_smiles", "cumulative_in_flux", "cumulative_out_flux"])
            flux = flux.groupby(["time_s", "species_smiles"], as_index=False)[["cumulative_in_flux", "cumulative_out_flux"]].mean()

        role_source = rdf.copy()
        if "row_kind" in role_source.columns:
            role_source = role_source[role_source["row_kind"].isin(["species_flux", "species_concentration"])].copy()
        coproduct_set = set()
        if {"species_smiles", "row_role"}.issubset(role_source.columns):
            role_source["species_smiles"] = role_source["species_smiles"].astype(str)
            role_source["row_role"] = role_source["row_role"].astype(str)
            coproduct_set = set(role_source.loc[role_source["row_role"] == "co_product", "species_smiles"].dropna().astype(str).tolist())

        all_species = sorted(
            set(str(x) for x in role_source.get("species_smiles", pd.Series(dtype=str)).dropna().tolist())
            | set(str(x) for x in conc.get("species_smiles", pd.Series(dtype=str)).dropna().tolist())
            | set(str(x) for x in flux.get("species_smiles", pd.Series(dtype=str)).dropna().tolist())
        )
        for keep_smi in [reagent_smiles, product_smiles]:
            if keep_smi and keep_smi not in all_species:
                all_species.append(keep_smi)

        product_x_final = np.nan
        top_nonproduct_smiles = ""
        top_nonproduct_x = np.nan
        if not conc.empty:
            conc = add_initial_concentration_anchor(conc, all_species, reagent_smiles)
            final_conc = (
                conc.sort_values(["species_smiles", "time_s"])
                .groupby("species_smiles", as_index=False)
                .tail(1)
            )
            if product_smiles in final_conc["species_smiles"].astype(str).tolist():
                p_rows = final_conc[final_conc["species_smiles"].astype(str) == product_smiles]
                if not p_rows.empty:
                    product_x_final = float(p_rows["concentration_x"].iloc[0])
            nonprod = final_conc[final_conc["species_smiles"].astype(str) != product_smiles].sort_values(
                "concentration_x", ascending=False
            )
            if not nonprod.empty:
                top_nonproduct_smiles = str(nonprod.iloc[0]["species_smiles"])
                top_nonproduct_x = float(nonprod.iloc[0]["concentration_x"])
        elif {"row_role", "species_smiles", "final_concentration"}.issubset(tdf.columns):
            product_rows = tdf[tdf["row_role"].astype(str) == "product"]
            if not product_rows.empty:
                product_x_final = float(pd.to_numeric(product_rows["final_concentration"], errors="coerce").dropna().iloc[0])
            nonprod = tdf[tdf["row_role"].astype(str) != "product"][["species_smiles", "final_concentration"]].copy()
            nonprod["final_concentration"] = pd.to_numeric(nonprod["final_concentration"], errors="coerce")
            nonprod = nonprod.dropna(subset=["final_concentration"]).sort_values("final_concentration", ascending=False)
            if not nonprod.empty:
                top_nonproduct_smiles = str(nonprod.iloc[0]["species_smiles"])
                top_nonproduct_x = float(nonprod.iloc[0]["final_concentration"])

        if not flux.empty:
            flux = add_initial_flux_anchor(flux, all_species)

        if conc.empty:
            save_placeholder_plot(
                conc_plot,
                title=f"product {pid}: species concentration",
                message="No species_concentration rows found in random profile.",
                dpi=fig_dpi,
            )
            save_placeholder_plot(
                p_ps_plot,
                title=f"product {pid}: [P] and [PS]",
                message="No species_concentration rows found in random profile.",
                dpi=fig_dpi,
            )
        else:
            conc_max = conc.groupby("species_smiles")["concentration_x"].max().sort_values(ascending=False)
            conc_species = list(conc_max.head(max_conc_species).index)
            for keep_smi in [reagent_smiles, product_smiles]:
                if keep_smi and keep_smi not in conc_species and keep_smi in conc_max.index:
                    conc_species.append(keep_smi)

            fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=fig_dpi)
            non_product_idx = 0
            conc_time_min = np.nan
            for smi in conc_species:
                d = conc[conc["species_smiles"] == smi].sort_values("time_s")
                if d.empty:
                    continue
                if smi == product_smiles:
                    color = edge_pink
                else:
                    color = non_product_cycle[non_product_idx % len(non_product_cycle)]
                    non_product_idx += 1
                label = f"[P] {smi}" if smi == product_smiles else smi
                lw = 2.4 if smi == product_smiles else (2.0 if smi == reagent_smiles else 1.3)
                line_time_min = plot_time_series(
                    ax,
                    d["time_s"],
                    d["concentration_x"],
                    log_time_axis=time_axis_log,
                    linewidth=lw,
                    color=color,
                    label=label,
                )
                if np.isfinite(line_time_min):
                    conc_time_min = min(conc_time_min, line_time_min) if np.isfinite(conc_time_min) else line_time_min
            apply_time_axis(ax, log_time_axis=time_axis_log, min_positive_time=conc_time_min)
            ax.set_ylabel("Concentration X")
            ax.set_title(f"Species Concentration vs Time | product {pid}")
            ax.set_ylim(bottom=0.0)
            ax.grid(False)
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
            fig.tight_layout()
            fig.savefig(conc_plot)
            plt.close(fig)

            pivot = (
                conc.pivot_table(index="time_s", columns="species_smiles", values="concentration_x", aggfunc="mean")
                .sort_index()
                .fillna(0.0)
            )
            if product_smiles not in pivot.columns:
                pivot[product_smiles] = 0.0
            p_series = pivot[product_smiles].astype(float)
            ps_series = pd.Series(0.0, index=pivot.index)
            for token, count in token_counts.items():
                if token in pivot.columns:
                    ps_series = ps_series + float(count) * pivot[token].astype(float)

            fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=fig_dpi)
            pps_time_min = np.nan
            p_time_min = plot_time_series(
                ax,
                pivot.index,
                p_series.values,
                log_time_axis=time_axis_log,
                linewidth=2.6,
                color=edge_pink,
                label=f"[P] {product_smiles}",
            )
            if np.isfinite(p_time_min):
                pps_time_min = p_time_min
            ps_time_min = plot_time_series(
                ax,
                pivot.index,
                ps_series.values,
                log_time_axis=time_axis_log,
                linewidth=2.1,
                linestyle="--",
                color=non_product_cycle[0],
                label="[PS] product-state sum",
            )
            if np.isfinite(ps_time_min):
                pps_time_min = min(pps_time_min, ps_time_min) if np.isfinite(pps_time_min) else ps_time_min
            apply_time_axis(ax, log_time_axis=time_axis_log, min_positive_time=pps_time_min)
            ax.set_ylabel("Concentration X")
            ax.set_title(f"Product vs Product-State Sum | product {pid}")
            ax.set_ylim(bottom=0.0)
            ax.grid(False)
            ax.legend(loc="best", fontsize=9, frameon=False)
            fig.tight_layout()
            fig.savefig(p_ps_plot)
            plt.close(fig)

        if flux.empty:
            save_placeholder_plot(
                flux_plot,
                title=f"product {pid}: species flux in/out",
                message="No species_flux rows found in random profile.",
                dpi=fig_dpi,
            )
        else:
            flux_strength = (
                flux.groupby("species_smiles")[["cumulative_in_flux", "cumulative_out_flux"]]
                .max()
                .max(axis=1)
                .sort_values(ascending=False)
            )
            flux_species = list(flux_strength.head(max_flux_species).index)
            for keep_smi in [reagent_smiles, product_smiles] + sorted(coproduct_set):
                if keep_smi and keep_smi not in flux_species and keep_smi in flux_strength.index:
                    flux_species.append(keep_smi)

            fig, ax = plt.subplots(figsize=(9.0, 5.2), dpi=fig_dpi)
            non_product_idx = 0
            flux_time_min = np.nan
            for smi in flux_species:
                d = flux[flux["species_smiles"] == smi].sort_values("time_s")
                if d.empty:
                    continue
                if smi == product_smiles:
                    color = edge_pink
                else:
                    color = non_product_cycle[non_product_idx % len(non_product_cycle)]
                    non_product_idx += 1
                tag = "[P]" if smi == product_smiles else ("[R]" if smi == reagent_smiles else ("[CoP]" if smi in coproduct_set else ""))
                label_base = f"{tag} {smi}".strip()
                in_time_min = plot_time_series(
                    ax,
                    d["time_s"],
                    d["cumulative_in_flux"],
                    log_time_axis=time_axis_log,
                    color=color,
                    linewidth=1.9,
                    linestyle="-",
                    label=f"in: {label_base}",
                )
                if np.isfinite(in_time_min):
                    flux_time_min = min(flux_time_min, in_time_min) if np.isfinite(flux_time_min) else in_time_min
                out_time_min = plot_time_series(
                    ax,
                    d["time_s"],
                    d["cumulative_out_flux"],
                    log_time_axis=time_axis_log,
                    color=color,
                    linewidth=1.3,
                    linestyle="--",
                    label=f"out: {label_base}",
                )
                if np.isfinite(out_time_min):
                    flux_time_min = min(flux_time_min, out_time_min) if np.isfinite(flux_time_min) else out_time_min
            apply_time_axis(ax, log_time_axis=time_axis_log, min_positive_time=flux_time_min)
            ax.set_ylabel("Cumulative Flux")
            ax.set_title(f"Species Flux In/Out vs Time | product {pid}")
            ax.set_ylim(bottom=0.0)
            ax.grid(False)
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False, ncol=1)
            fig.tight_layout()
            fig.savefig(flux_plot)
            plt.close(fig)

        reaction_plot_written = False
        reaction_current_plot_written = False
        reaction_fraction_plot_written = False
        n_reaction_labels_r_to_p = 0
        n_reaction_labels_i_to_p = 0
        small_dt_points_filtered = 0

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
                reaction_display_label = {}
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
                        source_state = str(row.get("from_smiles", "")).strip()
                        reaction_display_label[lbl] = (
                            f"{source_state} [{tag}]"
                            if (source_state and tag)
                            else (f"{lbl} [{tag}]" if tag else lbl)
                        )
                n_reaction_labels_r_to_p = int(sum(1 for x in reaction_class_by_label.values() if x == "R->P"))
                n_reaction_labels_i_to_p = int(sum(1 for x in reaction_class_by_label.values() if x == "I->P"))

                reaction_plot_df = reaction_flux.groupby(["reaction_label", "time_s"], as_index=False)["cumulative_abs_flux"].mean().sort_values(["reaction_label", "time_s"])
                reaction_plot_df["time_s"] = pd.to_numeric(reaction_plot_df["time_s"], errors="coerce").round(
                    reaction_time_round_decimals
                )
                reaction_plot_df = reaction_plot_df.dropna(subset=["time_s"])
                reaction_plot_df = (
                    reaction_plot_df.groupby(["reaction_label", "time_s"], as_index=False)["cumulative_abs_flux"].mean()
                    .sort_values(["reaction_label", "time_s"])
                )
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
                        style_map[lbl] = style_for_rank(i)
                    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=fig_dpi)
                    rxn_cum_time_min = np.nan
                    for lbl in top_labels:
                        d = reaction_plot_df[reaction_plot_df["reaction_label"] == lbl].sort_values("time_s")
                        color, linestyle = style_map[lbl]
                        y_vals = d["cumulative_abs_flux"].clip(lower=0.0)
                        if reaction_cumulative_log_y:
                            y_vals = y_vals.clip(lower=reaction_log_floor)
                        line_time_min = plot_time_series(
                            ax,
                            d["time_s"],
                            y_vals,
                            log_time_axis=time_axis_log,
                            linewidth=2.0,
                            color=color,
                            linestyle=linestyle,
                            label=decorate_reaction_label(lbl, reaction_class_by_label),
                        )
                        if np.isfinite(line_time_min):
                            rxn_cum_time_min = min(rxn_cum_time_min, line_time_min) if np.isfinite(rxn_cum_time_min) else line_time_min
                    apply_time_axis(ax, log_time_axis=time_axis_log, min_positive_time=rxn_cum_time_min)
                    ax.set_ylabel("Cumulative abs flux")
                    ax.set_title(f"Top Reaction Cumulative Flux vs Time | product {pid}")
                    if reaction_cumulative_log_y:
                        ax.set_yscale("log")
                    else:
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
                        positive_dt = dt[dt > 0.0]
                        if positive_dt.size == 0:
                            continue
                        median_dt = float(np.median(positive_dt))
                        dt_floor = max(float(reaction_abs_dt_floor_s), median_dt * float(reaction_dt_fraction_floor))
                        valid = dt >= dt_floor
                        small_dt_points_filtered += int(np.logical_and(dt > 0.0, dt < dt_floor).sum())
                        curr = np.zeros_like(dc)
                        curr[valid] = dc[valid] / dt[valid]
                        curr = np.clip(curr, 0.0, None)
                        for ti, vi in zip(t[1:], curr):
                            inst_rows.append({"time_s": float(ti), "reaction_label": str(lbl), "current_abs_flux": float(vi)})
                    inst_df = pd.DataFrame(inst_rows)
                    if not inst_df.empty:
                        top_current = (
                            inst_df.groupby("reaction_label", as_index=False)["current_abs_flux"]
                            .max()
                            .sort_values("current_abs_flux", ascending=False)
                        )
                        top_current_labels = top_current.head(max_reaction_lines)["reaction_label"].tolist()
                        fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=fig_dpi)
                        rxn_current_time_min = np.nan
                        for i, lbl in enumerate(top_current_labels):
                            d = inst_df[inst_df["reaction_label"] == lbl].sort_values("time_s")
                            if lbl in style_map:
                                color, linestyle = style_map[lbl]
                            else:
                                color, linestyle = style_for_rank(i)
                            y_vals = d["current_abs_flux"].clip(lower=0.0)
                            if reaction_current_log_y:
                                y_vals = y_vals.clip(lower=reaction_log_floor)
                            line_time_min = plot_time_series(
                                ax,
                                d["time_s"],
                                y_vals,
                                log_time_axis=time_axis_log,
                                linewidth=2.0,
                                color=color,
                                linestyle=linestyle,
                                label=decorate_reaction_label(lbl, reaction_class_by_label),
                            )
                            if np.isfinite(line_time_min):
                                rxn_current_time_min = (
                                    min(rxn_current_time_min, line_time_min)
                                    if np.isfinite(rxn_current_time_min)
                                    else line_time_min
                                )
                        apply_time_axis(ax, log_time_axis=time_axis_log, min_positive_time=rxn_current_time_min)
                        ax.set_ylabel("Current abs flux")
                        ax.set_title(f"Top Reaction Current Flux vs Time | product {pid}")
                        if reaction_current_log_y:
                            ax.set_yscale("log")
                        else:
                            ax.set_ylim(bottom=0.0)
                        ax.grid(False)
                        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)
                        fig.tight_layout()
                        fig.savefig(rxn_current_plot)
                        plt.close(fig)
                        reaction_current_plot_written = True

                terminal_label_set = {lbl for lbl, tag in reaction_class_by_label.items() if tag in {"R->P", "I->P"}}
                terminal_labels = [lbl for lbl in final_by_rxn["reaction_label"].astype(str).tolist() if lbl in terminal_label_set]
                if not terminal_labels:
                    terminal_labels = sorted(terminal_label_set)
                if terminal_labels:
                    terminal_df = reaction_plot_df[reaction_plot_df["reaction_label"].isin(terminal_labels)].copy()
                    if not terminal_df.empty:
                        terminal_df["cumulative_abs_flux"] = terminal_df["cumulative_abs_flux"].clip(lower=0.0)
                        denom = terminal_df.groupby("time_s", as_index=False)["cumulative_abs_flux"].sum()
                        denom = denom.rename(columns={"cumulative_abs_flux": "total_terminal_flux"})
                        terminal_df = terminal_df.merge(denom, on="time_s", how="left")
                        terminal_df["fractional_flux"] = np.where(
                            terminal_df["total_terminal_flux"] > 0.0,
                            terminal_df["cumulative_abs_flux"] / terminal_df["total_terminal_flux"],
                            0.0,
                        )

                        fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=fig_dpi)
                        style_map_terminal = {}
                        for i, lbl in enumerate(terminal_labels):
                            style_map_terminal[lbl] = style_for_rank(i)
                        rxn_frac_time_min = np.nan
                        for lbl in terminal_labels:
                            d = terminal_df[terminal_df["reaction_label"] == lbl].sort_values("time_s")
                            if d.empty:
                                continue
                            color, linestyle = style_map_terminal[lbl]
                            line_time_min = plot_time_series(
                                ax,
                                d["time_s"],
                                d["fractional_flux"],
                                log_time_axis=time_axis_log,
                                linewidth=1.9,
                                color=color,
                                linestyle=linestyle,
                                label=reaction_display_label.get(lbl, decorate_reaction_label(lbl, reaction_class_by_label)),
                            )
                            if np.isfinite(line_time_min):
                                rxn_frac_time_min = min(rxn_frac_time_min, line_time_min) if np.isfinite(rxn_frac_time_min) else line_time_min
                        apply_time_axis(ax, log_time_axis=time_axis_log, min_positive_time=rxn_frac_time_min)
                        ax.set_ylabel("Fractional flux to P")
                        ax.set_title(f"Terminal Reaction Fractional Flux vs Time | product {pid}")
                        ax.set_ylim(0.0, 1.0)
                        ax.grid(False)
                        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=7, frameon=False)
                        fig.tight_layout()
                        fig.savefig(rxn_fraction_plot)
                        plt.close(fig)
                        reaction_fraction_plot_written = True

        if not reaction_plot_written:
            save_placeholder_plot(
                rxn_flux_plot,
                title=f"product {pid}: reaction cumulative flux",
                message="No reaction_flux rows with cumulative_abs_flux were available.",
                dpi=fig_dpi,
            )
        if not reaction_current_plot_written:
            save_placeholder_plot(
                rxn_current_plot,
                title=f"product {pid}: reaction current flux",
                message="Insufficient reaction_flux timeseries to compute current flux.",
                dpi=fig_dpi,
            )
        if not reaction_fraction_plot_written:
            save_placeholder_plot(
                rxn_fraction_plot,
                title=f"product {pid}: terminal reaction fractional flux",
                message="No terminal reaction labels (R->P / I->P) were detected.",
                dpi=fig_dpi,
            )

        summary_rows.append(
            {
                "product_id": pid,
                "product_smiles": product_smiles,
                "profile_source_mode": profile_source_mode,
                "profile_found": True,
                "profile_path": str(profile_path) if profile_path is not None else "",
                "profile_size_mb": (
                    profile_path.stat().st_size / (1024 * 1024)
                    if profile_path is not None and Path(profile_path).exists()
                    else np.nan
                ),
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
                "plot_reaction_cumulative_flux": str(rxn_flux_plot),
                "plot_reaction_current_flux": str(rxn_current_plot),
                "plot_reaction_terminal_fraction": str(rxn_fraction_plot),
                "plot_time_axis_log": bool(time_axis_log),
                "n_reaction_labels_r_to_p": int(n_reaction_labels_r_to_p),
                "n_reaction_labels_i_to_p": int(n_reaction_labels_i_to_p),
                "current_flux_small_dt_points_filtered": int(small_dt_points_filtered),
            }
        )

        if small_dt_points_filtered > 0:
            print(
                f"[plot_qc] product {pid}: filtered {small_dt_points_filtered} tiny-dt derivative points "
                f"(likely adaptive-extension boundary near target window edges)."
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
