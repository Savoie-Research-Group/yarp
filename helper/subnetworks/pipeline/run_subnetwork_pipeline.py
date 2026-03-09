#!/usr/bin/env python3
"""Run one-network streaming subnetwork kinetics pipeline."""

import argparse
import copy
import os
import shutil
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

OUTPUT_MODE_PRESETS = {
    "debug": {
        "description": "retain all temp/profile artifacts and generate all network flux plots",
        "retain_terminal_table": True,
        "save_flux_for_every_subnetwork": True,
        "plot_all_network_flux_outputs": True,
        "cleanup_mode": "keep",
        "plot_profile_source_mode": "saved_flux_tables",
        "merge_all_tables_per_network": True,
        "remove_source_tables_after_full_merge": False,
        "clear_previous_outputs": True,
        "retained_timeseries_downsample_seconds": 0.0,
    },
    "subnetwork": {
        "description": "keep per-product tables plus merged network table",
        "retain_terminal_table": False,
        "save_flux_for_every_subnetwork": False,
        "plot_all_network_flux_outputs": False,
        "cleanup_mode": "clean",
        "plot_profile_source_mode": "saved_flux_tables",
        "merge_all_tables_per_network": True,
        "remove_source_tables_after_full_merge": False,
        "clear_previous_outputs": True,
        "retained_timeseries_downsample_seconds": 1.0,
    },
    "production": {
        "description": "keep only merged network output table",
        "retain_terminal_table": False,
        "save_flux_for_every_subnetwork": False,
        "plot_all_network_flux_outputs": False,
        "cleanup_mode": "clean",
        "plot_profile_source_mode": "saved_flux_tables",
        "merge_all_tables_per_network": True,
        "remove_source_tables_after_full_merge": True,
        "clear_previous_outputs": True,
        "retained_timeseries_downsample_seconds": 1.0,
    },
}

PRODUCT_RUN_STEPS = ("subnetwork", "cantera_yaml", "cantera_run")


def default_config_path(verbose=False):
    """Return the default pipeline config path."""
    cwd_cfg = Path("pipeline/configs/pipeline_config.yaml")
    if cwd_cfg.exists():
        if verbose:
            print(f"[verbose] Using config at {cwd_cfg}")
        return cwd_cfg
    fallback = SCRIPT_DIR / "configs" / "pipeline_config.yaml"
    if verbose:
        print(f"[verbose] Using fallback config at {fallback}")
    return fallback


def resolve_path(path_text, cfg_dir, verbose=False):
    """Resolve a config path relative to the config directory."""
    path = Path(path_text)
    if path.is_absolute():
        if verbose:
            print(f"[verbose] resolve_path absolute={path}")
        return path
    resolved = (cfg_dir / path).resolve()
    if verbose:
        print(f"[verbose] resolve_path relative={path} resolved={resolved}")
    return resolved


def load_config(config_path, verbose=False):
    """Load the pipeline config mapping and config directory."""
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    if verbose:
        print(f"[verbose] Loaded config mapping from {config_path}")
    return cfg, config_path.parent.resolve()


def normalize_output_mode(raw_mode):
    """Map output-mode text to one of: debug, subnetwork, production."""
    text = str(raw_mode or "subnetwork").strip().lower()
    if text not in OUTPUT_MODE_PRESETS:
        valid = ", ".join(sorted(OUTPUT_MODE_PRESETS.keys()))
        raise ValueError(f"Unsupported output_mode={raw_mode!r}. Valid modes: {valid}")
    return text


def apply_output_mode_defaults(cfg, verbose=False):
    """Apply output behavior from one output_mode preset and return chosen mode."""
    output_cfg = cfg.get("output", {}) or {}
    raw_mode = cfg.get("output_mode", output_cfg.get("mode", "subnetwork"))
    mode = normalize_output_mode(raw_mode)
    preset = OUTPUT_MODE_PRESETS[mode]
    cfg["output_mode"] = mode

    # Mode presets define defaults; explicit config keys can override.
    cfg["save_flux_for_every_subnetwork"] = bool(
        cfg.get("save_flux_for_every_subnetwork", preset["save_flux_for_every_subnetwork"])
    )
    cfg["plot_all_network_flux_outputs"] = bool(
        cfg.get("plot_all_network_flux_outputs", preset["plot_all_network_flux_outputs"])
    )
    cfg["write_flux_timeseries_table"] = bool(
        cfg.get(
            "write_flux_timeseries_table",
            cfg["save_flux_for_every_subnetwork"] or cfg["plot_all_network_flux_outputs"],
        )
    )
    cfg["cleanup_mode"] = str(preset["cleanup_mode"])
    cfg["merge_all_tables_per_network"] = bool(preset["merge_all_tables_per_network"])
    cfg["remove_source_tables_after_full_merge"] = bool(preset["remove_source_tables_after_full_merge"])
    cfg["clear_previous_outputs"] = bool(preset["clear_previous_outputs"])
    cfg["save_terminal_list"] = bool(preset["retain_terminal_table"])

    datatable_cfg = cfg.get("datatable", {}) or {}
    datatable_cfg["retained_timeseries_downsample_seconds"] = float(
        preset["retained_timeseries_downsample_seconds"]
    )
    cfg["datatable"] = datatable_cfg

    plot_cfg = cfg.get("plot_export", {}) or {}
    plot_cfg["enabled"] = bool(cfg["plot_all_network_flux_outputs"])
    plot_cfg["profile_source_mode"] = str(
        plot_cfg.get("profile_source_mode", preset["plot_profile_source_mode"])
    )
    plot_cfg["profile_pattern"] = str(plot_cfg.get("profile_pattern", "flux_timeseries*.parquet"))
    cfg["plot_export"] = plot_cfg

    if verbose:
        print(
            "[verbose] output mode resolved:",
            f"output_mode={mode}",
            f"description={preset['description']}",
        )
    return mode


def load_manifest(path, verbose=False):
    """Load non-comment manifest entries."""
    rows = []
    with path.open("r") as f:
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            rows.append(text)
    if not rows:
        raise RuntimeError(f"Manifest has no network paths: {path}")
    if verbose:
        print(f"[verbose] Loaded manifest: {path}")
        print(f"[verbose] Manifest entries: {len(rows)}")
    return rows


def task_index_from_env(env_name, verbose=False):
    """Return a 1-based task index from an environment variable."""
    raw = os.environ.get(env_name)
    if raw is None:
        if verbose:
            print(f"[verbose] {env_name} not set; defaulting task index to 1")
        return 1
    value = int(raw)
    if value < 1:
        raise ValueError(f"{env_name} must be >= 1")
    return value


def safe_label(text, verbose=False):
    """Convert arbitrary text into a filesystem-safe label."""
    value = "".join(ch if (ch.isalnum() or ch in "-._") else "_" for ch in str(text))
    if verbose:
        print(f"[verbose] safe_label input={text} output={value}")
    return value


def run_cmd(cmd, env, verbose=False):
    """Run a command in the repository root."""
    print("Running:", " ".join(cmd))
    if verbose:
        print(f"[verbose] cwd={REPO_ROOT}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def resolve_product_run_steps(cfg):
    """Normalize run_steps to product-level stages executed inside run_one_product."""
    configured_steps = cfg.get("run_steps", []) or []
    if not configured_steps:
        return list(PRODUCT_RUN_STEPS)

    no_op_steps = {"dummy_barriers", "terminal_products", "datatable"}
    allowed_steps = set(PRODUCT_RUN_STEPS)
    resolved = []
    for raw_step in configured_steps:
        step = str(raw_step).strip()
        if not step or step in no_op_steps:
            continue
        if step not in allowed_steps:
            valid = ", ".join(PRODUCT_RUN_STEPS)
            raise ValueError(f"Unsupported run_steps entry: {raw_step!r}. Valid product steps: {valid}")
        if step not in resolved:
            resolved.append(step)

    return resolved or list(PRODUCT_RUN_STEPS)


def verify_yarp_import(env, expected_root=None):
    """Print the imported YARP module path and validate root when configured."""
    probe = [
        sys.executable,
        "-c",
        (
            "import yarp.network.network as m; "
            "print(m.__file__)"
        ),
    ]
    result = subprocess.run(
        probe,
        check=True,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    imported = Path(result.stdout.strip().splitlines()[-1]).resolve()
    print(f"Using YARP network module: {imported}")
    if expected_root is not None:
        root = Path(expected_root).expanduser().resolve()
        try:
            imported.relative_to(root)
        except Exception as exc:
            raise RuntimeError(
                f"YARP import mismatch: expected under {root}, got {imported}. "
                "Check yarp_path and your conda environment."
            ) from exc


def iter_reaction_objects(container, verbose=False):
    """Yield reaction-like objects from nested containers."""
    seen = set()
    stack = [container]
    while stack:
        current = stack.pop()
        obj_id = id(current)
        if obj_id in seen:
            continue
        seen.add(obj_id)

        if hasattr(current, "reverse_barrier") and (
            hasattr(current, "forward_barrier") or hasattr(current, "barrier")
        ):
            if verbose:
                print(f"[verbose] Found reaction-like object id={obj_id}")
            yield current
            continue

        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, (list, tuple, set)):
            stack.extend(current)


def reverse_barrier_is_populated(value):
    """Return True when a reverse barrier value is populated."""
    if value is None:
        return False
    if isinstance(value, dict):
        if not value:
            return False
        for v in value.values():
            if v is not None:
                return True
        return False
    return True


def has_reverse_barriers(payload, verbose=False):
    """Check if all reactions in the payload have reverse barriers."""
    seen_any = False
    for rxn in iter_reaction_objects(payload, verbose=verbose):
        seen_any = True
        reverse_barrier = getattr(rxn, "reverse_barrier", None)
        if not reverse_barrier_is_populated(reverse_barrier):
            if verbose:
                rid = getattr(rxn, "id", None)
                rhash = getattr(rxn, "hash", None)
                print(
                    f"[verbose] Missing/empty reverse_barrier for rxn id={rid} hash={rhash} "
                    f"value={reverse_barrier}"
                )
            return False
    if verbose:
        print(f"[verbose] Reverse barrier check complete, found_reactions={seen_any}")
    return seen_any


def ensure_dummy_barriers(source_pickle, cfg, verbose=False):
    """Add dummy reverse barriers when required by config."""
    if not bool(cfg.get("dummy_barriers_required", True)):
        return source_pickle, False

    yarp_path = cfg.get("yarp_path")
    if yarp_path:
        yarp_str = str(Path(yarp_path).expanduser())
        if yarp_str not in sys.path:
            sys.path.insert(0, yarp_str)
    if verbose:
        print(f"[verbose] yarp_path={yarp_path}")
        print(f"[verbose] source_pickle={source_pickle}")

    from add_dummy_reverse_barriers import add_dummy_reverse_barriers, load_pickle_payload

    payload = load_pickle_payload(source_pickle)
    if has_reverse_barriers(payload, verbose=verbose):
        if verbose:
            print("[verbose] Source already has reverse barriers.")
        return source_pickle, False

    overwrite = bool(cfg.get("overwrite_source_with_dummy_barriers", True))
    output_path = source_pickle if overwrite else source_pickle.with_name(f"{source_pickle.stem}_with_dummy{source_pickle.suffix}")
    seed = ((cfg.get("subnetwork_gen", {}) or {}).get("dummy_reverse_barriers", {}) or {}).get("seed", 42)
    if verbose:
        print(f"[verbose] Adding dummy barriers overwrite={overwrite} seed={seed} output_path={output_path}")
    _, written, updated_count, skipped_count = add_dummy_reverse_barriers(payload, output_path=output_path, seed=seed, verbose=verbose)
    if verbose:
        print(f"[verbose] Dummy barrier result written={written} updated={updated_count} skipped={skipped_count}")
    return Path(written), True


def write_runtime_config(base_cfg, source_pickle, product_hash, work_dir, verbose=False):
    """Write a per-product runtime config in a dedicated temp directory."""
    cfg = copy.deepcopy(base_cfg)

    sg = (cfg.get("subnetwork_gen", {}) or {})
    sg_input = (sg.get("input", {}) or {})
    sg_input["pickle"] = str(source_pickle)
    sg_input["directory"] = None
    sg_input["recursive"] = False
    sg["input"] = sg_input
    sg_dummy = (sg.get("dummy_reverse_barriers", {}) or {})
    sg_dummy["enabled"] = False
    sg["dummy_reverse_barriers"] = sg_dummy
    sg_sub = (sg.get("subnetworks", {}) or {})
    sg_sub["output_root"] = str(work_dir / "subnetworks")
    sg_sub["only_end_hash"] = str(product_hash)
    sg["subnetworks"] = sg_sub
    cfg["subnetwork_gen"] = sg

    cfs = (cfg.get("cantera_from_subnetworks", {}) or {})
    cfs_in = (cfs.get("input", {}) or {})
    cfs_in["root"] = str(work_dir / "subnetworks")
    cfs_in["recursive"] = True
    cfs["input"] = cfs_in
    cfs_out = (cfs.get("output", {}) or {})
    cfs_out["root"] = str(work_dir / "subnetwork_cantera_yaml")
    cfs_out["preserve_tree"] = True
    cfs["output"] = cfs_out
    cfg["cantera_from_subnetworks"] = cfs

    crs = (cfg.get("cantera_run_subnetworks", {}) or {})
    crs_in = (crs.get("input", {}) or {})
    crs_in["cantera_yaml_root"] = str(work_dir / "subnetwork_cantera_yaml")
    crs_in["subnetwork_pickle_root"] = str(work_dir / "subnetworks")
    crs_in["recursive"] = True
    crs["input"] = crs_in
    out_cfg = (crs.get("output", {}) or {})
    out_cfg.setdefault("write_run_yaml", True)
    out_cfg.setdefault("write_reaction_flux_table", False)
    out_cfg.setdefault("write_final_product_flux_table", True)
    out_cfg["write_flux_timeseries_table"] = bool(base_cfg.get("write_flux_timeseries_table", False))
    out_cfg.setdefault("write_reactor_debug_yaml", False)
    out_cfg.setdefault("update_subnetwork_pickle_metadata", True)
    crs["output"] = out_cfg
    cfg["cantera_run_subnetworks"] = crs

    runtime_cfg = work_dir / "pipeline_runtime.yaml"
    with runtime_cfg.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    if verbose:
        print(f"[verbose] Runtime config written: {runtime_cfg}")
        print(f"[verbose] Runtime source_pickle={source_pickle}")
        print(f"[verbose] Runtime product_hash={product_hash}")
        print(f"[verbose] Runtime subnetwork_output={sg_sub.get('output_root')}")
        print(f"[verbose] Runtime cantera_yaml_root={cfs_out.get('root')}")
        print(f"[verbose] Runtime cantera_run_yaml_root={crs_in.get('cantera_yaml_root')}")
    return runtime_cfg


def first_match(root, pattern, verbose=False):
    """Return the first recursive match for a file glob pattern."""
    hits = sorted(root.rglob(pattern))
    if not hits:
        raise RuntimeError(f"No files matched {pattern} under {root}")
    if verbose:
        print(f"[verbose] first_match pattern={pattern} root={root} hit={hits[0]}")
    return hits[0]


def first_cantera_input_yaml(root, verbose=False):
    """Return the first cantera model YAML, excluding sidecar metadata YAMLs."""
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
        if verbose:
            print(f"[verbose] Selected cantera input yaml: {path}")
        return path
    raise RuntimeError(f"No cantera input yaml found under {root}")


def run_one_product(
    base_cfg,
    env,
    source_pickle,
    network_out_dir,
    product_hash,
    product_smiles,
    flux_output_path=None,
    verbose=False,
):
    """Run subnetwork->Cantera->datatable workflow for one terminal product."""
    product_label = safe_label(product_hash or product_smiles or "product", verbose=verbose)
    work_dir = network_out_dir / ".tmp" / product_label
    work_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[verbose] Product work_dir={work_dir}")
        print(
            f"[verbose] Product hash={product_hash} smiles={product_smiles} "
            f"flux_output={flux_output_path}"
        )
    runtime_cfg = write_runtime_config(base_cfg, source_pickle, product_hash, work_dir, verbose=verbose)

    run_steps = list(base_cfg.get("_resolved_product_run_steps", PRODUCT_RUN_STEPS))
    for step in run_steps:
        if step == "subnetwork":
            run_cmd([sys.executable, str(SCRIPT_DIR / "subnetwork_gen.py"), "--config", str(runtime_cfg)], env, verbose=verbose)
        elif step == "cantera_yaml":
            run_cmd([sys.executable, str(SCRIPT_DIR / "cantera_from_subnetworks.py"), "--config", str(runtime_cfg)], env, verbose=verbose)
        elif step == "cantera_run":
            run_cmd([sys.executable, str(SCRIPT_DIR / "cantera_run_subnetworks.py"), "--config", str(runtime_cfg)], env, verbose=verbose)

    subnetwork_pickle = first_match(work_dir / "subnetworks", "*.pkl", verbose=verbose)
    cantera_yaml = first_cantera_input_yaml(work_dir / "subnetwork_cantera_yaml", verbose=verbose)
    to_final_csv = first_match(work_dir / "subnetwork_cantera_yaml", "*.to_final.csv", verbose=verbose)
    flux_ts_csv = None
    try:
        flux_ts_csv = first_match(work_dir / "subnetwork_cantera_yaml", "*.flux_timeseries.csv", verbose=verbose)
    except RuntimeError:
        flux_ts_csv = None

    table_out = network_out_dir / f"product_{product_label}.parquet"
    datatable_cmd = [
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
        "--network-pickle",
        str(source_pickle),
        "--output",
        str(table_out),
    ]
    if flux_ts_csv is not None:
        datatable_cmd.extend(["--flux-timeseries-csv", str(flux_ts_csv)])
    if flux_output_path is not None:
        datatable_cmd.extend(["--flux-output", str(flux_output_path)])
    run_cmd(datatable_cmd, env, verbose=verbose)
    if verbose:
        print(f"[verbose] Product outputs table={table_out}")
    return work_dir, table_out


def cleanup_workdir(path, verbose=False):
    """Remove a per-product temp working directory."""
    if path.exists():
        if verbose:
            print(f"[verbose] Cleaning temp directory: {path}")
        shutil.rmtree(path, ignore_errors=True)


def clear_previous_outputs(network_out_dir, verbose=False):
    """Delete prior network output tables and plot directory."""
    patterns = [
        "product_*.parquet",
        "product_*.pkl",
        "flux_timeseries*.parquet",
        "flux_timeseries*.pkl",
        "network_all_rows_merged.parquet",
        "network_all_rows_merged.pkl",
        "terminal_products.parquet",
        "terminal_products.pkl",
    ]
    removed = 0
    for pattern in patterns:
        for path in network_out_dir.glob(pattern):
            if path.is_file():
                path.unlink()
                removed += 1
                if verbose:
                    print(f"[verbose] Removed previous output: {path}")
    plot_dir = network_out_dir / "bulk_profile_plots"
    if plot_dir.exists() and plot_dir.is_dir():
        shutil.rmtree(plot_dir, ignore_errors=True)
        removed += 1
        if verbose:
            print(f"[verbose] Removed previous output directory: {plot_dir}")
    if removed:
        print(f"Cleared {removed} previous output file(s) from {network_out_dir}")


def read_parquet_with_arrow_retry(path):
    """Handle duplicate Arrow extension registrations in reused notebook kernels."""
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


def load_table_fallback(path, verbose=False):
    """Load a table from parquet, then pickle fallback."""
    path = Path(path)
    if path.exists():
        if verbose:
            print(f"[verbose] loading parquet table: {path}")
        return read_parquet_with_arrow_retry(path), path
    pkl = path.with_suffix(".pkl")
    if pkl.exists():
        if verbose:
            print(f"[verbose] parquet missing, loading pickle table: {pkl}")
        return pd.read_pickle(pkl), pkl
    raise FileNotFoundError(f"Could not find table file: {path} or {pkl}")


def load_table_any(path):
    """Load a table file by suffix (.parquet or .pkl)."""
    path = Path(path)
    if path.suffix == ".parquet":
        return read_parquet_with_arrow_retry(path)
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported table suffix for merge: {path}")


def write_table_with_fallback(df, out_path):
    """Write parquet with pickle fallback on serialization errors."""
    out_path = Path(out_path)
    try:
        df.to_parquet(out_path, index=False)
        return out_path
    except Exception as exc:
        fallback = out_path.with_suffix(".pkl")
        df.to_pickle(fallback)
        print(
            f"[warning] parquet write failed ({exc.__class__.__name__}); "
            f"wrote pickle instead: {fallback}"
        )
        return fallback


def preferred_table_files(network_out_dir, parquet_pattern, pkl_pattern):
    """Choose preferred table files (parquet first, then pickle) by stem."""
    parquet_files = {p.stem: p for p in sorted(network_out_dir.glob(parquet_pattern))}
    pkl_files = {p.stem: p for p in sorted(network_out_dir.glob(pkl_pattern))}
    stems = sorted(set(parquet_files.keys()) | set(pkl_files.keys()))
    selected = []
    for stem in stems:
        if stem in parquet_files:
            selected.append(parquet_files[stem])
        elif stem in pkl_files:
            selected.append(pkl_files[stem])
    return selected


def merge_all_network_tables(network_out_dir, out_name="network_all_rows_merged.parquet", verbose=False):
    """Merge product summary tables into one network-level all-rows table."""
    sources = [
        ("product_summary", preferred_table_files(network_out_dir, "product_*.parquet", "product_*.pkl")),
    ]
    frames = []
    for table_origin, paths in sources:
        for table_path in paths:
            try:
                df = load_table_any(table_path)
            except Exception as exc:
                print(f"[warning] skipped merge input {table_path}: {exc}")
                continue
            if df.empty:
                continue
            work = df.copy()
            work["table_origin"] = table_origin
            work["source_table_file"] = table_path.name
            frames.append(work.dropna(axis=1, how="all"))
    if not frames:
        if verbose:
            print("[verbose] no non-empty product tables found for full merge")
        return None
    merged = pd.concat(frames, ignore_index=True, sort=False)
    out_path = network_out_dir / out_name
    written = write_table_with_fallback(merged, out_path)
    print(f"Merged full network table ({len(frames)} source tables) -> {written}")
    return written


def remove_network_source_tables_after_merge(network_out_dir, verbose=False):
    """Remove source tables after writing the merged all-rows table."""
    patterns = [
        "product_*.parquet",
        "product_*.pkl",
    ]
    removed = 0
    for pattern in patterns:
        for path in network_out_dir.glob(pattern):
            if path.is_file():
                path.unlink()
                removed += 1
                if verbose:
                    print(f"[verbose] Removed source table after merge: {path}")
    if removed:
        print(f"Removed {removed} source product table(s) after full merge.")


def detect_available_cores():
    """Detect available CPU cores from affinity, then os.cpu_count fallback."""
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except Exception:
            pass
    return max(1, int(os.cpu_count() or 1))


def resolve_parallel_workers(cfg, total_jobs, cli_workers=None, verbose=False):
    """Resolve worker count from CLI, config, scheduler env var, and CPU count."""
    if total_jobs <= 1:
        return 1

    if cli_workers is not None:
        workers = max(1, int(cli_workers))
        return min(workers, total_jobs)

    configured = cfg.get("parallel_subnetwork_workers", "auto")
    env_var = str(cfg.get("parallel_worker_env_var", "NSLOTS"))
    env_raw = os.environ.get(env_var)
    env_workers = None
    if env_raw:
        try:
            env_workers = max(1, int(env_raw))
        except Exception:
            env_workers = None

    if isinstance(configured, str) and configured.strip().lower() in {"auto", ""}:
        workers = env_workers if env_workers is not None else detect_available_cores()
    else:
        workers = max(1, int(configured))
        if env_workers is not None:
            workers = min(workers, env_workers)

    workers = min(workers, total_jobs)
    if verbose:
        print(
            "[verbose] parallel worker resolution:",
            f"configured={configured}",
            f"env_var={env_var}",
            f"env_workers={env_workers}",
            f"detected_cores={detect_available_cores()}",
            f"resolved={workers}",
        )
    return max(1, workers)


def run_product_job(
    cfg,
    env,
    source_pickle,
    network_out_dir,
    job,
    verbose=False,
):
    """Run one product job dict and return its output metadata."""
    work_dir, table_out = run_one_product(
        cfg,
        env,
        source_pickle,
        network_out_dir,
        job["product_hash"],
        job["product_smiles"],
        job["flux_output_path"],
        verbose=verbose,
    )
    return {"work_dir": work_dir, "table_out": table_out, "job": job}


def apply_single_core_subprocess_env(cfg, env, verbose=False):
    """Force spawned subprocesses to one thread unless explicitly disabled."""
    if not bool(cfg.get("subprocess_single_core", True)):
        return env
    thread_vars = cfg.get(
        "subprocess_thread_env_vars",
        [
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        ],
    )
    out = dict(env)
    for var_name in list(thread_vars):
        key = str(var_name).strip()
        if key:
            out[key] = "1"
    if verbose:
        print(f"[verbose] single-core subprocess env applied: {', '.join([str(v) for v in thread_vars])}")
    return out


def write_resolved_config_snapshot(cfg, network_out_dir, verbose=False):
    """Write one resolved config snapshot per network for provenance."""
    snapshot = copy.deepcopy(cfg)
    for key in list(snapshot.keys()):
        if str(key).startswith("_"):
            snapshot.pop(key, None)
    out_path = network_out_dir / "pipeline_runtime_resolved.yaml"
    with out_path.open("w") as f:
        yaml.safe_dump(snapshot, f, sort_keys=False)
    if verbose:
        print(f"[verbose] Wrote resolved config snapshot: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run one-network streaming subnetwork pipeline.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--task-id", type=int, default=None, help="1-based manifest index override for local runs.")
    parser.add_argument(
        "--network-pickle",
        default=None,
        help="Optional direct network pickle path override (bypasses manifest/task selection).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Optional override for per-network subnetwork worker count.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else default_config_path()
    cfg, cfg_dir = load_config(cfg_path)
    mode = apply_output_mode_defaults(cfg, verbose=bool(cfg.get("verbose", False)))
    verbose = bool(cfg.get("verbose", False))
    if verbose:
        print(f"[verbose] Config loaded from {cfg_path}")
        print(f"[verbose] Config dir: {cfg_dir}")
    print(f"Output mode: {mode} ({OUTPUT_MODE_PRESETS[mode]['description']})")
    cfg["_resolved_product_run_steps"] = resolve_product_run_steps(cfg)
    if verbose:
        print(f"[verbose] product run_steps={cfg['_resolved_product_run_steps']}")

    yarp_path = cfg.get("yarp_path")
    env = os.environ.copy()
    if yarp_path:
        yarp_root = Path(yarp_path).expanduser().resolve()
        if not yarp_root.exists():
            raise FileNotFoundError(f"Configured yarp_path does not exist: {yarp_root}")
        yarp_resolved = str(yarp_root)
        env["PYTHONPATH"] = yarp_resolved if not env.get("PYTHONPATH") else yarp_resolved + os.pathsep + env["PYTHONPATH"]
        verify_yarp_import(env, expected_root=yarp_root)
    else:
        verify_yarp_import(env, expected_root=None)
    env = apply_single_core_subprocess_env(cfg, env, verbose=verbose)

    if args.network_pickle:
        source_pickle = Path(args.network_pickle).expanduser().resolve()
        task_id = 1
        manifest_rows = [str(source_pickle)]
    else:
        manifest_path = resolve_path(str(cfg.get("network_manifest")), cfg_dir)
        manifest_rows = load_manifest(manifest_path, verbose=verbose)
        array_env = str(cfg.get("array_env_var", "SGE_TASK_ID"))
        task_id = int(args.task_id) if args.task_id is not None else task_index_from_env(array_env, verbose=verbose)
        if task_id < 1:
            raise ValueError(f"--task-id must be >= 1, got {task_id}")
        if task_id > len(manifest_rows):
            print(f"{array_env}={task_id} exceeds manifest length ({len(manifest_rows)}); nothing to do.")
            return
        source_pickle = Path(manifest_rows[task_id - 1]).expanduser().resolve()
    if not source_pickle.exists():
        raise FileNotFoundError(f"Network pickle not found: {source_pickle}")

    output_dir = resolve_path(str(cfg.get("output_dir")), cfg_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    network_out_dir = output_dir / f"networks__{safe_label(source_pickle.stem, verbose=verbose)}"
    network_out_dir.mkdir(parents=True, exist_ok=True)
    if bool(cfg.get("clear_previous_outputs", False)):
        clear_previous_outputs(network_out_dir, verbose=verbose)
    write_resolved_config_snapshot(cfg, network_out_dir, verbose=verbose)

    print(f"Task {task_id}/{len(manifest_rows)} | network: {source_pickle}")
    source_pickle, updated = ensure_dummy_barriers(source_pickle, cfg, verbose=verbose)
    if updated:
        print(f"Dummy barriers added in-place: {source_pickle}")
    else:
        print("Dummy barriers already present; skipping rewrite.")

    terminal_path = network_out_dir / "terminal_products.parquet"
    run_cmd(
        [
            sys.executable,
            str(SCRIPT_DIR / "terminal_products.py"),
            "--config",
            str(cfg_path),
            "--network-pickle",
            str(source_pickle),
            "--output",
            str(terminal_path),
        ],
        env,
        verbose=verbose,
    )

    terminals, terminal_loaded_path = load_table_fallback(terminal_path, verbose=verbose)
    if verbose:
        print(f"[verbose] Terminal table loaded: {terminal_loaded_path}")
        print(f"[verbose] Terminal columns: {list(terminals.columns)}")
    if terminals.empty:
        print("No terminal products found.")
        return
    retain_terminal_table = bool(cfg.get("save_terminal_list", False))
    if not retain_terminal_table:
        for p in [terminal_path, terminal_path.with_suffix(".pkl")]:
            if p.exists() and p.is_file():
                p.unlink()
                if verbose:
                    print(f"[verbose] Removed terminal product table due to output mode: {p}")

    only_product_hash = cfg.get("only_product_hash")
    max_products = cfg.get("max_products_per_network")
    print(
        "Product filters:",
        f"only_product_hash={only_product_hash}",
        f"max_products_per_network={max_products}",
    )
    if only_product_hash is not None:
        terminals = terminals[terminals["product_hash"].astype(str) == str(only_product_hash)].copy()
    if max_products is not None:
        limit = max(1, int(max_products))
        terminals = terminals.head(limit).copy()
        print(f"Applied product cap: {limit}")
    else:
        print("No product cap applied; processing all matching terminal products.")
    if terminals.empty:
        print("No terminal products matched the selected product filters.")
        return
    print(f"Selected terminal products for processing: {len(terminals)}")

    save_flux_tables = bool(cfg.get("save_flux_for_every_subnetwork", False))
    if save_flux_tables:
        print(f"Flux profile table retention enabled for all {len(terminals)} product(s).")
    else:
        print("Flux profile table retention disabled.")
    cleanup_mode = str(cfg.get("cleanup_mode", "clean"))
    failure_mode = str(cfg.get("failure_mode", "fail_product_continue_network"))
    keep_failed = cleanup_mode == "debug_keep_failed"
    jobs = []
    for i, row in terminals.reset_index(drop=True).iterrows():
        product_hash = str(row.get("product_hash", ""))
        product_smiles = row.get("product_smiles")
        product_label = safe_label(product_hash or product_smiles or "product", verbose=verbose)
        flux_output_path = None
        if save_flux_tables:
            flux_output_path = network_out_dir / f"flux_timeseries__{product_label}.parquet"
        jobs.append(
            {
                "index": i,
                "total": len(terminals),
                "product_hash": product_hash,
                "product_smiles": product_smiles,
                "product_label": product_label,
                "flux_output_path": flux_output_path,
                "work_dir": network_out_dir / ".tmp" / product_label,
            }
        )

    parallel_enabled = bool(cfg.get("parallelize_subnetworks", True))
    workers = resolve_parallel_workers(cfg, total_jobs=len(jobs), cli_workers=args.max_workers, verbose=verbose)
    if not parallel_enabled:
        workers = 1
    print(f"Subnetwork worker mode: parallelize_subnetworks={parallel_enabled} workers={workers}")

    if workers <= 1:
        for job in jobs:
            print(
                f"[{job['index']+1}/{job['total']}] "
                f"product_hash={job['product_hash']} smiles={job['product_smiles']}"
            )
            work_dir = None
            try:
                work_dir, table_out = run_one_product(
                    cfg,
                    env,
                    source_pickle,
                    network_out_dir,
                    job["product_hash"],
                    job["product_smiles"],
                    job["flux_output_path"],
                    verbose=verbose,
                )
                print(f"Wrote: {table_out}")
                if cleanup_mode in {"clean", "debug_keep_failed"} and work_dir is not None:
                    cleanup_workdir(work_dir, verbose=verbose)
            except Exception as exc:
                print(f"FAILED product {job['product_hash']}: {exc}")
                if work_dir is not None and not keep_failed:
                    cleanup_workdir(work_dir, verbose=verbose)
                if failure_mode != "fail_product_continue_network":
                    raise
    else:
        pending = []
        first_failure = None
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for job in jobs:
                print(
                    f"[queued {job['index']+1}/{job['total']}] "
                    f"product_hash={job['product_hash']} smiles={job['product_smiles']}"
                )
                future = executor.submit(
                    run_product_job,
                    cfg,
                    env,
                    source_pickle,
                    network_out_dir,
                    job,
                    verbose,
                )
                pending.append((future, job))

            future_map = {f: j for f, j in pending}
            while future_map:
                done, _ = wait(list(future_map.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    job = future_map.pop(future)
                    try:
                        result = future.result()
                        table_out = result["table_out"]
                        work_dir = result["work_dir"]
                        print(
                            f"[done {job['index']+1}/{job['total']}] "
                            f"product_hash={job['product_hash']} -> {table_out}"
                        )
                        if cleanup_mode in {"clean", "debug_keep_failed"} and work_dir is not None:
                            cleanup_workdir(work_dir, verbose=verbose)
                    except Exception as exc:
                        print(f"FAILED product {job['product_hash']}: {exc}")
                        if not keep_failed and job.get("work_dir") is not None:
                            cleanup_workdir(job["work_dir"], verbose=verbose)
                        if failure_mode != "fail_product_continue_network" and first_failure is None:
                            first_failure = exc
                            for other_future in list(future_map.keys()):
                                other_future.cancel()
                            future_map.clear()
                            break

        if first_failure is not None:
            raise first_failure

    tmp_root = network_out_dir / ".tmp"
    if tmp_root.exists() and not any(tmp_root.iterdir()):
        tmp_root.rmdir()

    if bool(cfg.get("plot_all_network_flux_outputs", False)):
        run_cmd(
            [
                sys.executable,
                str(SCRIPT_DIR / "generate_bulk_plots.py"),
                "--config",
                str(cfg_path),
                "--network-out-dir",
                str(network_out_dir),
            ],
            env,
            verbose=verbose,
        )

    if bool(cfg.get("merge_all_tables_per_network", False)):
        out_name = str(cfg.get("merged_all_table_name", "network_all_rows_merged.parquet"))
        merged_path = merge_all_network_tables(network_out_dir, out_name=out_name, verbose=verbose)
        if merged_path is not None and bool(cfg.get("remove_source_tables_after_full_merge", False)):
            remove_network_source_tables_after_merge(network_out_dir, verbose=verbose)


if __name__ == "__main__":
    main()
