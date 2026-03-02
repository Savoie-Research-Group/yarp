#!/usr/bin/env python3
"""Run one-network streaming subnetwork kinetics pipeline."""

from __future__ import annotations

import argparse
import copy
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def default_config_path(verbose=False):
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
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be a top-level mapping.")
    if verbose:
        print(f"[verbose] Loaded config mapping from {config_path}")
    return cfg, config_path.parent.resolve()


def load_manifest(path, verbose=False):
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
    value = "".join(ch if (ch.isalnum() or ch in "-._") else "_" for ch in str(text))
    if verbose:
        print(f"[verbose] safe_label input={text} output={value}")
    return value


def run_cmd(cmd, env, verbose=False):
    print("Running:", " ".join(cmd))
    if verbose:
        print(f"[verbose] cwd={REPO_ROOT}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT), env=env)


def verify_yarp_import(env, expected_root=None):
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


def reverse_barrier_is_populated(value, verbose=False):
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
    seen_any = False
    for rxn in iter_reaction_objects(payload, verbose=verbose):
        seen_any = True
        reverse_barrier = getattr(rxn, "reverse_barrier", None)
        if not reverse_barrier_is_populated(reverse_barrier, verbose=verbose):
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
    out_cfg.setdefault("write_flux_timeseries_table", True)
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
    hits = sorted(root.rglob(pattern))
    if not hits:
        raise RuntimeError(f"No files matched {pattern} under {root}")
    if verbose:
        print(f"[verbose] first_match pattern={pattern} root={root} hit={hits[0]}")
    return hits[0]


def first_cantera_input_yaml(root, verbose=False):
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
    random_flux_output_path=None,
    verbose=False,
):
    product_label = safe_label(product_hash or product_smiles or "product", verbose=verbose)
    work_dir = network_out_dir / ".tmp" / product_label
    work_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"[verbose] Product work_dir={work_dir}")
        print(
            f"[verbose] Product hash={product_hash} smiles={product_smiles} "
            f"random_flux_output={random_flux_output_path}"
        )
    runtime_cfg = write_runtime_config(base_cfg, source_pickle, product_hash, work_dir, verbose=verbose)

    run_steps = base_cfg.get("run_steps", []) or []
    for step in run_steps:
        if step == "subnetwork":
            run_cmd([sys.executable, str(SCRIPT_DIR / "subnetwork_gen.py"), "--config", str(runtime_cfg)], env, verbose=verbose)
        elif step == "cantera_yaml":
            run_cmd([sys.executable, str(SCRIPT_DIR / "cantera_from_subnetworks.py"), "--config", str(runtime_cfg)], env, verbose=verbose)
        elif step == "cantera_run":
            run_cmd([sys.executable, str(SCRIPT_DIR / "cantera_run_subnetworks.py"), "--config", str(runtime_cfg)], env, verbose=verbose)
        elif step in {"dummy_barriers", "terminal_products", "datatable"}:
            continue

    subnetwork_pickle = first_match(work_dir / "subnetworks", "*.pkl", verbose=verbose)
    cantera_yaml = first_cantera_input_yaml(work_dir / "subnetwork_cantera_yaml", verbose=verbose)
    to_final_csv = first_match(work_dir / "subnetwork_cantera_yaml", "*.to_final.csv", verbose=verbose)
    flux_ts_csv = first_match(work_dir / "subnetwork_cantera_yaml", "*.flux_timeseries.csv", verbose=verbose)

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
        "--flux-timeseries-csv",
        str(flux_ts_csv),
        "--network-pickle",
        str(source_pickle),
        "--output",
        str(table_out),
    ]
    if random_flux_output_path is not None:
        datatable_cmd.extend(["--random-flux-output", str(random_flux_output_path)])
    run_cmd(datatable_cmd, env, verbose=verbose)
    if verbose:
        print(f"[verbose] Product outputs table={table_out}")
    return work_dir, table_out


def cleanup_workdir(path, verbose=False):
    if path.exists():
        if verbose:
            print(f"[verbose] Cleaning temp directory: {path}")
        shutil.rmtree(path, ignore_errors=True)


def clear_previous_outputs(network_out_dir, verbose=False):
    patterns = [
        "product_*.parquet",
        "product_*.pkl",
        "random_flux_timeseries*.parquet",
        "random_flux_timeseries*.pkl",
        "network_products_merged.parquet",
        "network_products_merged.pkl",
        "network_random_flux_merged.parquet",
        "network_random_flux_merged.pkl",
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
    path = Path(path)
    if path.suffix == ".parquet":
        return read_parquet_with_arrow_retry(path)
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported table suffix for merge: {path}")


def write_table_with_fallback(df, out_path):
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


def merge_network_tables(network_out_dir, *, parquet_pattern, pkl_pattern, out_name, verbose=False):
    table_files = preferred_table_files(network_out_dir, parquet_pattern, pkl_pattern)
    if not table_files:
        if verbose:
            print(f"[verbose] no files matched merge pattern {parquet_pattern} / {pkl_pattern}")
        return None
    frames = []
    for table_path in table_files:
        try:
            df = load_table_any(table_path)
            if not df.empty:
                frames.append(df)
        except Exception as exc:
            print(f"[warning] skipped merge input {table_path}: {exc}")
    if not frames:
        if verbose:
            print(f"[verbose] merge inputs were empty for {out_name}")
        return None
    frames = [df.dropna(axis=1, how="all") for df in frames]
    merged = pd.concat(frames, ignore_index=True, sort=False)
    out_path = network_out_dir / out_name
    written = write_table_with_fallback(merged, out_path)
    print(f"Merged {len(frames)} table(s) -> {written}")
    return written


def merge_all_network_tables(network_out_dir, out_name="network_all_rows_merged.parquet", verbose=False):
    sources = [
        ("product_summary", preferred_table_files(network_out_dir, "product_*.parquet", "product_*.pkl")),
        ("random_timeseries", preferred_table_files(network_out_dir, "random_flux_timeseries*.parquet", "random_flux_timeseries*.pkl")),
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
            print("[verbose] no non-empty product/random tables found for full merge")
        return None
    merged = pd.concat(frames, ignore_index=True, sort=False)
    out_path = network_out_dir / out_name
    written = write_table_with_fallback(merged, out_path)
    print(f"Merged full network table ({len(frames)} source tables) -> {written}")
    return written


def remove_network_source_tables_after_merge(network_out_dir, verbose=False):
    patterns = [
        "product_*.parquet",
        "product_*.pkl",
        "random_flux_timeseries*.parquet",
        "random_flux_timeseries*.pkl",
        "terminal_products.parquet",
        "terminal_products.pkl",
        "network_products_merged.parquet",
        "network_products_merged.pkl",
        "network_random_flux_merged.parquet",
        "network_random_flux_merged.pkl",
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
        print(f"Removed {removed} source product/random table(s) after full merge.")


def main():
    parser = argparse.ArgumentParser(description="Run one-network streaming subnetwork pipeline.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--task-id", type=int, default=None, help="1-based manifest index override for local runs.")
    parser.add_argument(
        "--network-pickle",
        default=None,
        help="Optional direct network pickle path override (bypasses manifest/task selection).",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else default_config_path()
    cfg, cfg_dir = load_config(cfg_path)
    verbose = bool(cfg.get("verbose", False))
    if verbose:
        print(f"[verbose] Config loaded from {cfg_path}")
        print(f"[verbose] Config dir: {cfg_dir}")

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

    rng = random.Random(int(cfg.get("random_flux_seed", 42)) + task_id)
    retain_count = max(0, int(cfg.get("random_flux_retain_count", 1)))
    if bool(cfg.get("save_random_flux_table", False)) and retain_count > 0:
        keep_count = min(retain_count, len(terminals))
        random_indices = sorted(rng.sample(range(len(terminals)), k=keep_count))
    else:
        random_indices = []
    random_index_set = set(random_indices)
    if random_indices:
        print(f"Retaining random flux tables for {len(random_indices)} product(s): indices={random_indices}")
    else:
        print("Random flux table retention disabled.")
    cleanup_mode = str(cfg.get("cleanup_mode", "clean"))
    failure_mode = str(cfg.get("failure_mode", "fail_product_continue_network"))

    for i, row in terminals.reset_index(drop=True).iterrows():
        product_hash = str(row.get("product_hash", ""))
        product_smiles = row.get("product_smiles")
        print(f"[{i+1}/{len(terminals)}] product_hash={product_hash} smiles={product_smiles}")
        keep_failed = cleanup_mode == "debug_keep_failed"
        random_flux_output_path = None
        if i in random_index_set:
            product_label = safe_label(product_hash or product_smiles or "product", verbose=verbose)
            random_flux_output_path = network_out_dir / f"random_flux_timeseries__{product_label}.parquet"
        work_dir = None
        try:
            work_dir, table_out = run_one_product(
                cfg,
                env,
                source_pickle,
                network_out_dir,
                product_hash,
                product_smiles,
                random_flux_output_path,
                verbose=verbose,
            )
            print(f"Wrote: {table_out}")
            if cleanup_mode in {"clean", "debug_keep_failed"} and work_dir is not None:
                cleanup_workdir(work_dir, verbose=verbose)
        except Exception as exc:
            print(f"FAILED product {product_hash}: {exc}")
            if work_dir is not None and not keep_failed:
                cleanup_workdir(work_dir, verbose=verbose)
            if failure_mode != "fail_product_continue_network":
                raise

    tmp_root = network_out_dir / ".tmp"
    if tmp_root.exists() and not any(tmp_root.iterdir()):
        tmp_root.rmdir()

    plot_cfg = cfg.get("plot_export", {}) or {}
    if bool(plot_cfg.get("enabled", False)):
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

    if bool(cfg.get("merge_product_tables_per_network", False)):
        merge_network_tables(
            network_out_dir,
            parquet_pattern="product_*.parquet",
            pkl_pattern="product_*.pkl",
            out_name="network_products_merged.parquet",
            verbose=verbose,
        )
    if bool(cfg.get("merge_random_flux_tables_per_network", False)):
        merge_network_tables(
            network_out_dir,
            parquet_pattern="random_flux_timeseries*.parquet",
            pkl_pattern="random_flux_timeseries*.pkl",
            out_name="network_random_flux_merged.parquet",
            verbose=verbose,
        )
    if bool(cfg.get("merge_all_tables_per_network", False)):
        out_name = str(cfg.get("merged_all_table_name", "network_all_rows_merged.parquet"))
        merged_path = merge_all_network_tables(network_out_dir, out_name=out_name, verbose=verbose)
        if merged_path is not None and bool(cfg.get("remove_source_tables_after_full_merge", False)):
            remove_network_source_tables_after_merge(network_out_dir, verbose=verbose)


if __name__ == "__main__":
    main()
