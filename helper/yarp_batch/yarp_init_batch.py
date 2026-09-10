#!/usr/bin/env python3
"""Create one or more SMILES batches and run yarp-init in each work directory."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from .yarp_batch_common import (
    batch_name,
    load_batch_config,
    load_yaml,
    next_batch_index,
    read_smiles_csv,
    render_yarp_config,
    resolve_config_path,
    source_identifier,
    write_yaml,
)


def run_yarp_init(work_dir: Path, config_name: str, log_name: str, yarp_init: str) -> int:
    with open(work_dir / log_name, "a") as log:
        log.write(f"\n=== yarp-init {config_name} in {work_dir} ===\n")
        result = subprocess.run(
            [yarp_init, config_name],
            cwd=work_dir,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log.write(f"=== exit code: {result.returncode} ===\n")
    return result.returncode


def write_manifest(batch_dir: Path, manifest_rows: list[dict[str, str]]) -> None:
    with open(batch_dir / "manifest.csv", "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["global_index", "smiles", "source_id", "reaction_output", "work_dir", "status"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)


def select_batches(
    rows: list[dict],
    output_dir: Path,
    batch_size: int,
    start_batch: int,
    requested_batch: int | None,
    all_batches: bool,
) -> list[tuple[int, list[dict]]]:
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    total_batches = (len(rows) + batch_size - 1) // batch_size
    if requested_batch is not None:
        indices = [requested_batch]
    elif all_batches:
        indices = list(range(start_batch, start_batch + total_batches))
    else:
        indices = [next_batch_index(output_dir, start_batch)]

    selected = []
    for batch_index in indices:
        zero_based = batch_index - start_batch
        if zero_based < 0:
            raise ValueError(f"Batch {batch_index} is before start_batch {start_batch}")
        start = zero_based * batch_size
        stop = start + batch_size
        batch_rows = rows[start:stop]
        if batch_rows:
            selected.append((batch_index, batch_rows))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="High-throughput yarp-init over a SMILES CSV.")
    parser.add_argument(
        "batch_config",
        nargs="?",
        type=Path,
        default=Path("batch.yaml"),
        help="Batch YAML with smiles_csv, template_config, output_dir, and batch_size.",
    )
    parser.add_argument("--smiles-csv", type=Path, help="Override SMILES CSV path.")
    parser.add_argument("--template-config", type=Path, help="Override YARP template YAML path.")
    parser.add_argument("--output-dir", type=Path, help="Override batch output directory.")
    parser.add_argument("--batch-size", type=int, help="Override number of SMILES per batch.")
    parser.add_argument("--batch-index", type=int, help="Create this exact batch index.")
    parser.add_argument("--all-batches", action="store_true", help="Create every batch from the CSV.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without writing or running.")
    parser.add_argument("--no-init", action="store_true", help="Write directories/configs but skip yarp-init.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing workdir input files.")
    args = parser.parse_args()

    config, config_dir = load_batch_config(args.batch_config)
    smiles_csv = resolve_config_path(args.smiles_csv or config["smiles_csv"], config_dir)
    template_path = resolve_config_path(args.template_config or config["template_config"], config_dir)
    output_dir = resolve_config_path(args.output_dir or config["output_dir"], config_dir)
    batch_size = args.batch_size or int(config["batch_size"])
    start_batch = int(config["start_batch"])
    config_name = config["config_name"]
    status_name = config["status_name"]
    reaction_output_name = config["reaction_output_name"]
    init_log_name = config["init_log_name"]
    yarp_init = config["yarp_init"]

    smiles_rows = read_smiles_csv(smiles_csv)
    template = load_yaml(template_path)
    batches = select_batches(
        smiles_rows,
        output_dir,
        batch_size,
        start_batch,
        args.batch_index,
        args.all_batches,
    )

    if not batches:
        print("No SMILES rows selected for initialization.")
        return 0

    print(f"Loaded {len(smiles_rows)} SMILES from {smiles_csv}")
    print(f"Planning {len(batches)} batch(es) in {output_dir}")

    failures = 0
    for batch_index, batch_rows in batches:
        current_batch_dir = output_dir / batch_name(batch_index)
        manifest_rows = []

        for row in batch_rows:
            smiles = row["smiles"]
            source_id = source_identifier(smiles)
            output_name = f"{source_id}.pkl" if reaction_output_name == "auto" else reaction_output_name
            work_dir = current_batch_dir / f"{row['index']:06d}_{source_id}"
            input_path = work_dir / config_name
            status_path = work_dir / status_name

            manifest_rows.append(
                {
                    "global_index": str(row["index"]),
                    "smiles": smiles,
                    "source_id": source_id,
                    "reaction_output": output_name,
                    "work_dir": str(work_dir),
                    "status": "planned",
                }
            )

            if args.dry_run:
                print(f"[dry-run] {work_dir}: {smiles}")
                continue

            if input_path.exists() and not args.overwrite:
                print(f"[skip] {input_path} exists")
                continue

            work_dir.mkdir(parents=True, exist_ok=True)
            (work_dir / "smiles.txt").write_text(smiles + "\n")
            rendered = render_yarp_config(template, smiles, output_name, status_name)
            write_yaml(input_path, rendered)

            if args.no_init:
                print(f"[write] {work_dir}")
                continue

            if status_path.exists() and not args.overwrite:
                print(f"[skip] {status_path} exists")
                continue

            exit_code = run_yarp_init(work_dir, config_name, init_log_name, yarp_init)
            if exit_code == 0:
                print(f"[ok] {work_dir}")
            else:
                failures += 1
                print(f"[fail:{exit_code}] {work_dir}; see {work_dir / init_log_name}")

        if not args.dry_run:
            current_batch_dir.mkdir(parents=True, exist_ok=True)
            write_manifest(current_batch_dir, manifest_rows)

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
