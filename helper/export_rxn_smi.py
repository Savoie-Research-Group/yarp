#!/usr/bin/env python3
"""
Export reaction SMILES and optional metadata columns from a YARP pickle.

Default output:
    rxn_id, reactant_canon_smi, product_canon_smi

Use `-m/--mapped` to include atom-mapped SMILES columns.

Options:
    -i, --ids                Include rxn_hash.
    -c, --canon              Include canonical SMILES. This is on by default.
    -m, --mapped             Include atom-mapped SMILES.
    -e, --egat               Include rxn.barrier["egat"] as egat_barrier.
    -f, --forward            Include all levels in rxn.barrier.
    -r, --reverse            Include all levels in rxn.reverse_barrier.
    -g, --dg                 Include all levels in rxn.dg_rxn.
    -b, --barriers           Include forward and reverse barriers.
    -a, --all                Include forward, reverse, and reaction dG columns.

Usage:
    python export_rxn_smi.py [-icmefrgba] yarp.pkl output.csv
"""
import argparse
import pickle
import textwrap

import pandas as pd


def _format_optional_number(value):
    if value is None:
        return None
    try:
        return float(f"{float(value):.5g}")
    except (TypeError, ValueError):
        return value


def _selected_value_groups(args):
    if args.all:
        return [
            ("forward", "barrier"),
            ("reverse", "reverse_barrier"),
            ("reaction", "dg_rxn"),
        ]
    if args.barriers:
        return [("forward", "barrier"), ("reverse", "reverse_barrier")]

    groups = []
    if args.forward:
        groups.append(("forward", "barrier"))
    if args.reverse:
        groups.append(("reverse", "reverse_barrier"))
    if args.dg:
        groups.append(("reaction", "dg_rxn"))
    return groups


def _value_keys(rxns, attr):
    keys = set()
    for rxn in rxns.values():
        values = getattr(rxn, attr, None)
        if values:
            keys.update(values.keys())
    return sorted(keys)


def _value_columns(rxns, groups):
    columns = []
    for label, attr in groups:
        for key in _value_keys(rxns, attr):
            columns.append((label, attr, key))
    return columns


def _reaction_value(rxn, attr, key):
    values = getattr(rxn, attr, None)
    if not values:
        return None
    return values.get(key)


def _column_name(direction, attr, key):
    if attr == "dg_rxn":
        return f"reaction_dg_{key}"
    return f"{direction}_barrier_{key}"


def _state_smiles(state, attr):
    value = getattr(state, attr, None)
    if value:
        return value

    graph = getattr(state, "graph", None)
    if graph is not None:
        if getattr(graph, attr, None) is None and hasattr(graph, "get_smiles"):
            graph.get_smiles()
        return getattr(graph, attr, None)

    return None


def _load_pickle(filename):
    with open(filename, "rb") as handle:
        rxns = pickle.load(handle)
    if not isinstance(rxns, dict):
        raise TypeError(f"Expected a dictionary of reactions in {filename}, got {type(rxns).__name__}.")
    return rxns


def main(args):
    print("So I've heard you'd like some SMILES strings...")
    rxns = _load_pickle(args.filename)
    print(f"Loaded {len(rxns)} reactions from {args.filename}.")
    value_columns = _value_columns(rxns, _selected_value_groups(args))

    rows = []
    for rxn in rxns.values():
        row = {
            "rxn_id": getattr(rxn, "id", None),
            "reactant_canon_smi": _state_smiles(rxn.reactant, "canon_smi"),
            "product_canon_smi": _state_smiles(rxn.product, "canon_smi"),
        }

        if args.ids:
            row["rxn_hash"] = getattr(rxn, "hash", None)

        if args.mapped:
            row["reactant_map_smi"] = _state_smiles(rxn.reactant, "map_smi")
            row["product_map_smi"] = _state_smiles(rxn.product, "map_smi")

        if args.egat:
            row["egat_barrier"] = _format_optional_number(_reaction_value(rxn, "barrier", "egat"))

        for direction, attr, key in value_columns:
            row[_column_name(direction, attr, key)] = _format_optional_number(
                _reaction_value(rxn, attr, key)
            )

        rows.append(row)
    print(f"Extracted SMILES and values for {len(rows)} reactions.")
    base_cols = ["rxn_id", "reactant_canon_smi", "product_canon_smi"]
    if args.ids:
        base_cols.insert(1, "rxn_hash")
    if args.mapped:
        base_cols += ["reactant_map_smi", "product_map_smi"]
    if args.egat:
        base_cols.append("egat_barrier")
    extra_cols = [_column_name(direction, attr, key) for direction, attr, key in value_columns]

    output_cols = base_cols + extra_cols
    df = pd.DataFrame(rows, columns=output_cols)
    df.to_csv(args.output, index=False)
    print(f"...and now I've exported them to {args.output}!")


def _add_options(parser):
    parser.add_argument("-i", "--ids", action="store_true",
                        help="Include rxn_hash after rxn_id.")
    parser.add_argument("-c", "--canon", action="store_true",
                        help="Include canonical SMILES columns. This is the default output and the flag is accepted for symmetry.")
    parser.add_argument("-m", "--mapped", action="store_true",
                        help="Include reactant_map_smi and product_map_smi.")
    parser.add_argument("-e", "--egat", action="store_true",
                        help="Include rxn.barrier['egat'] as egat_barrier.")
    parser.add_argument("-f", "--forward", action="store_true",
                        help="Include one column for every key in rxn.barrier.")
    parser.add_argument("-r", "--reverse", action="store_true",
                        help="Include one column for every key in rxn.reverse_barrier.")
    parser.add_argument("-g", "--dg", action="store_true",
                        help="Include one column for every key in rxn.dg_rxn.")
    parser.add_argument("-b", "--barriers", action="store_true",
                        help="Include all forward and reverse barrier columns; equivalent to -fr.")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Include all forward, reverse, and reaction dG columns; equivalent to -frg.")


def cli():
    parser = argparse.ArgumentParser(
        description="Export reaction SMILES and optional reaction-value columns from a YARP pickle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Default columns:
              rxn_id, reactant_canon_smi, product_canon_smi

            Flag summary:
              -i  add rxn_hash
              -c  canonical SMILES columns (already included by default)
              -m  add mapped SMILES columns
              -e  add egat_barrier from rxn.barrier['egat']
              -f  add all forward barrier columns from rxn.barrier
              -r  add all reverse barrier columns from rxn.reverse_barrier
              -g  add all reaction dG columns from rxn.dg_rxn
              -b  add forward and reverse barriers (-fr)
              -a  add forward, reverse, and reaction dG columns (-frg)

            Examples:
              python export_rxn_smi.py yarp.pkl reactions.csv
              python export_rxn_smi.py -m yarp.pkl reactions.csv
              python export_rxn_smi.py -ime yarp.pkl reactions.csv
              python export_rxn_smi.py -imfrg yarp.pkl reactions.csv
              python export_rxn_smi.py -ima yarp.pkl reactions.csv
            """
        ),
    )
    parser.add_argument("filename", help="Path to the pickle file")
    parser.add_argument("output", help="Path to the CSV file with SMILES strings")
    _add_options(parser)
    main(parser.parse_args())


if __name__ == "__main__":
    cli()
