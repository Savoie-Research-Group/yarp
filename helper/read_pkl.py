"""
Print a table view of reactions in a YARP pickle.

Default output:
    Reactant, Product, and all forward barrier levels in rxn.barrier.

Options:
    -i, --ids       Include reaction hashes.
    -f, --forward   Include all levels in rxn.barrier.
    -r, --reverse   Include all levels in rxn.reverse_barrier.
    -g, --dg        Include all levels in rxn.dg_rxn.
    -b, --barriers  Include forward and reverse barriers.
    -a, --all       Include forward, reverse, and reaction dG columns.
    --limit N       Print at most N reactions.
    --visualize     Write reactant/product PDFs under visuals/.

Short flags can be combined, e.g. -ifrg.

Usage:
    python read_pkl.py [-ifrgba] [--visualize] yarp.pkl
"""
import argparse
import os
import pickle
import textwrap



from pathlib import Path

from tabulate import tabulate


def _format_optional_value(value):
    """Format a numeric value for display, handling missing values as 'none'."""
    if value is None:
        return "none"
    try:
        return f"{float(value):.5g}"
    except (TypeError, ValueError):
        return str(value)


def _nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("--limit must be greater than or equal to 0")
    return parsed


def _selected_value_groups(args, default=()):
    """Determine which reaction value dictionaries to include."""
    if args.all:
        return [
            ("forward", "barrier", "dG_activation"),
            ("reverse", "reverse_barrier", "dG_activation"),
            ("reaction", "dg_rxn", "dG_RXN"),
        ]
    if args.barriers:
        return [
            ("forward", "barrier", "dG_activation"),
            ("reverse", "reverse_barrier", "dG_activation"),
        ]

    groups = []
    if args.forward:
        groups.append(("forward", "barrier", "dG_activation"))
    if args.reverse:
        groups.append(("reverse", "reverse_barrier", "dG_activation"))
    if args.dg:
        groups.append(("reaction", "dg_rxn", "dG_RXN"))

    return groups or list(default)


def _value_keys(rxns, attr):
    keys = set()
    for rxn in rxns.values():
        values = getattr(rxn, attr, None)
        if values:
            keys.update(values.keys())
    return sorted(keys)


def _value_columns(rxns, groups):
    columns = []
    for direction, attr, label in groups:
        for key in _value_keys(rxns, attr):
            columns.append((direction, attr, label, key))
    return columns


def _reaction_value(rxn, attr, key):
    values = getattr(rxn, attr, None)
    if not values:
        return None
    return values.get(key)


def _state_smiles(state, attr="canon_smi"):
    value = getattr(state, attr, None)
    if value:
        return value

    graph = getattr(state, "graph", None)
    if graph is not None:
        if getattr(graph, attr, None) is None and hasattr(graph, "get_smiles"):
            graph.get_smiles()
        return getattr(graph, attr, None) or "none"

    return "none"


def _draw_state(state, outfile):
    graph = getattr(state, "graph", None)
    if graph is None:
        graph = getattr(state, "_graph", None)
    if graph is None:
        raise AttributeError("Reaction state does not expose a graph to draw.")
    graph.draw_bmats(outfile=outfile)


def _load_pickle(filename):
    with open(filename, "rb") as handle:
        rxns = pickle.load(handle)
    if not isinstance(rxns, dict):
        raise TypeError(f"Expected a dictionary of reactions in {filename}, got {type(rxns).__name__}.")
    return rxns


def main(args):
    rxns = _load_pickle(args.filename)
    print(f"Well folks, I've got {len(rxns)} reactions from {args.filename}")
    selected_rxns = list(rxns.values())
    if args.limit is not None:
        selected_rxns = selected_rxns[:args.limit]
        print(f"Displaying {len(selected_rxns)} reactions due to --limit {args.limit}")

    groups = _selected_value_groups(
        args,
        default=[("forward", "barrier", "dG_activation")],
    )
    columns = _value_columns(rxns, groups)

    headers = ["Reactant", "Product"]
    if args.ids:
        headers.insert(0, "Reaction Hash")
    for direction, _, label, key in columns:
        headers.append(f"{key} {direction} {label}")

    data = []
    for rxn in selected_rxns:
        row = [
            _state_smiles(rxn.reactant, "canon_smi"),
            _state_smiles(rxn.product, "canon_smi"),
        ]
        if args.ids:
            row.insert(0, getattr(rxn, "hash", "none"))

        for _, attr, _, key in columns:
            row.append(_format_optional_value(_reaction_value(rxn, attr, key)))
        data.append(row)

        if args.visualize:
            folder = Path(args.visual_dir) / str(getattr(rxn, "id", getattr(rxn, "hash", "reaction")))
            os.makedirs(folder, exist_ok=True)
            _draw_state(rxn.reactant, folder / "reactant.pdf")
            _draw_state(rxn.product, folder / "product.pdf")
    print(f"And here they are!")
    print(tabulate(data, headers=headers, tablefmt=args.tablefmt))


def _add_options(parser):
    parser.add_argument("-i", "--ids", action="store_true",
                        help="Include reaction hashes as the first table column.")
    parser.add_argument("--visualize", action="store_true",
                        help="Write reactant/product BEM PDFs for each reaction.")
    parser.add_argument("--visual-dir", default="visuals",
                        help="Directory for --visualize output. Default: visuals")
    parser.add_argument("--tablefmt", default="pretty",
                        help="tabulate table format. Default: pretty")
    parser.add_argument("--limit", type=_nonnegative_int, metavar="N",
                        help="Print at most N reactions. Default: no limit")
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
        description="Print a table view of reactions in a YARP pickle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Default columns:
              Reactant, Product, and all forward barrier columns from rxn.barrier

            Flag summary:
              -i  add reaction hash column
              -f  show all forward barrier columns from rxn.barrier
              -r  show all reverse barrier columns from rxn.reverse_barrier
              -g  show all reaction dG columns from rxn.dg_rxn
              -b  show forward and reverse barriers (-fr)
              -a  show forward, reverse, and reaction dG columns (-frg)
              --limit N  print at most N reactions
              --visualize  write reactant.pdf and product.pdf for each reaction
              --visual-dir DIR  choose the visualization output directory
              --tablefmt FMT  choose any tabulate format, e.g. pretty, github, csv

            Examples:
              python read_pkl.py yarp.pkl
              python read_pkl.py -i yarp.pkl
              python read_pkl.py -ifrg yarp.pkl
              python read_pkl.py --limit 25 yarp.pkl
              python read_pkl.py -ia --tablefmt github yarp.pkl
              python read_pkl.py --visualize --visual-dir visuals yarp.pkl
            """
        ),
    )
    parser.add_argument("filename", help="Path to the pickle file")
    _add_options(parser)
    main(parser.parse_args())


if __name__ == "__main__":
    cli()
