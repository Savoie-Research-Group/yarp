"""
Upgrade legacy YARP pickle files in place or into a new output directory.

This helper rebuilds YARP reaction/state/yarpecule objects using native class
constructors, then backfills any additional legacy attributes. Hashes are
recomputed with YARP's native hash utilities.

How to use:

python update_yarp_pickles.py old.pkl
python update_yarp_pickles.py old.pkl --overwrite
python update_yarp_pickles.py old_pickles_dir --output-dir updated_pickles
"""

import argparse
import ast
from copy import deepcopy
import importlib
import os
import pickle
from pathlib import Path
import sys
import tempfile
from types import ModuleType


PICKLE_SUFFIXES = {".p", ".pkl", ".pickle"}
_HASH_MODULE = None

# Allow running this script from the repo root without pip-installing yarp.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLASS_SCHEMA_GUARDRAIL = {
    "reaction": {
        "file": PROJECT_ROOT / "yarp" / "reaction" / "reaction.py",
        "attrs": {
            "reactant",
            "product",
            "ts",
            "barrier",
            "reverse_barrier",
            "heat_of_rxn",
            "id",
            "hash",
            "network_meta",
        },
    },
    "state": {
        "file": PROJECT_ROOT / "yarp" / "reaction" / "state.py",
        "attrs": {
            "_graph",
            "conformers",
            "_species",
            "conc",
        },
    },
    "yarpecule": {
        "file": PROJECT_ROOT / "yarp" / "yarpecule" / "yarpecule.py",
        "attrs": {
            "_geo",
            "_elements",
            "_q",
            "_masses",
            "_adj_mat",
            "_atom_hashes",
            "_mapping",
            "_lewis_struct",
            "_bond_order_dict",
            "_yarpecule_hash",
            "_canon_smi",
            "_map_smi",
            "_inchi",
        },
    },
}


def _is_reaction_obj(obj):
    cls = obj.__class__
    return cls.__name__ == "reaction" and cls.__module__.startswith("yarp.reaction")


def _is_state_obj(obj):
    cls = obj.__class__
    return cls.__name__ == "state" and cls.__module__.startswith("yarp.reaction")


def _is_yarpecule_obj(obj):
    cls = obj.__class__
    return cls.__name__ == "yarpecule" and cls.__module__.startswith("yarp.yarpecule")


def _get_attr_any(obj, *names, default=None):
    for name in names:
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if value is not None:
            return value
    return default


def _coerce_dict(value, stats, field_name):
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return dict(value)
    except Exception:
        stats["dict_coercion_failures"] += 1
        print(f"WARNING: {field_name} is not dict-like ({type(value).__name__}); using empty dict.")
        return {}


def _safe_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    return [value]


def _get_canon_smi(obj):
    smi = _get_attr_any(obj, "canon_smi", "_canon_smi")
    if smi is not None:
        return smi
    get_smiles = _get_attr_any(obj, "get_smiles")
    if callable(get_smiles):
        try:
            get_smiles()
        except Exception:
            return None
        return _get_attr_any(obj, "canon_smi", "_canon_smi")
    return None


def _get_inchi(obj):
    return _get_attr_any(obj, "inchi", "_inchi")


def _init_stats():
    return {
        "objects_visited": 0,
        "reactions_rebuilt": 0,
        "states_rebuilt": 0,
        "yarpecules_rebuilt": 0,
        "reaction_dicts_rekeyed": 0,
        "reaction_hash_collisions": 0,
        "dict_coercion_failures": 0,
        "dict_key_preserved_nonprimitive": 0,
        "reaction_hash_recompute_failures": 0,
        "yarpecule_hash_recompute_failures": 0,
        "validation_warnings": 0,
        "changes": 0,
    }


def _extract_self_attr_names(target):
    attrs = set()
    if isinstance(target, ast.Attribute):
        if isinstance(target.value, ast.Name) and target.value.id == "self":
            attrs.add(target.attr)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for item in target.elts:
            attrs.update(_extract_self_attr_names(item))
    return attrs


def _scan_class_init_attrs(file_path, class_name):
    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file_path))

    class_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            class_node = node
            break

    if class_node is None:
        raise RuntimeError(f"Class `{class_name}` not found in {file_path}")

    init_node = None
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            init_node = node
            break

    if init_node is None:
        raise RuntimeError(f"`{class_name}.__init__` not found in {file_path}")

    attrs = set()
    for node in ast.walk(init_node):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                attrs.update(_extract_self_attr_names(target))
        elif isinstance(node, ast.AnnAssign):
            attrs.update(_extract_self_attr_names(node.target))
        elif isinstance(node, ast.AugAssign):
            attrs.update(_extract_self_attr_names(node.target))

    return attrs


def _run_preflight_guardrail():
    errors = []

    for class_name, cfg in CLASS_SCHEMA_GUARDRAIL.items():
        file_path = cfg["file"]
        expected = set(cfg["attrs"])

        try:
            found = _scan_class_init_attrs(file_path, class_name)
        except Exception as exc:
            errors.append(f"{class_name}: failed to scan {file_path} ({exc})")
            continue

        missing = sorted(expected - found)
        extra = sorted(found - expected)

        if missing or extra:
            msg = [f"{class_name}: schema drift detected in {file_path}"]
            if missing:
                msg.append(f"  missing expected attrs: {missing}")
            if extra:
                msg.append(f"  new attrs in code: {extra}")
            errors.append("\n".join(msg))

    if errors:
        print("ERROR: YARP class schema guardrail failed.")
        print("Migration script assumptions are out of sync with current code.")
        print("Please update `CLASS_SCHEMA_GUARDRAIL` in helper/update_yarp_pickles.py before running.")
        for item in errors:
            print(f"\n- {item}")
        raise SystemExit(2)


def _load_hash_module():
    global _HASH_MODULE

    if _HASH_MODULE is None:
        _HASH_MODULE = importlib.import_module("yarp.yarpecule.hashes")

    return _HASH_MODULE


def _construct_with_canon_fallback(cls, *args):
    try:
        return cls(*args, canon=False)
    except Exception:
        try:
            return cls(*args)
        except Exception:
            return None


def _clone_generic_object(obj, memo, stats):
    oid = id(obj)
    try:
        new_obj = obj.__class__.__new__(obj.__class__)
    except Exception:
        memo[oid] = obj
        return obj

    memo[oid] = new_obj

    # No YARP classes currently use __getstate__/__setstate__ or __slots__,
    # but this keeps fallback cloning safer for nested third-party objects.
    if hasattr(obj, "__getstate__") and hasattr(new_obj, "__setstate__"):
        try:
            state = obj.__getstate__()
            new_obj.__setstate__(_rebuild(state, memo, stats))
            return new_obj
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        for key, value in vars(obj).items():
            setattr(new_obj, key, _rebuild(value, memo, stats))

    slots = getattr(obj.__class__, "__slots__", ())
    if isinstance(slots, str):
        slots = (slots,)
    for slot in slots:
        if slot in ("__dict__", "__weakref__"):
            continue
        if hasattr(obj, slot):
            try:
                setattr(new_obj, slot, _rebuild(getattr(obj, slot), memo, stats))
            except Exception:
                pass

    return new_obj


def _copy_attrs(old_obj, new_obj, excluded, memo, stats):
    for key, value in vars(old_obj).items():
        if key in excluded:
            continue
        setattr(new_obj, key, _rebuild(value, memo, stats))
        stats["changes"] += 1


def _extract_graph(node):
    if node is None:
        return None
    if _is_yarpecule_obj(node):
        return node
    if _is_state_obj(node):
        return _get_attr_any(node, "_graph", "graph")
    return None


def _extract_yarpecule_core(yp):
    adj = _get_attr_any(yp, "_adj_mat", "adj_mat")
    geo = _get_attr_any(yp, "_geo", "geo")
    elements = _get_attr_any(yp, "_elements", "elements")
    charge = _get_attr_any(yp, "_q", "q")

    if adj is None or geo is None or elements is None or charge is None:
        return None

    return deepcopy(adj), deepcopy(geo), list(elements), int(charge)


def _rebuild_yarpecule(old_yp, memo, stats):
    oid = id(old_yp)
    core = _extract_yarpecule_core(old_yp)
    if core is None:
        return _clone_generic_object(old_yp, memo, stats)

    new_yp = _construct_with_canon_fallback(old_yp.__class__, core)
    if new_yp is None:
        return _clone_generic_object(old_yp, memo, stats)

    memo[oid] = new_yp

    excluded = {
        "_adj_mat",
        "_geo",
        "_elements",
        "_q",
        "_masses",
        "_atom_hashes",
        "_mapping",
        "_lewis_struct",
        "_bond_order_dict",
        "_yarpecule_hash",
    }
    _copy_attrs(old_yp, new_yp, excluded, memo, stats)

    # Preserve old string descriptors if present and constructor left them unset.
    for attr in ("_canon_smi", "_map_smi", "_inchi"):
        old_val = vars(old_yp).get(attr, None)
        if old_val is not None and getattr(new_yp, attr, None) is None:
            setattr(new_yp, attr, old_val)
            stats["changes"] += 1

    # Recompute hash with YARP native hash logic.
    try:
        hash_mod = _load_hash_module()
        new_yp._yarpecule_hash = hash_mod.yarpecule_hash(new_yp)
    except Exception:
        stats["yarpecule_hash_recompute_failures"] += 1

    stats["yarpecules_rebuilt"] += 1
    return new_yp


def _rebuild_state(old_state, memo, stats):
    oid = id(old_state)
    graph_old = _extract_graph(old_state)
    graph_new = _rebuild(graph_old, memo, stats) if graph_old is not None else None

    if graph_new is not None:
        new_state = _construct_with_canon_fallback(old_state.__class__, graph_new)
        if new_state is None:
            new_state = _clone_generic_object(old_state, memo, stats)
            memo[oid] = new_state
            return new_state
    else:
        new_state = _clone_generic_object(old_state, memo, stats)
        memo[oid] = new_state
        return new_state

    memo[oid] = new_state

    excluded = {"_graph", "_species", "conc", "conformers"}
    _copy_attrs(old_state, new_state, excluded, memo, stats)

    new_state._graph = graph_new

    # Keep constructor-generated _species to preserve current state invariants.
    if not hasattr(new_state, "_species") or new_state._species is None:
        new_state._species = []

    if "conformers" in vars(old_state):
        new_state.conformers = _safe_list(_rebuild(vars(old_state)["conformers"], memo, stats))
    elif not hasattr(new_state, "conformers") or new_state.conformers is None:
        new_state.conformers = []

    base_conc = _coerce_dict(getattr(new_state, "conc", None), stats, "state.conc")
    old_conc = None
    if "conc" in vars(old_state):
        old_conc = _coerce_dict(_rebuild(vars(old_state)["conc"], memo, stats), stats, "legacy state.conc")
    if old_conc:
        base_conc.update(old_conc)
    new_state.conc = base_conc

    for mol in new_state._species:
        smi = _get_canon_smi(mol)
        if smi and smi not in new_state.conc:
            new_state.conc[smi] = 0.0

    stats["states_rebuilt"] += 1
    return new_state


def _rebuild_reaction(old_rxn, memo, stats):
    oid = id(old_rxn)
    old_reactant = getattr(old_rxn, "reactant", None)
    old_product = getattr(old_rxn, "product", None)

    reactant_graph_old = _extract_graph(old_reactant)
    product_graph_old = _extract_graph(old_product)
    if reactant_graph_old is None or product_graph_old is None:
        return _clone_generic_object(old_rxn, memo, stats)

    reactant_graph_new = _rebuild(reactant_graph_old, memo, stats)
    product_graph_new = _rebuild(product_graph_old, memo, stats)

    try:
        new_rxn = old_rxn.__class__(reactant_graph_new, product_graph_new)
    except Exception:
        return _clone_generic_object(old_rxn, memo, stats)

    memo[oid] = new_rxn

    # If legacy payload stored enriched state objects, carry those over.
    # We recompute id/hash below to keep reaction metadata consistent.
    if _is_state_obj(old_reactant):
        new_rxn.reactant = _rebuild(old_reactant, memo, stats)
    if _is_state_obj(old_product):
        new_rxn.product = _rebuild(old_product, memo, stats)

    excluded = {"reactant", "product", "id", "hash"}
    _copy_attrs(old_rxn, new_rxn, excluded, memo, stats)

    # Keep known reaction fields in expected shape.
    for attr in ("ts", "barrier", "reverse_barrier", "network_meta"):
        setattr(new_rxn, attr, _coerce_dict(getattr(new_rxn, attr, None), stats, f"reaction.{attr}"))

    heat_of_reaction = _coerce_dict(getattr(new_rxn, "heat_of_reaction", None), stats, "reaction.heat_of_reaction")
    heat_of_rxn = _coerce_dict(getattr(new_rxn, "heat_of_rxn", None), stats, "reaction.heat_of_rxn")
    if len(heat_of_rxn) == 0 and len(heat_of_reaction) > 0:
        heat_of_rxn = dict(heat_of_reaction)
    if len(heat_of_reaction) == 0 and len(heat_of_rxn) > 0:
        heat_of_reaction = dict(heat_of_rxn)
    new_rxn.heat_of_rxn = heat_of_rxn
    new_rxn.heat_of_reaction = heat_of_reaction

    reactant_inchi = _get_inchi(getattr(new_rxn, "reactant", None))
    product_inchi = _get_inchi(getattr(new_rxn, "product", None))
    if reactant_inchi and product_inchi:
        new_rxn.id = f"{reactant_inchi}_to_{product_inchi}"

    # Recompute hash with YARP native hash logic.
    try:
        hash_mod = _load_hash_module()
        new_rxn.hash = hash_mod.reaction_hash(new_rxn)
    except Exception:
        stats["reaction_hash_recompute_failures"] += 1

    stats["reactions_rebuilt"] += 1
    return new_rxn


def _maybe_rekey_reaction_dict(data, stats):
    if not isinstance(data, dict) or len(data) == 0:
        return
    if not all(_is_reaction_obj(value) for value in data.values()):
        return

    keyed = {}
    for rxn in data.values():
        hsh = getattr(rxn, "hash", None)
        if hsh is None:
            return
        if hsh in keyed and keyed[hsh] is not rxn:
            stats["reaction_hash_collisions"] += 1
            # Keep first-seen entry to avoid silent last-write wins.
            continue
        keyed[hsh] = rxn

    if len(keyed) == 0:
        return

    if any(data.get(key) is not value for key, value in keyed.items()):
        data.clear()
        data.update(keyed)
        stats["reaction_dicts_rekeyed"] += 1
        stats["changes"] += 1


def _is_primitive_dict_key(key):
    return isinstance(key, (str, bytes, int, float, bool, type(None)))


def _rebuild(obj, memo, stats):
    if obj is None or isinstance(obj, (str, bytes, int, float, bool)):
        return obj

    if isinstance(obj, (ModuleType, type)):
        return obj

    oid = id(obj)
    if oid in memo:
        return memo[oid]

    stats["objects_visited"] += 1

    if _is_reaction_obj(obj):
        return _rebuild_reaction(obj, memo, stats)
    if _is_state_obj(obj):
        return _rebuild_state(obj, memo, stats)
    if _is_yarpecule_obj(obj):
        return _rebuild_yarpecule(obj, memo, stats)

    if isinstance(obj, dict):
        new_dict = {}
        memo[oid] = new_dict
        for key, value in obj.items():
            # Preserve original keys to avoid changing key hash/eq semantics.
            if not _is_primitive_dict_key(key):
                stats["dict_key_preserved_nonprimitive"] += 1
            new_dict[key] = _rebuild(value, memo, stats)
        _maybe_rekey_reaction_dict(new_dict, stats)
        return new_dict

    if isinstance(obj, list):
        new_list = []
        memo[oid] = new_list
        for value in obj:
            new_list.append(_rebuild(value, memo, stats))
        return new_list

    if isinstance(obj, tuple):
        new_tuple = tuple(_rebuild(value, memo, stats) for value in obj)
        memo[oid] = new_tuple
        return new_tuple

    if isinstance(obj, set):
        new_set = set()
        memo[oid] = new_set
        for value in obj:
            new_set.add(_rebuild(value, memo, stats))
        return new_set

    if isinstance(obj, frozenset):
        new_frozen = frozenset(_rebuild(value, memo, stats) for value in obj)
        memo[oid] = new_frozen
        return new_frozen

    if hasattr(obj, "__dict__"):
        return _clone_generic_object(obj, memo, stats)

    memo[oid] = obj
    return obj


def _collect_pickle_files(input_path):
    if input_path.is_file():
        return [input_path], None

    files = []
    for suffix in PICKLE_SUFFIXES:
        files.extend(input_path.rglob(f"*{suffix}"))
    files = sorted({path for path in files})
    return files, input_path


def _output_path_for(input_file, input_root, output_dir):
    if input_root is None:
        return output_dir / input_file.name
    rel_path = input_file.relative_to(input_root)
    return output_dir / rel_path


def validate_migrated_payload(payload):
    """
    Warning-only invariant checks after migration.
    Returns (warning_count, warning_samples).
    """
    warning_count = 0
    warning_samples = []
    seen = set()
    stack = [payload]

    def add_warning(message):
        nonlocal warning_count
        warning_count += 1
        if len(warning_samples) < 25 and message not in warning_samples:
            warning_samples.append(message)

    while stack:
        obj = stack.pop()
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)

        if isinstance(obj, dict):
            stack.extend(obj.values())
            continue
        if isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
            continue
        if obj is None or isinstance(obj, (str, bytes, int, float, bool, ModuleType, type)):
            continue

        if _is_reaction_obj(obj):
            reactant = getattr(obj, "reactant", None)
            product = getattr(obj, "product", None)
            if not _is_state_obj(reactant):
                add_warning("reaction.reactant is not a state object")
            if not _is_state_obj(product):
                add_warning("reaction.product is not a state object")
            for attr in ("ts", "barrier", "reverse_barrier", "network_meta", "heat_of_rxn"):
                if not isinstance(getattr(obj, attr, None), dict):
                    add_warning(f"reaction.{attr} is not a dict")
            if getattr(obj, "hash", None) is None:
                add_warning("reaction.hash is missing")

        if _is_state_obj(obj):
            graph = _extract_graph(obj)
            if graph is None or not _is_yarpecule_obj(graph):
                add_warning("state._graph is missing or not a yarpecule")
            if not isinstance(getattr(obj, "_species", None), list):
                add_warning("state._species is not a list")
            if not isinstance(getattr(obj, "conc", None), dict):
                add_warning("state.conc is not a dict")
            if not isinstance(getattr(obj, "conformers", None), list):
                add_warning("state.conformers is not a list")

        if _is_yarpecule_obj(obj):
            for attr in ("_adj_mat", "_geo", "_elements", "_q", "_yarpecule_hash"):
                if getattr(obj, attr, None) is None:
                    add_warning(f"yarpecule.{attr} is missing")
                    break

        if hasattr(obj, "__dict__"):
            stack.extend(vars(obj).values())

    return warning_count, warning_samples


def _write_pickle_atomic(output_file, payload):
    output_file.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=str(output_file.parent),
            prefix=f".{output_file.name}.",
            suffix=".tmp",
        ) as tmp_handle:
            tmp_path = Path(tmp_handle.name)
            pickle.dump(payload, tmp_handle, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_handle.flush()
            os.fsync(tmp_handle.fileno())

        os.replace(tmp_path, output_file)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _process_pickle(input_file, output_file):
    with open(input_file, "rb") as handle:
        payload = pickle.load(handle)

    stats = _init_stats()
    rebuilt_payload = _rebuild(payload, memo={}, stats=stats)
    warning_count, warning_samples = validate_migrated_payload(rebuilt_payload)
    if warning_count:
        stats["validation_warnings"] += warning_count
        print(f"WARNING: Found {warning_count} validation warning(s) in migrated payload.")
        for message in warning_samples[:10]:
            print(f"  - {message}")
        if warning_count > len(warning_samples[:10]):
            print(f"  - ... {warning_count - len(warning_samples[:10])} more")

    _write_pickle_atomic(output_file, rebuilt_payload)

    return stats


def _merge_stats(total, current):
    for key, value in current.items():
        total[key] += value


def main(args):
    _run_preflight_guardrail()

    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files, input_root = _collect_pickle_files(input_path)
    if not files:
        raise RuntimeError(f"No pickle files found under: {input_path}")

    output_dir = Path(args.output_dir).expanduser().resolve()

    print(f"Found {len(files)} pickle file(s) to process.")
    if args.overwrite:
        print("Overwrite mode enabled; original files will be replaced.")
    else:
        print(f"Writing updated pickles to: {output_dir}")

    total_stats = _init_stats()
    failures = []

    for index, input_file in enumerate(files, start=1):
        if args.overwrite:
            output_file = input_file
        else:
            output_file = _output_path_for(input_file, input_root, output_dir)

        print(f"[{index}/{len(files)}] Updating {input_file}")
        try:
            file_stats = _process_pickle(input_file, output_file)
            _merge_stats(total_stats, file_stats)
            print(
                f"    done -> {output_file} "
                f"(changes={file_stats['changes']}, "
                f"rxns={file_stats['reactions_rebuilt']}, "
                f"states={file_stats['states_rebuilt']}, "
                f"yarpecules={file_stats['yarpecules_rebuilt']}, "
                f"validation_warnings={file_stats['validation_warnings']})"
            )
        except Exception as exc:
            failures.append((input_file, str(exc)))
            print(f"    failed -> {exc}")

    print("\nSummary")
    print(f"  objects visited: {total_stats['objects_visited']}")
    print(f"  reactions rebuilt: {total_stats['reactions_rebuilt']}")
    print(f"  states rebuilt: {total_stats['states_rebuilt']}")
    print(f"  yarpecules rebuilt: {total_stats['yarpecules_rebuilt']}")
    print(f"  reaction dicts rekeyed: {total_stats['reaction_dicts_rekeyed']}")
    print(f"  reaction hash collisions: {total_stats['reaction_hash_collisions']}")
    print(f"  dict coercion failures: {total_stats['dict_coercion_failures']}")
    print(f"  non-primitive dict keys preserved: {total_stats['dict_key_preserved_nonprimitive']}")
    print(f"  reaction hash recompute failures: {total_stats['reaction_hash_recompute_failures']}")
    print(f"  yarpecule hash recompute failures: {total_stats['yarpecule_hash_recompute_failures']}")
    print(f"  validation warnings: {total_stats['validation_warnings']}")
    print(f"  total changes: {total_stats['changes']}")
    print(f"  files failed: {len(failures)}")

    if failures:
        print("\nFailures")
        for failed_file, error in failures:
            print(f"  {failed_file}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upgrade legacy YARP pickle files.")
    parser.add_argument(
        "input_path",
        help="Input pickle file or directory containing pickle files.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="updated_pickles",
        help="Output directory used when --overwrite is not set. Default: updated_pickles",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite input pickle file(s) in place.",
    )

    main(parser.parse_args())
