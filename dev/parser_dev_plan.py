"""
Python planning artifact for the current SMILES/parser refactor discussion.

This file is intentionally non-production. It captures the current target API,
agreed decisions, and unresolved implementation touchpoints in Python literals
so we can iterate on structure without editing the core parser yet.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


ATOM_INFO_SCHEMA = {
    "atom_index": "int",
    "atom_map": "int | None",
    "element": "str  # lowercase in stored dict for YARP consistency",
    "formal_charge": "int",
    "mass": "float  # default natural mass unless isotope override provided",
    "stereo": {
        "atom": "str | None  # '@' / '@@' / None",
        "bonds": "dict[int, str]  # '/' or '\\\\', owned by atom on the left/right rule",
    },
    "aromatic_input": "bool",
}


CURRENT_DECISIONS = {
    "edit_scope": [
        "keep code edits minimal",
        "match existing yarp-again style as closely as possible",
        "add as few new functions as possible",
        "do not add new functions without explicit confirmation",
    ],
    "_atom_info_shape": [
        "dictionary keyed by current atom index",
        "lowercase element field in stored metadata",
        "stereo retained as metadata only",
        "mass field always populated",
        "_atom_info should always be generated for every yarpecule, whether or not user atom maps are provided",
    ],
    "stereo_policy": [
        "deterministically assign stereo bond/atom metadata to the atom on the right",
        "do not emit stereo tokens in output SMILES for this workflow",
    ],
    "mapping_policy": [
        "preserve user-provided atom maps",
        "do not overwrite existing user maps",
        "do not duplicate user maps",
        "if no user atom maps are present, canonical reorder first and then assign maps",
        "xyz atom ordering must remain stable from reactant to product",
        "canonical reordering may update atom_index bookkeeping but must not mutate atom_map identity",
        "atom_index and atom_map are separate concepts and must stay separate",
        "partial user-mapped input should be preserved",
        "canonicalization should never overwrite existing user-provided maps",
        "enumeration should carry parent _atom_info forward instead of regenerating it from scratch",
    ],
    "aromatic_policy": [
        "start with kekulization",
        "if kekulization fails, fall back to current aromatic handling policy with a printed warning",
        "uppercase explicit-bond output is acceptable",
        "aromatic outputs may be non-deterministic if chemically relevant",
    ],
    "rdkit_integration": [
        "do not build a separate YARP-only SMILES generator as the main path",
        "step into the RDKit generation path where the file/mol is created",
        "assign atom maps from _atom_info into the RDKit molecule before mapped SMILES export",
        "keep canonical and mapped SMILES coming from RDKit once mapping is injected correctly",
        "provide a call-site fallback path where yarp SMILES parsing can fall back to RDKit with a printed warning",
        "mapped export can be driven from an in-memory RDKit Mol rather than a temp-file round trip",
    ],
    "fallback_policy": [
        "strict=False default",
        "print a warning when fallback behavior is used",
        "put the try/except wrapper at the call site, not inside smiles.py",
        "strict=False on the yarp SMILES parser path should fall back to RDKit if yarp parsing fails",
    ],
    "xyz_mol_population": [
        "_atom_info must also populate from xyz inputs",
        "_atom_info must also populate from mol inputs",
        "for xyz export, put the RDKit canonical smiles in the comment line",
        "do not expect xyz imports to provide mapping metadata",
        "for xyz inputs, generate atom mappings after canonicalization/adjacency generation",
    ],
    "verbose_policy": [
        "all new verbose flags default to False",
        "verbose=True should print the generated mol block line by line before map injection",
        "verbose=True should print the generated mol block line by line after map injection",
        "verbose=True should print written mol-file contents for inspection as well",
        "verbose=True should print everything available for the rdkit mol dump",
        "the dev notebook should call the verbose paths with verbose=True",
    ],
    "isotope_policy": [
        "input config gains optional isotope section",
        "proposed format: isotope: {12: 14}",
        "keys are atom_map values and values are isotope masses",
        "defaults should behave as absent or empty dict",
        "natural masses should auto-populate from yarp.util.properties.el_mass when no override is given",
        "applied isotope overrides should persist through enumeration via carried _atom_info",
    ],
}


TOUCHPOINTS = {
    "parser": [
        REPO_ROOT / "yarp/yarpecule/graph/smiles.py",
        REPO_ROOT / "yarp/yarpecule/input_parsers.py",
    ],
    "properties": [
        REPO_ROOT / "yarp/util/properties.py",
    ],
    "yarpecule": [
        REPO_ROOT / "yarp/yarpecule/yarpecule.py",
        REPO_ROOT / "yarp/yarpecule/atom_mapping.py",
        REPO_ROOT / "yarp/reaction/enum.py",
    ],
    "exports": [
        REPO_ROOT / "yarp/util/write_files.py",
    ],
    "tests": [
        REPO_ROOT / "test/yarpecule/graph/test_smiles.py",
        REPO_ROOT / "test/yarpecule/test_input_parser.py",
        REPO_ROOT / "test/yarpecule/test_yarpecule.py",
        REPO_ROOT / "test/conftest.py",
    ],
}


IMPLEMENTATION_SKETCH = {
    "smiles2adjmat": {
        "return_shape": "(adj_mat, bond_electron_mat, atom_info_dict)",
        "needs": [
            "dict-based atom_info",
            "stereo capture",
            "map preservation and fill logic",
            "lowercase element in metadata",
            "kekulize-first aromatic handling",
        ],
    },
    "xyz_from_smiles": {
        "return_shape": "(elements, geo, adj_mat, q, atom_info)",
        "needs": [
            "pass atom_info upstream",
            "shared atom_info schema between yarp and rdkit parse modes",
            "support call-site fallback from yarp parser to rdkit parser when strict=False",
        ],
    },
    "_read_structure": {
        "needs": [
            "initialize _atom_info for smiles/xyz/mol inputs",
            "accept carried tuple atom_info when provided by enum or other graph transforms",
            "apply canonical reorder before generated maps when user maps absent",
            "for xyz inputs, generate atom mappings after adjacency/canonicalization rather than reading them from file",
        ],
    },
    "_order_atoms": {
        "needs": [
            "reorder _atom_info alongside elements/adj_mat/geo/masses",
            "preserve existing atom_map values across reorder",
            "only atom_index should change during reorder bookkeeping",
        ],
    },
    "get_smiles": {
        "needs": [
            "inject _atom_info atom_map values into RDKit atoms before mapped export",
            "avoid overwriting user mapping semantics",
            "make it obvious how atom_index and atom_map correspond before and after map injection",
            "support verbose inspection of the RDKit-generated mol representation before and after mapping",
        ],
    },
    "exports": {
        "needs": [
            "write RDKit canonical smiles in xyz comment line",
            "write maps in mol atom mapping field",
        ],
    },
    "tests": {
        "needs": [
            "update parser tests that currently unpack four returns from mol_parse()",
            "update any xyz_from_smiles() call sites that unpack four returns",
            "migrate smiles parser tests from list-based atom_info assertions to dict-based assertions",
            "update exact map_smi string expectations once mapped smiles use _atom_info atom_map values",
            "add isotope override tests keyed by atom_map",
            "add enumeration persistence tests for carried _atom_info and isotope masses",
        ],
    },
    "enum": {
        "needs": [
            "carry parent _atom_info into tuple-created product yarpecules",
            "avoid regenerating atom maps and isotope-adjusted masses during enumeration",
        ],
    },
}


PROPOSED_API = {
    "smiles2adjmat": [
        "def smiles2adjmat(smiles, verbose=False):",
        "    # parser only; no rdkit fallback logic inside this function",
    ],
    "xyz_from_smiles": [
        "def xyz_from_smiles(smiles, mode='yarp'):",
        "    # returns (elements, geo, adj_mat, q, atom_info)",
    ],
    "yarpecule.__init__": [
        "def __init__(self, mol, mode='yarp', canon=True, strict=False):",
        "    # strict controls caller-side fallback behavior for smiles parsing only",
    ],
    "yarpecule._read_structure": [
        "def _read_structure(self, mol, mode, strict=False):",
        "    # try yarp smiles parse first when requested; if strict is False, warn and fall back to rdkit path",
    ],
    "yarpecule.get_smiles": [
        "def get_smiles(self, verbose=False):",
        "    # canonical and mapped smiles both come from rdkit after map injection",
        "    # verbose prints mol content before and after mapping",
    ],
    "xyz_write": [
        "def xyz_write(name, elements, geo, comment=None, append_opt=False):",
        "    # standard xyz body; comment line should carry RDKit canonical smiles",
    ],
}


CALL_FLOW_SKETCH = [
    "1. User requests yarpecule(smiles, mode='yarp', strict=False).",
    "2. _read_structure() calls the yarp parser path first.",
    "3. If yarp parsing succeeds, continue normally.",
    "4. If yarp parsing fails and strict is False, print a warning with the exception summary.",
    "5. Retry via rdkit-based smiles ingestion.",
    "6. Build _atom_info for the rdkit result using the same schema.",
    "7. Apply canonical reorder if requested.",
    "8. Preserve any user-provided atom_map values exactly, including partial-map inputs.",
    "9. If no user maps were supplied for some atoms, assign only the missing atom_map values after canonical reorder.",
    "10. Refresh atom_index bookkeeping after reorder without mutating preserved atom_map values.",
]


DATA_MODEL_NOTES = {
    "atom_index": [
        "tracks current local position in the active yarpecule ordering",
        "must be refreshed after any reorder/slice/join operation",
        "must not be used as a surrogate for atom_map",
        "should be the key used to find the corresponding atom record when building the RDKit molecule",
    ],
    "atom_map": [
        "external identity used for mapped smiles/export",
        "preserve exactly if provided by the user",
        "generate only when absent",
        "must survive canonical reorder unchanged",
        "should be assigned onto the RDKit atom properties from _atom_info[atom_index]['atom_map']",
    ],
    "xyz_input": [
        "never attempt to recover atom_map from xyz comments",
        "atom maps are generated after adjacency generation and optional canonical reorder",
    ],
    "xyz_output": [
        "comment line should carry RDKit canonical smiles, not raw map arrays",
    ],
}


NEXT_ITERATION_CHECKLIST = [
    "Reuse yarp.util.properties.el_mass rather than adding a duplicate atom-mass dictionary.",
    "Define the call-site try/fallback flow for strict=False yarp parsing to RDKit parsing.",
    "Prototype deterministic stereo ownership on the atom to the right.",
    "Implement verbose inspection output around in-memory RDKit mol generation and map injection.",
    "Define exactly how the notebook will write and inspect xyz and mol files for each yarpecule.",
    "Include lowercase naphthalene in the parser notebook and future parser tests.",
    "Make _read_structure() part of the first production migration so get_smiles()/export become testable on real yarpecules.",
    "Propose tuple-input and enum changes so parent _atom_info survives product generation.",
    "Propose YAML isotope parsing keyed by atom_map, e.g. isotope: {12: 14}.",
]


OPEN_QUESTIONS = [
    "For verbose=True output, besides the full RDKit mol dump, do you also want an explicit atom-index to atom-map table in the printout?",
]


if __name__ == "__main__":
    print("Parser refactor planning artifact")
    print()
    print("Atom info schema:")
    for key, value in ATOM_INFO_SCHEMA.items():
        print(f"  {key}: {value}")
    print()
    print("Current decisions:")
    for section, items in CURRENT_DECISIONS.items():
        print(f"  {section}:")
        for item in items:
            print(f"    - {item}")
