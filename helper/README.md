Here are some scripts meant to help users visualize and modify YARP outputs (i.e. pickle files)

- `read_pkl.py`: print a table view of reaction data in a pickle and optionally render PDFs
- `export_rxn_smi.py`: export canonical reaction SMILES to CSV, with optional mapped SMILES and barriers
- `update_yarp_pickles.py`: recursively migrate old pickle payloads to current reaction/state/yarpecule attributes

When installed, `read_pkl.py` is available as `yarp-read` and `export_rxn_smi.py`
is available as `yarp-out`.

`yarp-read` accepts `-i` for reaction hashes, `-f` for forward barriers, `-r`
for reverse barriers, `-g` for reaction dG, and `-b`/`-a` for grouped barrier
views. Short flags can be combined, e.g. `-ifrg`.

`yarp-out` mirrors the reaction-value flags from `yarp-read`: `-i` for reaction
hashes, `-m` for mapped SMILES, `-c` for canonical SMILES (included by
default), `-e` for `rxn.barrier["egat"]`, `-f` for forward barriers, `-r` for
reverse barriers, `-g` for reaction dG, and `-b`/`-a` for grouped barrier views.
