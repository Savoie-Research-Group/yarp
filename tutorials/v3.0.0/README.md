# Welcome to the YARP 3.0.0 tutorial suite!

## Contents of this tutorial

### `yarpecules`
Contains a Jupyter notebook with a step-by-step guide on the `yarpecule` object, which is used to represent molecules as molecular graphs

### `initial_structure`
Contains several examples of how to initialize YARP from different starting structure formats

### `product_enumeration`
Contains materials related to product enumeration
- `fundamentals`:
    - Jupyter notebook with a step-by-step guide on YARP's product enumeration strategy
    - Details the concerted and sequential enumeration modes
    - Explains the basic `post_enum_filter` options
- `depth1_crn`:
    - Contains an example of how to run product enumeration via `yarp-init`
    - Also includes input required to directly predict energy of activation barriers with EGAT
- `depth2_crn`:
    - Contains examples of how to run multiple rounds (depths) of product enumeration in order to build out multi-step chemical reaction networks via `yarp-init`
    - Includes demos of `pre_enum_filter` applications, such as `property_filtering` and `separate_products`

### `reaction-characterization`
Contains materials related to characterizing reactions in YARP
- `1_ml_rxn_prop`:
    - Direct prediction of reaction properties (i.e. energy of activation barriers) using ML models
- `2_init_rxn_path`:
    - Generation of conformers for reactants and products
    - Generation of initial guesses of transition states via growing string method
- `3_refine_rxn_path`:
    - Adds transition state refinement on top of `2_init_rxn_path`

## YARP execution scripts

There are three core YARP execution scripts:
1. `yarp-init`: (source code --> `yarp/initialize_yarp.py`)
    - Parses user input configuration, initializes reaction objects, and sets up task scheduler
    - If product enumeration is requested by user, that logic is executed here
    - How to use from command-line:
    ```bash
    cd /path/to/your/working/directory
    yarp-init input.yaml
    ```
2. `yarp-progress`: (source code --> `yarp/progress_yarp.py`)
    - Manages the directed acyclic graph workflow for individual YARP tasks
    - Checks active jobs, submits additional ones, analyzes completed jobs
    - Requires user to re-execute in order to complete all tasks specified by `yarp-init`
    - How to use from command-line:
    ```bash
    yarp-progress /path/to/your/working/directory
    ```
3. `yarp-loop`: (source code --> `yarp/run_yarp_loop.py`)
    - This is a convenience wrapper for `yarp-progress`, which will re-execute that script at set intervals
    - All YARP outputs will be written to a `yarp_loop.out` file, generated in the provided working directory
    - How to use from command-line (via nohup background processes):
    ```bash
    nohup yarp-loop -w /path/to/your/working/directory -i <interval_length_in_minutes> -d <total_duration_in_minutes> &
    ```

There are also various `helper` scripts that have been developed for examining YARP output files:
1. `yarp-read`: (source code --> `helper/read_pkl.py`)
    ```
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
    ```
2. `yarp-out`: (source code --> `helper/export_rxn_smi.py`)
    ```
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
    ```
