# Welcome to the YARP 3.0.0 tutorial suite!

Hmmm, what would be good to have in here...?

Things I should leave to the YARP 3.0 tutorial manuscript!
- Overview of product enumeration concepts
- Overview of reaction characterization concepts

## Materials for this tutorial - Detailed notes on the input file
- Explaining the `initialize` block
    - `verbose`, `output`, `initial_species`, `job_manager`, `enumeration` sub-blocks
        - Should `reactive_atoms` go in `enumeration`, rather than `initial_species`?
        - Seems to make sense as a `pre_enumeration_filter`...
        - I don't think it will be a relevant feature without product enumeration...!
    - Go through all options and default settings
- Explaining the `stages` blocks
    - Outlining the different `method` entries: `ml_rxn_prop`, `init_rxn_path`, `refine_rxn_path`
    - And then go through all the options and default settings for each `method`

## Materials for this tutorial - Common workflow walkthroughs
- Explaining the basics: `yarp-init`, `yarp-progress`, and `yarp-loop`
- Explaining helper scripts:
    - *I will return to this, as I should make some of these pip install executables!!!*
    - In the future, put the model reactions stuff here?
- Characterization of user provided reactions
    - EGAT only barrier prediction
        - Initialize from XYZ, SMILES, or YARP pickle file!
    - EGAT -> LL xTB TS searching -> LL xTB TS refinement
    - EGAT -> LL xTB TS searching -> LL xTB TS refinement -> HL DFT TS refinement
- Depth 1 product enumeration (stability degradation study)
    - Concerted enumeration -> EGAT
        - Closed-shell example (3HP)
        - Open-shell example (LiEC)
    - Sequential enumeration -> EGAT (maybe don't put this in until v3.0.1 or later?)
- Multi-depth product enumeration (chemical reaction network exploration)
    -  Concerted, closed-shell example (3HP)
        - Separate products (change this to True/False, rather than allowing user to select certain reactions; we need a better implementation if that's a desired feature)
        - Property filtering example -> EGAT barrier filtering
        - In the future, add a product blinders case study here!
