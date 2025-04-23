class Reaction:
    """

    Attributes:
    -----------

    self.reactant : yarpecule
        reactant molecule
        initialized based on user input

    self.product : yarpecule
        product molecule
        initialized based on user input or product enumeration

    self.ID : str
        unique identifier for reaction
        <insert description of how this is generated and then how it's used>

    self.status : dict
        key-value pairs of known stages and their status 


    """

    def __init__(self, reactant, product):
        self.reactant = reactant
        self.product = product

        self.status = dict()
        self.ID = "carl"

    def check_status(self, method):
        """
        method : str
            name of the method to check the status of

        Check if any of the pre-built methods have been completed for this reaction object.

        Expand self.status dictionary with True/False values for each method

        Or maybe have a more complex logic than simple True/False...?
        """

        # Look for some sort of a flag/file indicating "submitted", "running", "completed with error", "completed successfully"

    def compute(self, input):
        """
        Decide which method class to run based on input file.

        input : dict
            dictionary of input parameters parsed from input YAML file.
            should only contain parameters relevant to the current stage begin run.
        """

        if input.get('method') == 'initial_path':
            # basically, this will do the same thing as main_xtb.py
            # but there will be the flexibility to run LL methods besides xTB (i.e. AIMNet or other ML potentials)
            # also, we will ensure that inputs required for one part (i.e. conformers) will have no impact on separate parts (i.e. ts optimization)
            self.generate_conformers(input.get('conformer generation'))
            self.select_conformer_pair(input.get('conformer generation'))
            self.run_gsm(input.get('gsm'))
            self.optimize_ts(input.get('TS optimization'))
            self.validate_irc(input)

        elif input.get('method') == 'refine_path':
            # this will do the same thing as main_dft.py
            # but there will be flexiblity to run HL methods besides DFT (i.e. CC)
            self.rp_opt(input)
            self.optimize_ts(input.get('TS optimization'))
            self.validate_irc(input)

        # Allow user to isolate any of the individual steps within the refinement process
        # Should we also allow the user to isolate any of the individual steps within the initial path generation process?
        elif input.get('method') == 'rp_opt':
            self.rp_opt(input)

        elif input.get('method') == 'ts_opt':
            self.optimize_ts(input.get('TS optimization'))

        elif input.get('method') == 'irc_val':
            self.validate_irc(input)
