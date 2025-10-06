"""
Definition of input object class
"""


class input:
    """
    Class object to hold parameters set by user-provided input file.
    The goal of this object is to provide a central location to define
    all default settings used throughout YARP.

    Parameters:
    -----------
    file : dictionary
        Read in from input YAML file. Contains all user-set

    Attributes:
    -----------

    """

    ###############
    # Constructor #
    ###############

    def __init__(self, file):

        # Extract attributes from initialize node
        initnode = file.get('initialize', None)
        if not initnode:
            raise RuntimeError("Hey bro beans, I need some molecules or reactions to work with. "
                               "Missing `initialize` node in YAML file.")

        self.enum_on = initnode.get("enumerate", False)
        self.enum_mode = initnode.get("mode", "concerted")
        self.n_break = initnode.get("bonds to break", 2)
        self.n_form = initnode.get("bonds to form", 2)
        self.l_cutoff = initnode.get("lewis score", 0.0)
        self.d0_node = initnode.get("initial species", None)
        if not self.d0_node:
            raise RuntimeError("Please provide an initial species for enumeration. "
                               "Can be a single structure SMILES or XYZ, "
                               "or a previous YARP pickle file.")

        self.quick_opt_lot = initnode.get("quick opt lot", "uff")
        self.prod_visuals_on = initnode.get("visualize", False)

        self.out_file = initnode.get("output", "reactions.pkl")

        self.separate_prods = initnode.get("separate products", None)
        print(self.separate_prods, type(self.separate_prods))
        if self.separate_prods is None:
            self.separate_prods = []
        elif isinstance(self.separate_prods, str) and self.separate_prods.lower() == 'all':
            self.separate_prods = 'all'
        elif isinstance(self.separate_prods, int):
            self.separate_prods = [self.separate_prods]
        elif isinstance(self.separate_prods, list):
            self.separate_prods = self.separate_prods
        else:
            raise RuntimeError("Invalid value for separate products. Accepted inputs: 'all', an integer, or a list of integers")
        
