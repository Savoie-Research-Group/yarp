import sys
import os
import yaml
import pickle
import omegaconf
from pathlib import Path

from yarp.util.input import InputParser
from yarp.reaction.generate_rxns import generate_rxns
from yarp.reaction.egat.predict_from_smiles import load_model
from yarp.reaction.ml_barrier import get_egat_barriers

def save_reactions(output, yp_rxns):
    """
    Write YARP reaction objects to a pickle file
    """
    with open(output, "wb") as f:
            pickle.dump(yp_rxns, f)
    print(f"Reactions dictionary has been pickled to {output}.")

def main(file):

    current_dir = Path(__file__).resolve().parent
    print(f"Script executing from {current_dir}")

    print(f"""Welcome to
                __   __ _    ____  ____  
                \ \ / // \  |  _ \|  _ \ 
                 \ V // _ \ | |_) | |_) |
                  | |/ ___ \|  _ <|  __/ 
                  |_/_/   \_\_| \_\_|
                          // Yet Another Reaction Program
        """)

    # Figure out a way to print the current version/commit hash
    print("First off, here's the input file you provided:")
    print("=====================================")
    print(yaml.dump(file))
    print("=====================================")

    # TO-DO: Check input file for bad syntax or missing things!

    # Initialize a class object to set default parameters
    inp = InputParser(file)

    ###############################################
    ####         STAGE 1                       ####
    ###############################################

    # Read in initialize key to determine how to initialize reaction objects

    # BUT FIRST!!! Check CONTROL.yaml to see if the initialization has been done!

    reactions = generate_rxns(inp)

    if reactions == {}:
        print("No reaction objects created!")
        sys.exit()
    else:
        print(f"Number of reaction objects initialized: {len(reactions)}")
        for index, rxn in enumerate(reactions.values()):
            print(f" -- Reaction {index}: {rxn.id} == {rxn.reactant.graph.canon_smi} -> {rxn.product.graph.canon_smi}")

    ###############################################
    ####         STAGE 2                       ####
    ###############################################

    # Access the list of stage keys
    # Exit if stage keys are not defined
    stages = file.get('stages')
    if not stages:
        print("No stages defined in input YAML file. Exiting.")
        save_reactions(inp.out_file, reactions)
        return

    for stage in stages:
        print(f'Processing stage {stage}')
        stg_inp = file.get(stage)
        method = stg_inp.get('method', None)

        if method is None:
            print("No method specified in stage. Exiting.")
            save_reactions(inp.out_file, reactions)
            return

        elif method == 'ml_predict':
            print("Reaction characterization via ML model selected")

            model = stg_inp.get('model', 'egat_pretrain')
            print(f' - Loading {model} model')
            if model == 'egat_pretrain':
                model_path = os.path.join(current_dir, '..', 'test', 'models', 'v1.pth')
                config_path = os.path.join(current_dir, '..', 'test', 'models', 'auto0.yaml')
                model, args = load_model(model_path, omegaconf.OmegaConf.load(config_path))
            else:
                print(f"Only available model is egat_pretrain. Please re-run with corrected input file. Exiting.")
                save_reactions(inp.out_file, reactions)
                return

            print(f' - Predicting barriers for {len(reactions)} reactions')
            reactions = get_egat_barriers(reactions, model, args)

            save_reactions(inp.out_file, reactions)

        else:
            print("Method not recognized. Exiting")
            save_reactions(inp.out_file, reactions)
            return


if __name__ == "__main__":
    inp = sys.argv[1]
    with open(inp, "r") as file:
        inp = yaml.safe_load(file)

    main(inp)
