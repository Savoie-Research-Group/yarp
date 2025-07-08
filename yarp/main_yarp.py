import sys
import yaml

from yarp.util.input import input
from yarp.reaction.generate_rxns import generate_rxns


def main(file):

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
    inp = input(file)

    ###############################################
    ####         STAGE 1                       ####
    ###############################################

    # Read in initialize key to determine how to initialize reaction objects

    # BUT FIRST!!! Check CONTROL.yaml to see if the initialization has been done!

    reactions = generate_rxns(inp)

    if reactions == {}:
        print("No reaction objects created!")
        exit

    ###############################################
    ####         STAGE 2                       ####
    ###############################################

    # Iterate through each reaction and apply the appropriate methods
    for rxn in reactions:
        print(f"Processing reaction: {rxn.ID}")

        # Access the list of stage keys
        # Exit if stage keys are not defined
        stages = file.get('stages')
        if not stages:
            print("No stages defined in input YAML file. Exiting.")
            # find a way to dump out the reaction objects to a pickle file
            exit

        for stage in stages:
            # Check if the reaction object has already completed this step
            # Probably will interface with CONTROL.yaml file
            rxn.check_status(file.get(stage).get('method'))

            # If not, run the appropriate method
            if rxn.status.get('stage') == True:
                print(
                    f"Reaction has completed stage {stage}, progressing to next stage.")
                break
            else:
                print(
                    f"Running stage {stage} for reaction {rxn.rxn_id} with method {file.get(stage).get('method')}")
                rxn.compute(file.get(stage))


if __name__ == "__main__":
    inp = sys.argv[1]
    with open(inp, "r") as file:
        inp = yaml.safe_load(file)

    main(inp)
