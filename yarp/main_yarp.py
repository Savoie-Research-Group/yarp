import sys
import yaml

from reaction import generate_rxns


def main(input):

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
    print(yaml.dump(input))
    print("=====================================")

    ###############################################
    ####         STAGE 1                       ####
    ###############################################

    # Read in initialize key to determine how to initialize reaction objects

    # BUT FIRST!!! Check CONTROL.yaml to see if the initialization has been done!

    initnode = input.get('initialize')
    if not initnode:
        raise RuntimeError(
            "Hey bro beans, I need some molecules or reactions to work with. Missing `initialize` node in YAML file.")

    reactions = dict()
    reactions = generate_rxns(initnode)

    ###############################################
    ####         STAGE 2                       ####
    ###############################################

    # Iterate through each reaction and apply the appropriate methods
    for rxn in reactions:
        print(f"Processing reaction: {rxn.ID}")

        # Access the list of stage keys
        # throw a RuntimeError if 'stages' doesn't exist
        stages = input.get('stages')
        if not stages:
            raise RuntimeError(
                "No stages provided for reaction object generation.")
        for stage in stages:
            # Check if the reaction object has already completed this step
            # Probably will interface with CONTROL.yaml file
            rxn.check_status(input.get(stage).get('method'))

            # If not, run the appropriate method
            if rxn.status.get('stage') == True:
                print(
                    f"Reaction has completed stage {stage}, progressing to next stage.")
                break
            else:
                print(
                    f"Running stage {stage} for reaction {rxn.rxn_id} with method {input.get(stage).get('method')}")
                rxn.compute(input.get(stage))


if __name__ == "__main__":
    inp = sys.argv[1]
    with open(inp, "r") as file:
        inp = yaml.safe_load(file)

    main(inp)
