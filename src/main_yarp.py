import sys
import yaml


def main(input):

    print(f"""Welcome to
                __   __ _    ____  ____  
                \ \ / // \  |  _ \|  _ \ 
                 \ V // _ \ | |_) | |_) |
                  | |/ ___ \|  _ <|  __/ 
                  |_/_/   \_\_| \_\_|
                          // Yet Another Reaction Program
        """)

    print("First off, here's the input file you provided:")
    print("=====================================")
    print(yaml.dump(input))
    print("=====================================")

    # Read in stages key as a list

    # Iterate through each stage, initializing different method classes as required


if __name__ == "__main__":
    input = sys.argv[1]
    with open(input, "r") as file:
        input = yaml.safe_load(file)

    main(input)
