import csv
from pathlib import Path
import shutil

from yarp.reaction.external.calc_base import AsyncYarpCalculator

class MLPredictTask(AsyncYarpCalculator):
    def has_prerequisites(self) -> bool:
        # A global task requires the full dictionary of reactions
        if not self.reactions:
            return False
        return True
    

class EgatMLPredict(MLPredictTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_name = "erm42/yarp:egat"
        
        # We need a way to track which SMILES string belongs to which reaction hash
        # so we can parse the results back to the right object later!
        self.smiles_to_hash = {}

    def generate_input(self):
        in_csv_path = self.scratch_dir / "in.csv"
        with open(in_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["reactions"])

            for rxn_hash, rxn in self.reactions.items():
                mapped_smiles = rxn.reactant.map_smi + ">>" + rxn.product.map_smi
                self.smiles_to_hash[mapped_smiles] = rxn_hash
                writer.writerow([mapped_smiles])
    

    def write_submission_script(self) -> Path:
        script_path = self.scratch_dir / "run_egat.sh"

        # Grab the container execution command from the base class
        prefix = self.get_container_prefix(self.image_name, str(self.scratch_dir))

        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n\n")

            self.write_scheduler_headers(f)

            if self.job_manager.module_container:
                f.write(f"{self.job_manager.module_container}\n\n")

            cmd = f"{prefix} egat --input in.csv --output out.csv"
            f.write(f"{cmd} > egat.log 2> egat.err\n")

        script_path.chmod(0o755)
        return script_path

    def check_output(self) -> bool:
        return (self.scratch_dir / "out.csv").exists()

    def scrape_data(self):
        out_csv_path = self.scratch_dir / "out.csv"

        with open(out_csv_path, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                rxn_smiles = row["reaction_smiles"]
                barrier = float(row["activation_barrier"])
                enthalpy = float(row["reaction_enthalpy"])

                # Retrieve the original reaction hash
                rxn_hash = self.smiles_to_hash.get(rxn_smiles)

                if rxn_hash:
                    rxn = self.reactions[rxn_hash]
                    rxn.barrier[self.config.model] = barrier
                    rxn.heat_of_rxn[self.config.model] = enthalpy

    def cleanup(self):
        # # EGAT is lightweight, maybe just delete the folder
        # shutil.rmtree(self.scratch_dir)
        pass