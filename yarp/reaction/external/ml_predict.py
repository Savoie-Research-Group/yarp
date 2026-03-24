from pathlib import Path
import shutil

from yarp.reaction.external.calc_base import AsyncYarpCalculator

class MLPredictTask(AsyncYarpCalculator):
    def has_prerequisites(self) -> bool:
        # ML usually just needs the 2D graph (SMILES/InChI), which is guaranteed to exist.
        # ERM: Ehhhhh is it though? We should put an actual check in this slot eventually...
        return True
    

class EgatMLPredict(MLPredictTask):
    def generate_input(self):
        # Write SMILES to a text file for EGAT to read
        pass
    def write_submission_script(self) -> Path:
        # Write script calling the EGAT container
        pass
    def check_output(self) -> bool:
        return (self.scratch_dir / "egat_results.csv").exists()
    def scrape_data(self):
        # Read CSV, assign self.rxn.barrier['egat'] = value
        pass
    def cleanup(self):
        # EGAT is lightweight, maybe just delete the folder
        shutil.rmtree(self.scratch_dir)