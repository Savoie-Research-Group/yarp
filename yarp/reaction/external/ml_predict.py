import csv
from pathlib import Path
import shutil

from yarp.reaction.external.calc_base import AsyncYarpCalculator
from yarp.reaction.ml_barrier import dense_reaction_smiles_for_egat

class MLPredictTask(AsyncYarpCalculator):
    def has_prerequisites(self) -> bool:
        # A global task requires the full dictionary of reactions
        if not self.reactions:
            return False
        return True
    

class EgatMLPredict(MLPredictTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_name = {'barrier': 'egat-barrier:test',
                           'enthalpy': 'egat-enthalpy:test'}
        
    def generate_input(self):
        model = self.config.model

        skipped_forward = 0
        forward_csv = self.scratch_dir / "forward_in.csv"
        with open(forward_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["reactions"])

            for rxn_hash, rxn in self.reactions.items():
                # Skip if already evaluated by this model (barrier and enthalpy)
                if hasattr(rxn, 'barrier') and model in rxn.barrier:
                    if hasattr(rxn, 'heat_of_rxn') and model in rxn.heat_of_rxn:
                        skipped_forward +=1
                        continue

                mapped_smiles = dense_reaction_smiles_for_egat(rxn.reactant.map_smi, rxn.product.map_smi)
                writer.writerow([mapped_smiles])

        skipped_reverse = 0
        reverse_csv = self.scratch_dir / "reverse_in.csv"
        with open(reverse_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["reactions"])

            for rxn_hash, rxn in self.reactions.items():
                # Skip if already evaluated by this model (barrier only)
                if hasattr(rxn, 'reverse_barrier') and model in rxn.reverse_barrier:
                    skipped_reverse += 1
                    continue

                mapped_smiles = dense_reaction_smiles_for_egat(rxn.product.map_smi, rxn.reactant.map_smi)
                writer.writerow([mapped_smiles])

        if skipped_forward > 0 or skipped_reverse > 0:
            print(f"   * Previously characterized reactions detected! Skipping {skipped_forward} forward and {skipped_reverse} reverse reactions!")

    def write_submission_script(self) -> Path:
        script_path = self.scratch_dir / "run_egat.sh"

        env_vars = {'EGAT_THREADS': self.config.n_cpus}

        # EGAT flags (--input/--output) follow Docker ENTRYPOINT; use `apptainer run`, not `exec`.
        bar_prefix = self.get_container_prefix(self.image_name['barrier'], str(self.scratch_dir), apptainer_run=True, env_vars=env_vars)
        enth_prefix = self.get_container_prefix(self.image_name['enthalpy'], str(self.scratch_dir), apptainer_run=True, env_vars=env_vars)

        with open(script_path, "w") as f:
            f.write("#!/bin/bash\n\n")

            self.write_scheduler_headers(f)

            f.write(f"cd {self.scratch_dir}\n")

            f.write("echo 'Running energy of activation barrier prediction'\n")

            bar_cmd1 = f"{bar_prefix} --input forward_in.csv --output forward_barrier_out.csv --no-enthalpy"
            f.write(f"{bar_cmd1} > forward_barrier.log 2> forward_barrier.err\n")

            bar_cmd2 = f"{bar_prefix} --input reverse_in.csv --output reverse_barrier_out.csv --no-enthalpy"
            f.write(f"{bar_cmd2} > reverse_barrier.log 2> reverse_barrier.err\n")

            f.write("echo 'Running enthalpy of reaction prediction'\n")

            enth_cmd1 = f"{enth_prefix} --input forward_in.csv --output forward_enthalpy_out.csv"
            f.write(f"{enth_cmd1} > forward_enthalpy.log 2> forward_enthalpy.err\n")


        script_path.chmod(0o755)
        return script_path

    def check_output(self) -> bool:
        barrier_done = (self.scratch_dir / "forward_barrier_out.csv").exists() and (self.scratch_dir / "reverse_barrier_out.csv").exists()
        enthalpy_done = (self.scratch_dir / "forward_enthalpy_out.csv").exists()

        return barrier_done and enthalpy_done

    def scrape_data(self):
        forward_smiles_to_hash = dict()
        reverse_smiles_to_hash = dict()
        for rxn_hash, rxn in self.reactions.items():
            fwd_smiles = dense_reaction_smiles_for_egat(rxn.reactant.map_smi, rxn.product.map_smi)
            forward_smiles_to_hash[fwd_smiles] = rxn_hash

            rev_smiles = dense_reaction_smiles_for_egat(rxn.product.map_smi, rxn.reactant.map_smi)
            reverse_smiles_to_hash[rev_smiles] = rxn_hash

        # Parse energy of activation barriers (forward and reverse)
        def parse_barrier(row):
            value = (row.get("activation_barrier") or "").strip()
            if not value:
                return None
            try:
                return float(value)
            except ValueError:
                return None

        forward_out_csv = self.scratch_dir / "forward_barrier_out.csv"
        with open(forward_out_csv, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                rxn_smiles = row["reaction_smiles"]
                barrier = parse_barrier(row)
                if barrier is None:
                    continue

                rxn_hash = forward_smiles_to_hash.get(rxn_smiles)
                if rxn_hash:
                    rxn = self.reactions[rxn_hash]
                    rxn.barrier[self.config.model] = barrier

        reverse_out_csv = self.scratch_dir / "reverse_barrier_out.csv"
        with open(reverse_out_csv, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                rxn_smiles = row["reaction_smiles"]
                barrier = parse_barrier(row)
                if barrier is None:
                    continue

                rxn_hash = reverse_smiles_to_hash.get(rxn_smiles)
                if rxn_hash:
                    rxn = self.reactions[rxn_hash]
                    rxn.reverse_barrier[self.config.model] = barrier

                    f_barrier = rxn.barrier.get(self.config.model)
                    if f_barrier is None:
                        continue

                    dg_rxn = barrier - f_barrier
                    rxn.dg_rxn[self.config.model] = dg_rxn

        # Parse heat of reaction (forward only)
        def parse_enthalpy(row):
            value = (row.get("activation_barrier") or "").strip()
            if not value:
                return None
            try:
                return float(value)
            except ValueError:
                return None

        enthalpy_csv = self.scratch_dir / "forward_enthalpy_out.csv"
        with open(enthalpy_csv, "r") as f:
            reader = csv.DictReader(f)

            for row in reader:
                rxn_smiles = row["reaction_smiles"]
                enthalpy = parse_enthalpy(row)
                if enthalpy is None:
                    continue

                rxn_hash = forward_smiles_to_hash.get(rxn_smiles)
                if rxn_hash:
                    rxn = self.reactions[rxn_hash]
                    rxn.heat_of_rxn[self.config.model] = enthalpy

    def cleanup(self):
        # remove everything except output csv files and submission script
        keep = {"forward_barrier_out.csv", "reverse_barrier_out.csv", "forward_enthalpy_out.csv", "run_egat.sh"}
        for item in self.scratch_dir.iterdir():
            if item.name not in keep:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
        return
