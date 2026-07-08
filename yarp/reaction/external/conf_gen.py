import shutil
from pathlib import Path

from yarp.reaction.external.calc_base import AsyncYarpCalculator
from yarp.yarpecule.input_parsers import xyz_parse
from yarp.reaction.conformer import conformer

class ConfTask(AsyncYarpCalculator):
    def has_prerequisites(self) -> bool:
        if not self.rxn.reactant.conformers.get('initial_geom') or not self.rxn.product.conformers.get('initial_geom'):
            return False
        return True


class CrestConfCalculator(ConfTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_name = "erm42/yarp:crest"
        self.xyz_file = "input.xyz"

        # Determine if we are working on the reactant or the product
        if "reactant" in self.task_def.task_type:
            self.target_species = self.rxn.reactant
        else:
            self.target_species = self.rxn.product

    def generate_input(self):
        """Write the initial 3D geometry for CREST to start from."""
        input_xyz_path = self.scratch_dir / self.xyz_file
        with open(input_xyz_path, "w") as f:
            # Assuming yarpecule has a method to get a basic 3D string
            # (e.g., generated via RDKit/ETKDG during initialization)
            f.write(self.target_species.conformers.get('initial_geom').to_xyz_string())

    def write_submission_script(self) -> Path:
        """Write the bash script that the JobManager will execute."""
        script_path = self.scratch_dir / "run_crest_cmd.sh"

        # When a seed is set, pin OMP/MKL threads inside the container so xTB's
        # geometry optimizations use the same thread count as CREST's -T setting.
        env_vars = None
        if self.config.seed is not None:
            env_vars = {
                "OMP_NUM_THREADS": self.config.n_cpus,
                "MKL_NUM_THREADS": self.config.n_cpus,
            }

        prefix = self.get_container_prefix(self.image_name, self.scratch_dir, env_vars=env_vars)
        crest_cmd = self._get_crest_command()
        full_command = f"{prefix} {crest_cmd}"

        try:
            with open(script_path, "w") as f:
                f.write("#!/bin/bash\n")
                self.write_scheduler_headers(f)
                f.write(f"cd {self.scratch_dir}\n")
                f.write(f"{full_command} > crest_run.log 2> crest_run.err\n")
        except PermissionError:
            raise PermissionError(
                f"Cannot write submission script to {script_path}. "
                f"Delete the SCRATCH directory and try again."
            )

        # Make the script executable (important for LocalJobManager)
        script_path.chmod(0x755)

        return script_path

    def check_output(self) -> bool:
        """Verify CREST actually finished and produced conformers."""
        # ERM: Should we add a check here to make sure there are at minimum n_conf available?
        xyz_file_name = self.scratch_dir / "crest_conformers.xyz"
        ene_file_name = self.scratch_dir / "crest.energies"

        xyz_exists = xyz_file_name.exists()
        energies_exists = ene_file_name.exists()

        termination_msg_exists = False
        outfile = self.scratch_dir / "crest_run.log"
        if outfile.exists():
            try:
                lines = open(outfile, 'r', encoding="utf-8").readlines()
                for n_line, line in enumerate(reversed(lines)):
                    if 'CREST terminated normally.' in line:
                        termination_msg_exists = True
            except:
                termination_msg_exists = False
            
        if not termination_msg_exists:
            print('   ! Successful termination message not found in crest_run.log. Check mem_per_cpu allocation for tasks using "crest"')

        if not (xyz_exists and energies_exists and termination_msg_exists):
            reasons = []
            if not xyz_exists:
                reasons.append("missing crest_conformers.xyz")
            if not energies_exists:
                reasons.append("missing crest.energies")
            if not termination_msg_exists:
                reasons.append("'CREST terminated normally.' not found in crest_run.log")
            print(f"     [CREST] Output validation failed: {'; '.join(reasons)}")

            # Surface any apptainer/container errors from stderr
            errfile = self.scratch_dir / "crest_run.err"
            if errfile.exists():
                try:
                    err_lines = open(errfile, 'r', encoding="utf-8").readlines()
                    if err_lines:
                        tail = err_lines[-10:]
                        print(f"     [CREST] Last lines of crest_run.err:")
                        for l in tail:
                            print(f"       {l}", end="")
                except Exception:
                    pass
            return False

        return True

    def scrape_data(self) -> bool:
        """Parse the XYZ and update self.target_species."""
        confs = self._get_all_conformers()
        for conf in confs:
            conf['lot'] = self.config.lot
            conf['software'] = 'crest'
            conf_obj = conformer(calc_type='conf_gen', calc_data=conf)
            self.target_species.conformers[conf_obj.type] = conf_obj

        return True

    def cleanup(self):
        """Delete CREST intermediate files, keeping conformer data and logs."""
        keep = {"crest_conformers.xyz", "cre_members", "crest.energies", "crest_run.log", self.xyz_file, "run_crest_cmd.sh"}    # SHQK : Keeping cre_members helps readily get the total number of generated conformers. Please keep it.
        for item in self.scratch_dir.iterdir():
            if item.name not in keep:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

    def _get_crest_command(self):

        # basic command (ERM: no way to set memory_per_cpu in CREST????)
        cmd = f"crest {self.xyz_file} --{self.config.lot} -nozs -T {self.config.n_cpus}"

        # molecular descriptors
        cmd += f" --chrg {self.config.charge} --uhf {self.config.n_unpaired_electrons}"

        if self.config.seed is not None:
            cmd += f" --seed {self.config.seed}"

        # conformer generation thresholds (ERM: expand this later, if needed)
        # ERM: no current way to cap CREST outputs at a set number of generated conformers!
        # You can damp down via adjusting the energy window threshold, but that's it
        # cmd += f" -ewin {self.config.energy_window}"

        # implicit solvation models
        alpb_solv = set([
            'acetone', 'acetonitrile', 'aniline', 'benzaldehyde', 'benzene',
            'ch2cl2', 'chcl3', 'cs2', 'dioxane', 'dmf', 'dmso', 'ether',
            'ethylacetate', 'furane', 'hexandecane', 'hexane', 'methanol',
            'nitromethane', 'octanol', 'woctanol', 'phenol', 'toluene',
            'thf', 'water'
        ])
        gbsa_solv = set([
            'acetone', 'acetonitrile', 'aniline', 'benzaldehyde',
            'CH2Cl2', 'CHCl3', 'CS2', 'DMSO', 'ether', 'H2O', 'methanol',
            'THF', 'toluene'
        ])
        if self.config.solvent is not None:
            model = self.config.solvent.get('model', '')
            solv = self.config.solvent.get('solvent', '')
            if model == 'alpb' and solv.lower() in alpb_solv:
                cmd += f" --{model} {solv}"
            elif model == 'gbsa' and solv.lower() in gbsa_solv:
                cmd += f" --{model} {solv}"

        return cmd

    def _get_all_conformers(self):
        """
        Get the entire set of geometry (and elements) from crest output files.
        Returns a dictionary for each conformer with the geometry, elements,
        relative energy ranking, and total energy in Eh
        """
        xyz_file_name = self.scratch_dir / "crest_conformers.xyz"

        confs=[]
        elements, geometries = xyz_parse(xyz_file_name, multiple=True)
        for count_i, i in enumerate(elements):
            conf = {
                'conf_rank': count_i,
                'elements': elements[count_i],
                'geometry': geometries[count_i]
            }
            confs.append(conf)

        return confs
