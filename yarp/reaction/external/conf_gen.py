import shutil
from pathlib import Path

from rdkit import Chem

from yarp.reaction.external.calc_base import AsyncYarpCalculator, CalculatorInputError
from yarp.reaction.enum_support import return_radicals
from yarp.yarpecule.input_parsers import xyz_parse
from yarp.yarpecule.graph.adjacency import compare_adjacency
from yarp.reaction.conformer import conformer
from yarp.util.rdkit import yarpecule_to_rdmol

# Conformer key prefix written by the xTB pre-optimization.
PREOPT_PREFIX = "preopt"

# MD timestep (fs) used when the system carries a free diatomic fragment.
#
# CREST's metadynamics defaults to 5 fs, which is only stable because SHAKE
# constrains the bonds -- and SHAKE can only constrain bonds present in the
# topology. xtb perceives bonds by a covalent-radius cutoff, roughly 0.768 A
# for H-H, while GFN2's equilibrium H-H is 0.7750 A. So an xTB-relaxed free H2
# falls just outside its own topology cutoff, never gets constrained, and a
# 5 fs step on an unconstrained ~4400 cm-1 oscillator (period ~7.6 fs) diverges
# immediately: the metadynamics runs "terminate EARLY" and CREST then crashes
# sorting an ensemble that is too small.
#
# 1.0 fs gives ~7.6 integration steps per period. 2.0 fs also worked when
# measured but gives only ~3.8, which is uncomfortably close to the edge.
# Measured cost is ~2.5x runtime with identical conformer counts -- a smaller
# timestep is strictly more accurate MD, so this buys safety with wall time and
# nothing else. Evidence: debug/.../implementation_checks/08_crest_h2/.
DIATOMIC_MD_TIMESTEP_FS = 1.0

# Elements each Auto3D potential is trained on (lowercase, as yarpecule.elements).
# AIMNet2 and its spin-aware variant: the elements with a trained per-element
# energy shift in the weights the auto3d_conf image ships (both models, the same
# 14). ANI-2x: Auto3D 3.2.2's own ANI_ELEMENTS. Evidence:
# Read from the model weights and Auto3D's source (engines/models/policy.py).
_AIMNET2_ELEMENTS = frozenset({"h", "b", "c", "n", "o", "f", "si", "p", "s", "cl", "as", "se", "br", "i"})
AUTO3D_ELEMENTS = {
    "aimnet2": _AIMNET2_ELEMENTS,
    "aimnet2-nse": _AIMNET2_ELEMENTS,
    "ani2x": frozenset({"h", "c", "n", "o", "f", "s", "cl"}),
}


class ConfTask(AsyncYarpCalculator):
    @property
    def target_species(self):
        """The state this task generates conformers for."""
        return self.rxn.reactant if "reactant" in self.task_def.task_type else self.rxn.product

    def preopt_conformer(self):
        """
        The xTB pre-optimized geometry this task starts from, or None.

        Conformer generation used to start from `initial_geom`, the raw
        yarpecule graph geometry -- which for an enumerated product is the
        parent's coordinates under the product's bonding, and can be wildly
        strained. The pre-optimization stage now supplies a relaxed structure,
        and there is no path that skips it.
        """
        for key, conf in self.target_species.conformers.items():
            if key.startswith(PREOPT_PREFIX) and conf.geo is not None:
                return conf
        return None

    def has_prerequisites(self) -> bool:
        # Only this task's own side. The reactant and product legs of the
        # pre-optimization finish at different times -- the product leg waits on
        # the reactant leg -- so requiring both here would fail the reactant's
        # conformer task the moment its dependency was satisfied.
        return self.preopt_conformer() is not None

    def has_free_diatomic(self) -> bool:
        """
        Whether any fragment of this state is a free two-atom molecule.

        A diatomic is the only fragment that can be left wholly unconstrained
        when its single bond falls outside xtb's perception cutoff, which is
        what breaks CREST's default 5 fs metadynamics (see
        DIATOMIC_MD_TIMESTEP_FS). Measured on H2; applied to every diatomic
        because the cost lands only on the runs that carry one, and a diatomic
        has no intramolecular conformational sampling to slow down anyway.
        """
        return any(len(frag.elements) == 2 for frag in self.target_species.species)


class CrestConfCalculator(ConfTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_name = "erm42/yarp:crest"
        self.xyz_file = "input.xyz"

    def generate_input(self):
        """Write the pre-optimized 3D geometry for CREST to start from."""
        initial_conf = self.preopt_conformer()
        if initial_conf is None:
            raise CalculatorInputError(
                f"No '{PREOPT_PREFIX}_*' conformer on the "
                f"{'reactant' if 'reactant' in self.task_def.task_type else 'product'}; "
                "the xTB pre-optimization has not produced a geometry for this species."
            )

        input_xyz_path = self.scratch_dir / self.xyz_file
        with open(input_xyz_path, "w") as f:
            f.write(initial_conf.to_xyz_string())

    def _warn_if_o2_multiplicity_mismatch(self):
        """
        CREST needs O2 to be run as a triplet (n_unpaired_electrons = 2) to converge;
        ground-state O2 is a triplet, not a singlet. YARP applies a single, user-configured
        n_unpaired_electrons value to the whole reactant/product state, so there's no way
        to special-case O2 without overriding what the user explicitly asked for.
        Instead, just warn loudly and let the (likely doomed) CREST job run anyway.
        """
        species_label = "reactant" if "reactant" in self.task_def.task_type else "product"
        species = self.rxn.reactant if species_label == "reactant" else self.rxn.product
    
        has_o2 = any(
            len(sp.elements) == 2 and all(el.lower() == 'o' for el in sp.elements)
            for sp in species.species
        )
        if has_o2 and self.config.n_unpaired_electrons != 2:
            print(
                f"   ! WARNING: Detected diatomic O2 in the {species_label} species for "
                f"task '{self.task_def.task_type}', but conf_gen is configured with "
                f"n_unpaired_electrons={self.config.n_unpaired_electrons}. Ground-state O2 is a "
                f"triplet (n_unpaired_electrons=2), and CREST is unlikely to converge for O2 run "
                f"as anything else. Proceeding with the configured multiplicity anyway, but expect "
                f"this CREST job to fail."
            )

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

        # A free diatomic needs a shorter MD timestep or the metadynamics
        # diverges; see DIATOMIC_MD_TIMESTEP_FS. Only applied when one is
        # present, so the ~70% of systems without one keep CREST's default 5 fs.
        if self.has_free_diatomic():
            cmd += f" --tstep {DIATOMIC_MD_TIMESTEP_FS}"

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


class RdkitConfCalculator(ConfTask):
    """
    RDKit ETKDG conformer generation, ported from classy_yarp's conf_rdkit().

    The embedding and force-field optimization run in the rdkit_conf container
    (containers/rdkit_conf/run_conf_gen.py), which writes every conformer
    lowest energy first. The connectivity filter classy_yarp applied runs here,
    on the host, in scrape_data.

    Auto3dConfCalculator runs the same lifecycle on another image; the class
    attributes below are what it changes.
    """

    TERMINATION_MSG = "RDKit conformer generation terminated normally."
    IMAGE_NAME = "erm42/yarp:rdkit_conf"
    # Output files are <FILE_PREFIX>_conformers.xyz, <FILE_PREFIX>.energies, ...
    FILE_PREFIX = "rdkit"
    # Name in log messages, and the conformers' software tag.
    LABEL = "RDKit"
    SOFTWARE = "rdkit"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_name = self.IMAGE_NAME
        self.mol_file = "input.mol"
        self.xyz_file = f"{self.FILE_PREFIX}_conformers.xyz"
        self.energies_file = f"{self.FILE_PREFIX}.energies"
        self.log_file = f"{self.FILE_PREFIX}_run.log"
        self.err_file = f"{self.FILE_PREFIX}_run.err"
        self.script_file = f"run_{self.FILE_PREFIX}_cmd.sh"

    def generate_input(self):
        """
        Write the state's graph as a MOL file for the container to embed.

        The mol comes straight from yarpecule_to_rdmol, so RDKit receives the
        yarpecule's own bonding, charges and radicals rather than re-deriving
        them. The pre-optimized geometry is attached for stereo perception;
        EmbedMultipleConfs discards it when it embeds.
        """
        initial_conf = self.preopt_conformer()
        if initial_conf is None:
            raise CalculatorInputError(
                f"No '{PREOPT_PREFIX}_*' conformer on the "
                f"{'reactant' if 'reactant' in self.task_def.task_type else 'product'}; "
                "the xTB pre-optimization has not produced a geometry for this species."
            )

        graph = self.target_species.graph
        mol = yarpecule_to_rdmol(
            elements=graph.elements,
            adj=graph.adj_mat,
            bond_orders=graph.bond_mats[0],
            atom_info=graph._atom_info,
            geo=initial_conf.geo,
        )
        Chem.MolToMolFile(mol, str(self.scratch_dir / self.mol_file))

    def _container_args(self) -> str:
        """Arguments for the image's entrypoint script (run_conf_gen.py)."""
        args = (
            f"{self.mol_file} --lot {self.config.lot} --n_conf {self.config.n_conf} "
            f"--prune_rms_thresh {self.config.prune_rms_thresh} --n_threads {self.config.n_cpus}"
        )
        if self.config.seed is not None:
            args += f" --seed {self.config.seed}"
        return args

    def write_submission_script(self) -> Path:
        """Write the bash script that the JobManager will execute."""
        script_path = self.scratch_dir / self.script_file

        # The image's entrypoint is run_conf_gen.py, so only its arguments follow.
        prefix = self.get_container_prefix(self.image_name, self.scratch_dir, apptainer_run=True)

        try:
            with open(script_path, "w") as f:
                f.write("#!/bin/bash\n")
                self.write_scheduler_headers(f)
                f.write(f"cd {self.scratch_dir}\n")
                f.write(f"{prefix} {self._container_args()} > {self.log_file} 2> {self.err_file}\n")
        except PermissionError:
            raise PermissionError(
                f"Cannot write submission script to {script_path}. "
                f"Delete the SCRATCH directory and try again."
            )

        script_path.chmod(0o755)

        return script_path

    def check_output(self) -> bool:
        """Verify the container finished and wrote its conformers."""
        xyz_exists = (self.scratch_dir / self.xyz_file).exists()
        energies_exists = (self.scratch_dir / self.energies_file).exists()

        outfile = self.scratch_dir / self.log_file
        terminated = outfile.exists() and self.TERMINATION_MSG in outfile.read_text(encoding="utf-8")

        if xyz_exists and energies_exists and terminated:
            return True

        reasons = []
        if not xyz_exists:
            reasons.append(f"missing {self.xyz_file}")
        if not energies_exists:
            reasons.append(f"missing {self.energies_file}")
        if not terminated:
            reasons.append(f"'{self.TERMINATION_MSG}' not found in {self.log_file}")
        print(f"     [{self.LABEL}] Output validation failed: {'; '.join(reasons)}")

        # The script exits with a reason on stderr (no conformers embedded, no
        # force-field parameters, ...), alongside any container errors.
        errfile = self.scratch_dir / self.err_file
        if errfile.exists():
            err_lines = errfile.read_text(encoding="utf-8").splitlines()
            if err_lines:
                print(f"     [{self.LABEL}] Last lines of {self.err_file}:")
                for line in err_lines[-10:]:
                    print(f"       {line}")
        return False

    def scrape_data(self) -> bool:
        """
        Keep the conformers that reproduce the state's bonding, ranked by energy.

        classy_yarp's connectivity filter: perceive bonds from each geometry and
        drop any conformer that differs from the yarpecule graph. The container
        already wrote them lowest energy first, so survivors are ranked in file
        order.
        """
        graph = self.target_species.graph
        elements, geometries = xyz_parse(self.scratch_dir / self.xyz_file, multiple=True)

        rank = 0
        for conf_elements, geo in zip(elements, geometries):
            if not compare_adjacency(graph.elements, geo, graph.adj_mat)[0]:
                continue
            conf_obj = conformer(calc_type='conf_gen', calc_data={
                'conf_rank': rank,
                'elements': conf_elements,
                'geometry': geo,
                'lot': self.config.lot,
                'software': self.SOFTWARE,
            })
            self.target_species.conformers[conf_obj.type] = conf_obj
            rank += 1

        n_dropped = len(geometries) - rank
        if n_dropped:
            print(f"     [{self.LABEL}] Dropped {n_dropped} of {len(geometries)} conformers that "
                  f"did not reproduce the {'reactant' if 'reactant' in self.task_def.task_type else 'product'} bonding.")
        if rank == 0:
            print(f"     [{self.LABEL}] No conformer reproduced the state's bonding.")
            return False

        return True

    def cleanup(self):
        """Keep inputs, outputs and logs; remove anything else in scratch."""
        keep = {self.mol_file, self.xyz_file, self.energies_file,
                self.log_file, self.err_file, self.script_file}
        for item in self.scratch_dir.iterdir():
            if item.name not in keep:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)


class Auto3dConfCalculator(RdkitConfCalculator):
    """
    Auto3D conformer generation: RDKit's embedding, optimized with a
    neural-network potential (AIMNet2, spin-aware AIMNet2, or ANI-2x).

    The container (containers/auto3d_conf/run_conf_gen.py, Auto3D 3.2.2) embeds
    and places fragments exactly as the rdkit_conf one does, then optimizes and
    ranks with Auto3D. So the lifecycle is RdkitConfCalculator's: the same MOL
    handoff in, the same connectivity filter on what comes back. The one
    addition is the spin multiplicity, passed explicitly so the conformers are
    made in the spin state the rest of the run uses, rather than one Auto3D
    infers from the Lewis structure (which draws O2 as closed-shell O=O).
    """

    TERMINATION_MSG = "Auto3D conformer generation terminated normally."
    IMAGE_NAME = "erm42/yarp:auto3d_conf"
    FILE_PREFIX = "auto3d"
    LABEL = "Auto3D"
    SOFTWARE = "auto3d"

    def _container_args(self) -> str:
        return f"{super()._container_args()} --multiplicity {self.config.multiplicity}"


def check_auto3d_compatibility(reactions, pipeline_tasks, max_listed=10):
    """
    yarp-init check for every Auto3D conformer task, run before anything is saved or submitted.

    - Elements: every state an Auto3D task will run on must contain only
      elements its potential is trained on. Otherwise raise ValueError naming
      the states, so the run stops here instead of failing job by job.
    - Spin: a state whose Lewis structure has unpaired electrons, in a run
      with conf_gen multiplicity 1, only gets a warning. Lewis structures are
      not a reliable guide to spin (YARP draws O2 as closed-shell O=O), and
      CREST and RDKit runs treat such states as singlets the same way.

    The config-only checks (open-shell needs aimnet2-nse, ANI-2x needs charge 0)
    already ran when ConformerConfig was built.
    """
    for task in pipeline_tasks.values():
        if task.task_type not in ("reactant_conformer", "product_conformer"):
            continue
        cfg = task.config
        if getattr(cfg, "software", None) != "auto3d":
            continue
        side = "reactant" if task.task_type == "reactant_conformer" else "product"
        allowed = AUTO3D_ELEMENTS[cfg.lot]

        unsupported, radical = {}, set()
        for rxn in reactions.values():
            st = getattr(rxn, side)
            bad = set(st.graph.elements) - allowed
            if bad:
                unsupported[st.canon_smi] = bad
            if cfg.multiplicity == 1 and return_radicals(st.graph):
                radical.add(st.canon_smi)

        if unsupported:
            all_bad = set().union(*unsupported.values())
            if all_bad <= _AIMNET2_ELEMENTS:
                hint = "AIMNet2 covers them: use 'lot: aimnet2' (or 'aimnet2-nse' for open-shell)."
            else:
                hint = ("No Auto3D potential covers them (AIMNet2: "
                        f"{' '.join(sorted(e.capitalize() for e in _AIMNET2_ELEMENTS))}): "
                        "use 'software: rdkit' or 'software: crest' for this run.")
            listed = "\n".join(f"  {smi}: {' '.join(sorted(e.capitalize() for e in els))}"
                                for smi, els in sorted(unsupported.items())[:max_listed])
            more = f"\n  ... and {len(unsupported) - max_listed} more" if len(unsupported) > max_listed else ""
            raise ValueError(
                f"{task.task_id}: {len(unsupported)} {side} state(s) contain elements Auto3D's "
                f"{cfg.lot} is not trained on:\n{listed}{more}\n{hint}"
            )

        if radical:
            listed = ", ".join(sorted(radical)[:max_listed])
            more = f", ... ({len(radical)} in total)" if len(radical) > max_listed else ""
            print(f"WARNING: {task.task_id}: {len(radical)} {side} state(s) have unpaired electrons in "
                  f"their Lewis structure, but conf_gen multiplicity is 1, so Auto3D will make their "
                  f"conformers as closed-shell singlets: {listed}{more}. If they are meant to be "
                  f"open-shell, set the multiplicity and 'lot: aimnet2-nse'.")
