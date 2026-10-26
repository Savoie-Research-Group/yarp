"""
Tests for the conformer-generation calculators.

`CrestConfCalculator` and `RdkitConfCalculator` sit between the xTB
pre-optimization and GSM. These tests pin where CREST reads its starting
geometry from -- the pre-optimized structure, never the raw yarpecule graph
geometry -- the MD timestep it requests when a fragment is a free diatomic, and
how the RDKit calculator hands its graph to the container and filters what
comes back.
"""
import numpy as np
import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock

from rdkit import Chem

from yarp.reaction.external.calc_base import CalculatorInputError
from yarp.reaction.external.calc_factory import get_calculator
from yarp.reaction.external.conf_gen import (
    CrestConfCalculator, RdkitConfCalculator, Auto3dConfCalculator, DIATOMIC_MD_TIMESTEP_FS,
    check_auto3d_compatibility,
)
from yarp.reaction.state import state
import yarp as yp
from yarp.reaction.reaction import reaction
from yarp.util.config import ConformerConfig


def _conf(geo, elements, ctype):
    return SimpleNamespace(geo=geo, elements=elements, type=ctype,
                           to_xyz_string=lambda: "stub")



# =====================================================================
# Handoff from the pre-optimization into conformer generation
# =====================================================================

class TestCrestReadsThePreOptGeometry:
    """
    CREST used to start from `initial_geom`, the raw yarpecule graph geometry.
    For an enumerated product that is the parent's coordinates under the
    product's bonding. If CREST kept reading it, the pre-optimization would run
    and then be thrown away.
    """

    def _calc(self, rxn, task_type):
        task_def = SimpleNamespace(task_type=task_type, config=MagicMock(), task_id=f"s.{task_type}")
        return CrestConfCalculator(task_def, rxn, MagicMock())

    def test_not_ready_before_the_pre_opt_runs(self, khp_reaction):
        assert not self._calc(khp_reaction, "reactant_conformer").has_prerequisites()

    def test_initial_geom_alone_is_not_enough(self, khp_reaction):
        """A fresh reaction already has initial_geom; that must no longer satisfy it."""
        assert "initial_geom" in khp_reaction.reactant.conformers

        assert not self._calc(khp_reaction, "reactant_conformer").has_prerequisites()

    def test_ready_once_its_own_side_is_pre_optimized(self, khp_reaction):
        graph = khp_reaction.reactant.graph
        khp_reaction.reactant.conformers["preopt_xtb_pysisyphus"] = _conf(
            graph.geo.copy(), graph.elements, "preopt_xtb_pysisyphus")

        assert self._calc(khp_reaction, "reactant_conformer").has_prerequisites()

    def test_reactant_does_not_wait_on_the_product_leg(self, khp_reaction):
        """
        The product pre-opt depends on the reactant pre-opt, so it finishes
        later. Requiring both sides here would fail the reactant's conformer
        task the moment its own dependency was met.
        """
        graph = khp_reaction.reactant.graph
        khp_reaction.reactant.conformers["preopt_xtb_pysisyphus"] = _conf(
            graph.geo.copy(), graph.elements, "preopt_xtb_pysisyphus")

        assert self._calc(khp_reaction, "reactant_conformer").has_prerequisites()
        assert not self._calc(khp_reaction, "product_conformer").has_prerequisites()

    def test_writes_the_pre_opt_geometry_not_the_graph_geometry(self, khp_reaction, tmp_path):
        graph = khp_reaction.reactant.graph
        relaxed = _conf(graph.geo.copy(), graph.elements, "preopt_xtb_pysisyphus")
        relaxed.to_xyz_string = lambda: "RELAXED"
        khp_reaction.reactant.conformers["preopt_xtb_pysisyphus"] = relaxed

        calc = self._calc(khp_reaction, "reactant_conformer")
        calc.set_scratch_dir(tmp_path)
        calc.generate_input()

        assert (tmp_path / "input.xyz").read_text() == "RELAXED"

    def test_missing_pre_opt_raises_rather_than_writing_nothing(self, khp_reaction, tmp_path):
        calc = self._calc(khp_reaction, "product_conformer")
        calc.set_scratch_dir(tmp_path)

        with pytest.raises(CalculatorInputError, match="pre-optimization"):
            calc.generate_input()



# =====================================================================
# CREST MD timestep for free diatomics
# =====================================================================

class TestCrestDiatomicTimestep:
    """
    CREST's default 5 fs metadynamics is only stable because SHAKE constrains
    the bonds, and SHAKE can only constrain bonds xtb perceives. An xTB-relaxed
    free H2 sits at 0.7750 A, just outside the ~0.768 A H-H perception cutoff,
    so it is never constrained and the MD diverges. Measured: every --tstep
    variant survives, every 5 fs variant fails, and --shake 1 also fails --
    confirming the issue is topology, not SHAKE mode.
    """

    def _calc(self, rxn, task_type, lot="gfn2", n_cpus=4):
        config = SimpleNamespace(lot=lot, n_cpus=n_cpus, charge=0,
                                 n_unpaired_electrons=0, seed=42, solvent=None)
        task_def = SimpleNamespace(task_type=task_type, config=config,
                                   task_id=f"s.{task_type}")
        return CrestConfCalculator(task_def, rxn, MagicMock())

    def test_noopt_is_gone(self, khp_reaction):
        """
        --noopt was the first attempt at this and does not work: the failure
        reproduces with and without it, and keeping it costs conformers.
        """
        cmd = self._calc(khp_reaction, "reactant_conformer")._get_crest_command()

        assert "--noopt" not in cmd

    def test_no_timestep_flag_without_a_diatomic(self, khp_reaction):
        """The ~70% of systems with no diatomic keep CREST's default 5 fs."""
        calc = self._calc(khp_reaction, "reactant_conformer")

        assert not calc.has_free_diatomic()
        assert "--tstep" not in calc._get_crest_command()

    def test_timestep_flag_when_a_diatomic_is_present(self, khp_parent, khp_products):
        """O=C=CCOO.[H][H] carries a free H2 -- this is the failing case."""
        product = khp_products["O=C=CCOO.[H][H]"]
        rxn = reaction(khp_parent, product)
        calc = self._calc(rxn, "product_conformer")

        assert calc.has_free_diatomic()
        assert f"--tstep {DIATOMIC_MD_TIMESTEP_FS}" in calc._get_crest_command()

    def test_the_reactant_side_of_that_reaction_is_unaffected(self, khp_parent, khp_products):
        """
        The H2 is on the product side only. Flagging the reactant too would
        pay the runtime cost for nothing.
        """
        rxn = reaction(khp_parent, khp_products["O=C=CCOO.[H][H]"])
        calc = self._calc(rxn, "reactant_conformer")

        assert not calc.has_free_diatomic()
        assert "--tstep" not in calc._get_crest_command()

    def test_timestep_is_short_enough_to_integrate_an_H2_stretch(self):
        """
        H2 stretches near 4400 cm-1, a period of ~7.6 fs. Stable integration
        wants roughly ten steps per period; 2.0 fs gives ~3.8 and was measured
        working but marginal, so the default must stay at or below 1.0.
        """
        assert DIATOMIC_MD_TIMESTEP_FS <= 1.0


class TestCrestHasNoSeed:
    """
    CREST has no seed option. YARP used to append `--seed <n>` anyway; CREST
    3.0.2 ignored it, and two runs of the same input gave different conformer
    sets for every KHP depth-1 reaction. So the flag only suggested a
    reproducibility CREST does not have.
    """

    def test_seed_is_not_passed_to_crest(self, khp_reaction):
        """Even with seed 42 in the config, the CREST command carries no --seed."""
        config = SimpleNamespace(lot="gfn2", n_cpus=4, charge=0, n_unpaired_electrons=0,
                                 seed=42, solvent=None)
        task_def = SimpleNamespace(task_type="reactant_conformer", config=config,
                                   task_id="s.reactant_conformer")
        cmd = CrestConfCalculator(task_def, khp_reaction, MagicMock())._get_crest_command()

        assert "--seed" not in cmd



# =====================================================================
# RDKit as a conf_gen software option
# =====================================================================

class TestRdkitConformerConfig:
    """
    The `conf_gen:` block of the YAML input is parsed into a ConformerConfig.
    These tests check that YARP accepts a valid RDKit block and rejects a
    mismatched one at start-up, rather than letting a bad setting reach the
    container and fail every conformer job hours into a run.
    """

    def test_rdkit_with_a_force_field_lot_is_accepted(self):
        """
        `software: rdkit, lot: uff` is a valid block, and the two RDKit-only
        settings fall back to their defaults: 50 conformers embedded, and
        embedded conformers within 0.1 A RMSD of each other pruned.
        """
        cfg = ConformerConfig(software="rdkit", lot="uff", charge=0)

        assert cfg.n_conf == 50 and cfg.prune_rms_thresh == 0.1

    def test_rdkit_does_not_need_n_unpaired_electrons(self):
        """
        CREST configs must give n_unpaired_electrons (it becomes CREST's --uhf
        flag). A force field has no electrons to count, so an RDKit block
        that leaves it out must still load.
        """
        ConformerConfig(software="rdkit", lot="mmff94", charge=0)

    def test_rdkit_rejects_a_crest_lot(self):
        """
        `lot: gfn2` is an xTB method that RDKit cannot run. Switching
        `software` to rdkit but forgetting to change `lot` must fail at
        config load, with an error that names RDKit.
        """
        with pytest.raises(ValueError, match="RDKit"):
            ConformerConfig(software="rdkit", lot="gfn2", charge=0)

    def test_crest_rejects_a_force_field_lot(self):
        """The reverse mistake: a force-field lot with `software: crest`."""
        with pytest.raises(ValueError, match="CREST"):
            ConformerConfig(software="crest", lot="uff", charge=0, n_unpaired_electrons=0)

    def test_unknown_software_is_still_rejected(self):
        """
        Adding 'rdkit' and 'auto3d' to the allowed list must not have loosened
        the check: any other name (here openbabel, not implemented) is refused.
        """
        with pytest.raises(ValueError, match="Invalid 'software'"):
            ConformerConfig(software="openbabel", lot="uff", charge=0)


def _rdkit_calc(rxn, task_type, lot="uff"):
    """An RdkitConfCalculator for one side of `rxn`, with a real ConformerConfig and a mock scheduler."""
    config = ConformerConfig(software="rdkit", lot=lot, charge=0, n_cpus=2)
    task_def = SimpleNamespace(task_type=task_type, config=config, task_id=f"s.{task_type}")
    return RdkitConfCalculator(task_def, rxn, MagicMock())


def _add_preopt(state, geo=None):
    """Store a fake xTB pre-optimized conformer on `state`, as the pre-opt task would."""
    graph = state.graph
    geo = graph.geo.copy() if geo is None else geo
    state.conformers["preopt_xtb_pysisyphus"] = _conf(geo, graph.elements, "preopt_xtb_pysisyphus")


def _write_xyz(path, elements, geometries):
    """Write a multi-frame xyz, standing in for the container's rdkit_conformers.xyz."""
    with open(path, "w") as f:
        for geo in geometries:
            f.write(f"{len(elements)}\n\n")
            for el, (x, y, z) in zip(elements, geo):
                f.write(f"{el.capitalize()} {x} {y} {z}\n")


class TestRdkitRouting:

    def test_factory_routes_rdkit_to_the_rdkit_calculator(self, khp_reaction):
        """
        The task pipeline never names a calculator class; it asks
        calc_factory.get_calculator for one based on `software`. This checks
        that `software: rdkit` gets the RDKit calculator, not CREST's.
        """
        config = ConformerConfig(software="rdkit", lot="uff", charge=0)
        task_def = SimpleNamespace(task_type="reactant_conformer", config=config, task_id="s.r")

        assert isinstance(get_calculator(task_def, khp_reaction, MagicMock()), RdkitConfCalculator)


class TestRdkitGenerateInput:
    """
    Before submitting, the calculator writes `input.mol`, the only thing the
    container sees. It is built with yarpecule_to_rdmol from the yarpecule's
    graph, so RDKit gets YARP's own bonding instead of re-perceiving it.

    Atom order matters: the conformers that come back are matched to the
    graph atom by atom (see TestRdkitScrapeData), so atom i in the MOL file
    must be atom i in the graph.
    """

    def test_mol_file_carries_the_graph_and_the_pre_opt_geometry(self, khp_reaction, tmp_path):
        """
        Give the reactant a pre-opt geometry that is the graph geometry
        shifted by 1.5 A, so the two can be told apart. Then read input.mol
        back with RDKit and check that it has:
          - the graph's elements, in the same order;
          - the graph's bonds (same adjacency matrix);
          - the shifted coordinates, i.e. the xTB pre-opt geometry, not the
            raw graph geometry (for a product, that is still the parent's).
        """
        graph = khp_reaction.reactant.graph
        shifted = graph.geo + 1.5
        _add_preopt(khp_reaction.reactant, geo=shifted)

        calc = _rdkit_calc(khp_reaction, "reactant_conformer")
        calc.set_scratch_dir(tmp_path)
        calc.generate_input()

        mol = Chem.MolFromMolFile(str(tmp_path / "input.mol"), removeHs=False)
        assert [a.GetSymbol().lower() for a in mol.GetAtoms()] == list(graph.elements)
        assert np.array_equal(Chem.GetAdjacencyMatrix(mol), graph.adj_mat)
        assert np.allclose(mol.GetConformer().GetPositions(), shifted, atol=1e-4)

    def test_missing_pre_opt_raises_rather_than_writing_nothing(self, khp_reaction, tmp_path):
        """
        With no pre-opt conformer on the product, generate_input must raise
        CalculatorInputError. progress_yarp catches that error and fails just
        this task; any other failure mode would either crash the whole pass
        or submit a job with no input file.
        """
        calc = _rdkit_calc(khp_reaction, "product_conformer")
        calc.set_scratch_dir(tmp_path)

        with pytest.raises(CalculatorInputError, match="pre-optimization"):
            calc.generate_input()

    def test_command_passes_the_config_to_the_container(self, khp_reaction, tmp_path):
        """
        Write the job script with the container prefix mocked out, and check
        the command line. Each YAML setting must reach the container's
        arguments: lot, n_conf and prune_rms_thresh from the config, n_threads
        from n_cpus (2 here), and the config's default seed of 42, which
        makes repeated runs give the same conformers.
        """
        calc = _rdkit_calc(khp_reaction, "reactant_conformer", lot="mmff94")
        calc.set_scratch_dir(tmp_path)
        calc.get_container_prefix = MagicMock(return_value="PREFIX")
        calc.write_scheduler_headers = MagicMock()

        script = calc.write_submission_script().read_text()

        # The image's entrypoint is the script, so apptainer must use `run`.
        assert calc.get_container_prefix.call_args.kwargs["apptainer_run"] is True
        assert ("PREFIX input.mol --lot mmff94 --n_conf 50 --prune_rms_thresh 0.1 "
                "--n_threads 2 --seed 42") in script


class TestRdkitScrapeData:
    """
    After the job, scrape_data reads rdkit_conformers.xyz (frames already
    sorted lowest energy first by the container) and stores the conformers on
    the state as conf_gen_rank0_uff_rdkit, conf_gen_rank1_uff_rdkit, ...

    The force-field optimization can break or form bonds. So, as classy_yarp
    did, each frame's bonding is re-perceived from its coordinates and
    compared with the yarpecule graph; a frame that does not match is
    dropped. Otherwise GSM would be given an endpoint that is a different
    molecule from the reaction it is meant to characterize.
    """

    def _squashed(self, geo):
        """Pull every atom toward the centroid until bonds form that the graph lacks."""
        centroid = geo.mean(axis=0)
        return centroid + 0.3 * (geo - centroid)

    def test_drops_broken_conformers_and_ranks_the_rest_in_order(self, khp_reaction, tmp_path):
        """
        Three frames: the graph geometry (good), a squashed copy whose
        bonding no longer matches (broken), and the graph geometry shifted
        by 2 A (good; bonding does not depend on position). Expected:
          - scrape_data reports success;
          - exactly two conformers are stored, rank0 and rank1;
          - rank1 is the third frame, so dropping a frame renumbers the
            survivors with no gap, keeping their energy order;
          - they are tagged software 'rdkit'.
        """
        graph = khp_reaction.reactant.graph
        good, broken = graph.geo, self._squashed(graph.geo)
        _write_xyz(tmp_path / "rdkit_conformers.xyz", graph.elements, [good, broken, good + 2.0])

        calc = _rdkit_calc(khp_reaction, "reactant_conformer")
        calc.set_scratch_dir(tmp_path)

        assert calc.scrape_data()
        confs = khp_reaction.reactant.conformers
        assert {k for k in confs if k.startswith("conf_gen")} == {
            "conf_gen_rank0_uff_rdkit", "conf_gen_rank1_uff_rdkit"}
        # Rank 1 is the third frame -- the broken second frame left no gap.
        assert np.allclose(confs["conf_gen_rank1_uff_rdkit"].geo, good + 2.0)
        assert confs["conf_gen_rank0_uff_rdkit"].software == "rdkit"

    def test_fails_when_no_conformer_keeps_the_bonding(self, khp_reaction, tmp_path):
        """
        If every frame is broken, scrape_data returns False, so the task is
        marked failed, and nothing is stored. A state with no conformers
        must not look like a finished conformer search.
        """
        graph = khp_reaction.reactant.graph
        _write_xyz(tmp_path / "rdkit_conformers.xyz", graph.elements, [self._squashed(graph.geo)])

        calc = _rdkit_calc(khp_reaction, "reactant_conformer")
        calc.set_scratch_dir(tmp_path)

        assert not calc.scrape_data()
        assert not any(k.startswith("conf_gen") for k in khp_reaction.reactant.conformers)



# =====================================================================
# Auto3D as a conf_gen software option
# =====================================================================

class TestAuto3dConformerConfig:
    """
    `software: auto3d` runs a neural-network potential, and each one models
    only some states. The checks the config alone can answer run at config
    load, before anything is submitted: the multiplicity must be given, an
    open-shell multiplicity needs the spin-aware AIMNet2, and ANI-2x needs a
    neutral state. (Which elements a state contains is checked at yarp-init.)
    """

    def test_closed_shell_aimnet2_is_accepted(self):
        """The common case: neutral singlet with the default AIMNet2."""
        cfg = ConformerConfig(software="auto3d", lot="aimnet2", charge=0, multiplicity=1)
        assert cfg.n_conf == 50 and cfg.prune_rms_thresh == 0.1

    def test_multiplicity_is_required(self):
        """No silent singlet: like n_unpaired_electrons for CREST, it must be given."""
        with pytest.raises(ValueError, match="multiplicity"):
            ConformerConfig(software="auto3d", lot="aimnet2", charge=0)

    def test_rejects_a_force_field_lot(self):
        """`lot: uff` left over from an RDKit block fails at load, naming Auto3D."""
        with pytest.raises(ValueError, match="Auto3D"):
            ConformerConfig(software="auto3d", lot="uff", charge=0, multiplicity=1)

    @pytest.mark.parametrize("lot", ["aimnet2", "ani2x"])
    def test_open_shell_needs_the_spin_aware_model(self, lot):
        """
        Multiplicity 3 with a closed-shell potential is refused. The default
        AIMNet2 would otherwise run it as closed-shell with only a warning
        (measured with Auto3D 3.2.2), i.e. wrong conformers
        with no error.
        """
        with pytest.raises(ValueError, match="aimnet2-nse"):
            ConformerConfig(software="auto3d", lot=lot, charge=0, multiplicity=3)

    def test_open_shell_with_the_spin_aware_model_is_accepted(self):
        ConformerConfig(software="auto3d", lot="aimnet2-nse", charge=0, multiplicity=3)

    def test_ani2x_rejects_a_charged_state(self):
        """ANI-2x is trained on neutral molecules; AIMNet2 takes a charge."""
        with pytest.raises(ValueError, match="neutral"):
            ConformerConfig(software="auto3d", lot="ani2x", charge=1, multiplicity=1)
        ConformerConfig(software="auto3d", lot="aimnet2", charge=1, multiplicity=1)


def _auto3d_calc(rxn, task_type, lot="aimnet2", multiplicity=1):
    """An Auto3dConfCalculator for one side of `rxn`, with a real ConformerConfig and a mock scheduler."""
    config = ConformerConfig(software="auto3d", lot=lot, charge=0, multiplicity=multiplicity, n_cpus=2)
    task_def = SimpleNamespace(task_type=task_type, config=config, task_id=f"s.{task_type}")
    return Auto3dConfCalculator(task_def, rxn, MagicMock())


class TestAuto3dCalculator:
    """
    Auto3dConfCalculator reuses RdkitConfCalculator's lifecycle (MOL handoff,
    output check, connectivity filter), which TestRdkit* covers. These tests
    pin what is different: the routing, the image and the command it runs,
    and the files and names its conformers come back under.
    """

    def test_factory_routes_auto3d_to_the_auto3d_calculator(self, khp_reaction):
        """Exactly Auto3D's class -- it subclasses RDKit's, so isinstance alone would not tell."""
        config = ConformerConfig(software="auto3d", lot="aimnet2", charge=0, multiplicity=1)
        task_def = SimpleNamespace(task_type="reactant_conformer", config=config, task_id="s.r")

        assert type(get_calculator(task_def, khp_reaction, MagicMock())) is Auto3dConfCalculator

    def test_command_runs_the_auto3d_image_with_the_multiplicity(self, khp_reaction, tmp_path):
        """
        Write the job script with the container prefix mocked out. It must ask
        for the auto3d_conf image (not rdkit_conf), pass the lot and the
        multiplicity on top of the RDKit-style embedding arguments, and log to
        auto3d_run.log / auto3d_run.err.
        """
        calc = _auto3d_calc(khp_reaction, "reactant_conformer", lot="aimnet2-nse", multiplicity=3)
        calc.set_scratch_dir(tmp_path)
        calc.get_container_prefix = MagicMock(return_value="PREFIX")
        calc.write_scheduler_headers = MagicMock()

        script = calc.write_submission_script()

        assert script.name == "run_auto3d_cmd.sh"
        assert calc.get_container_prefix.call_args.args[0] == "erm42/yarp:auto3d_conf"
        assert calc.get_container_prefix.call_args.kwargs["apptainer_run"] is True
        assert ("PREFIX input.mol --lot aimnet2-nse --n_conf 50 --prune_rms_thresh 0.1 "
                "--n_threads 2 --seed 42 --multiplicity 3 > auto3d_run.log 2> auto3d_run.err") in script.read_text()

    def test_output_check_needs_auto3d_files_and_termination_line(self, khp_reaction, tmp_path):
        """
        The job counts as finished only with auto3d_conformers.xyz,
        auto3d.energies and Auto3D's termination line in auto3d_run.log. The
        container exits without that line when no conformer survives ranking.
        """
        calc = _auto3d_calc(khp_reaction, "reactant_conformer")
        calc.set_scratch_dir(tmp_path)
        for name in ("auto3d_conformers.xyz", "auto3d.energies"):
            (tmp_path / name).write_text("x")
        (tmp_path / "auto3d_run.log").write_text("Embedded 50 conformers\n")
        assert not calc.check_output()

        (tmp_path / "auto3d_run.log").write_text("Auto3D conformer generation terminated normally.\n")
        assert calc.check_output()

    def test_scrape_keeps_bonded_conformers_under_auto3d_names(self, khp_reaction, tmp_path):
        """
        The same connectivity filter as RDKit's, on auto3d_conformers.xyz: a
        broken frame is dropped, the survivors are renumbered in order, and
        they are stored as conf_gen_rank<i>_<lot>_auto3d -- the key that
        ll_refine's initial_geom (lot: aimnet2-nse, software: auto3d) looks up.
        """
        graph = khp_reaction.reactant.graph
        centroid = graph.geo.mean(axis=0)
        broken = centroid + 0.3 * (graph.geo - centroid)
        _write_xyz(tmp_path / "auto3d_conformers.xyz", graph.elements, [broken, graph.geo])

        calc = _auto3d_calc(khp_reaction, "reactant_conformer", lot="aimnet2-nse", multiplicity=3)
        calc.set_scratch_dir(tmp_path)

        assert calc.scrape_data()
        confs = {k: c for k, c in khp_reaction.reactant.conformers.items() if k.startswith("conf_gen")}
        assert list(confs) == ["conf_gen_rank0_aimnet2-nse_auto3d"]
        assert confs["conf_gen_rank0_aimnet2-nse_auto3d"].software == "auto3d"
        assert np.allclose(confs["conf_gen_rank0_aimnet2-nse_auto3d"].geo, graph.geo)


def _conf_tasks(software="auto3d", lot="aimnet2", multiplicity=1):
    """The two conf_gen pipeline tasks, as InputParser builds them, sharing one ConformerConfig."""
    if software == "auto3d":
        cfg = ConformerConfig(software="auto3d", lot=lot, charge=0, multiplicity=multiplicity)
    else:
        cfg = ConformerConfig(software="crest", lot="gfn2", charge=0, n_unpaired_electrons=0)
    return {f"ll_path.{side}_conformer": SimpleNamespace(
                task_id=f"ll_path.{side}_conformer", task_type=f"{side}_conformer", config=cfg)
            for side in ("reactant", "product")}


def _reactions(*pairs):
    """{key: reaction-like object} with real reactant/product states built from SMILES."""
    return {str(i): SimpleNamespace(reactant=state(yp.yarpecule(r)), product=state(yp.yarpecule(p)))
            for i, (r, p) in enumerate(pairs)}


class TestAuto3dInitCheck:
    """
    check_auto3d_compatibility runs in yarp-init, after the reactions are built
    and before anything is saved or submitted. It checks what needs the
    species: each potential's trained elements (an error), and radical Lewis
    structures in a multiplicity-1 run (only a warning: YARP draws O2 as
    closed-shell O=O, and CREST/RDKit runs treat such states as singlets too).
    """

    def test_supported_states_pass_silently(self, capsys):
        """C/H/O chemistry with AIMNet2: no error, no warning."""
        check_auto3d_compatibility(_reactions(("O=CCCOO", "C=COCOO")), _conf_tasks())
        assert capsys.readouterr().out == ""

    def test_ani2x_stops_on_an_element_aimnet2_covers(self):
        """
        Phosphorus is in AIMNet2's 14 elements but not ANI-2x's 7. The error
        names the state and the element, and points to AIMNet2.
        """
        with pytest.raises(ValueError, match=r"(?s)CP: P.*lot: aimnet2"):
            check_auto3d_compatibility(_reactions(("O=CCCOO", "CP")), _conf_tasks(lot="ani2x"))

    def test_stops_on_an_element_no_potential_covers(self):
        """Potassium is in no Auto3D potential: the hint is to use rdkit or crest instead."""
        with pytest.raises(ValueError, match=r"(?s)\[KH\]: K\n.*software: rdkit"):
            check_auto3d_compatibility(_reactions(("O=CCCOO", "[KH]")), _conf_tasks())

    def test_other_conformer_software_is_not_checked(self):
        """The same potassium state with CREST is not Auto3D's business."""
        check_auto3d_compatibility(_reactions(("O=CCCOO", "[KH]")), _conf_tasks(software="crest"))

    def test_radical_in_a_singlet_run_warns_but_continues(self, capsys):
        """
        The ethyl radical's Lewis structure has an unpaired electron, but the
        run says multiplicity 1: a warning naming it, no error. With the
        spin-aware model and multiplicity 2 there is nothing to warn about.
        """
        rxns = _reactions(("CC", "C[CH2]"))
        check_auto3d_compatibility(rxns, _conf_tasks())
        out = capsys.readouterr().out
        assert "WARNING" in out and "[CH2]C" in out and "product" in out

        check_auto3d_compatibility(rxns, _conf_tasks(lot="aimnet2-nse", multiplicity=2))
        assert capsys.readouterr().out == ""

