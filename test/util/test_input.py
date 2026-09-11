"""
Tests to ensure the input parser correctly reads YAML files and raises appropriate errors
"""
import copy
import pytest
from yarp.util.input import InputParser

class TestInvalid:
    def test_no_initial_struct(self, no_initial_struct):
        with pytest.raises(ValueError) as exc_info:
            InputParser(no_initial_struct)
        assert str(exc_info.value) == "Missing required block! 'initial_structure' must be provided!"

    def test_species_noenum(self, species_noenum):
        with pytest.raises(ValueError) as exc_info:
            InputParser(species_noenum)
        assert str(exc_info.value) == "Invalid input configuration! Enumeration must be turned on if starting from a 'species' rather than a 'reaction'!"

    def test_slurm_no_queue(self, slurm_no_queue):
        with pytest.raises(ValueError) as exc_info:
            InputParser(slurm_no_queue)
        assert str(exc_info.value) == "Sanity Check Failed: A 'queue' must be specified when using the 'slurm' scheduler."

    def test_sge_no_queue(self, sge_no_queue):
        with pytest.raises(ValueError) as exc_info:
            InputParser(sge_no_queue)
        assert str(exc_info.value) == "Sanity Check Failed: A 'queue' must be specified when using the 'sge' scheduler."

class TestEnumHappyPath:
    def test_enum_full_options(self, enum_full_options):
        inp = InputParser(enum_full_options)
        expected_attrs = [
            "out_file",
            "status_file",
            "verbose",
            "init_struct",
            "job_manager",
            "enum"
        ]
        assert all(hasattr(inp, name) for name in expected_attrs)


def _block(cfg, path):
    """Walk a dotted block path (e.g. 'initialize.job_manager') down to its dict."""
    node = cfg
    for part in path.split("."):
        node = node[part]
    return node


# Each of these keys parses cleanly and is silently discarded by the pre-strict
# parser, which is the whole defect: the run starts and the setting you thought
# you changed was never read. `intended` is the key difflib should suggest.
TYPO_CASES = [
    ("initialize",                               "outpt",           "output"),
    ("initialize.initial_structure",             "sorce",           "source"),
    ("initialize.job_manager",                   "schedular",       "scheduler"),
    ("initialize.enumeration",                   "n_brake",         "n_break"),
    ("initialize.enumeration.pre_enum_filters",  "separate_prod",   "separate_prods"),
    ("initialize.enumeration.post_enum_filters", "lewis_scor",      "lewis_score"),
    ("egat",                                     "n_cpu",           "n_cpus"),
    ("ll_path",                                  "conf_genn",       "conf_gen"),
    ("ll_path",                                  "pre_opts",        "pre_opt"),
    ("ll_path.conf_gen",                         "n_cpu",           "n_cpus"),
    ("ll_path.ts_guess",                         "n_confs",         "n_conf"),
    ("ll_refine.initial_geom",                   "transiton_state", "transition_state"),
    ("ll_refine.initial_geom.reactant",          "lott",            "lot"),
    ("ll_refine.rp_opt",                         "max_cycle",       "max_cycles"),
    ("ll_refine.ts_opt",                         "conv_threshold",  "conv_thresh"),
    ("ll_refine.irc_val",                        "conv_thresholds", "conv_thresh"),
]


class TestUnknownKeyRejection:
    """
    Every config block used to filter its keys with
    `{k: v for k, v in ... if k in X.__dataclass_fields__}`, so a misspelled key
    parsed cleanly and the setting quietly kept its default.

    `enum_egat_llpath_llrefine` is the fixture here because it is the only one
    exercising all three stage methods, so one valid config reaches every block.
    """

    @pytest.mark.parametrize("block_path, typo, intended", TYPO_CASES)
    def test_typo_is_rejected(self, enum_egat_llpath_llrefine, block_path, typo, intended):
        cfg = copy.deepcopy(enum_egat_llpath_llrefine)
        _block(cfg, block_path)[typo] = "some_value"

        with pytest.raises(ValueError) as exc_info:
            InputParser(cfg)

        assert typo in str(exc_info.value)

    @pytest.mark.parametrize("block_path, typo, intended", TYPO_CASES)
    def test_typo_suggests_intended_key(self, enum_egat_llpath_llrefine, block_path, typo, intended):
        cfg = copy.deepcopy(enum_egat_llpath_llrefine)
        _block(cfg, block_path)[typo] = "some_value"

        with pytest.raises(ValueError) as exc_info:
            InputParser(cfg)

        assert f"Did you mean '{intended}'?" in str(exc_info.value)

    def test_error_names_the_block(self, enum_egat_llpath_llrefine):
        cfg = copy.deepcopy(enum_egat_llpath_llrefine)
        cfg["ll_refine"]["irc_val"]["conv_thresholds"] = "gau"

        with pytest.raises(ValueError) as exc_info:
            InputParser(cfg)

        assert "ll_refine.irc_val" in str(exc_info.value)

    def test_all_typos_in_a_block_reported_at_once(self, enum_egat_llpath_llrefine):
        """One edit should be enough to fix a block, not one edit per typo."""
        cfg = copy.deepcopy(enum_egat_llpath_llrefine)
        cfg["initialize"]["job_manager"]["schedular"] = "local"
        cfg["initialize"]["job_manager"]["kontainer"] = "docker"

        with pytest.raises(ValueError) as exc_info:
            InputParser(cfg)

        message = str(exc_info.value)
        assert "schedular" in message and "kontainer" in message

    def test_unrecognized_stage_method_is_rejected(self, enum_egat_llpath_llrefine):
        """
        An unknown method used to fall through the if/elif chain in
        `_parse_stage` and return a StageConfig with zero tasks, so a typo'd
        method produced a stage that silently did nothing.
        """
        cfg = copy.deepcopy(enum_egat_llpath_llrefine)
        cfg["ll_path"]["method"] = "init_rxn_paths"

        with pytest.raises(ValueError) as exc_info:
            InputParser(cfg)

        message = str(exc_info.value)
        assert "init_rxn_paths" in message
        assert "Did you mean 'init_rxn_path'?" in message

    def test_property_filter_typo_is_rejected(self, enum_egat_llpath_llrefine):
        cfg = copy.deepcopy(enum_egat_llpath_llrefine)
        cfg["initialize"]["enumeration"]["pre_enum_filters"]["property_filter"] = {
            "type": "barrier", "source": "egat_rgd1", "treshold": 100.0,
        }

        with pytest.raises(ValueError) as exc_info:
            InputParser(cfg)

        assert "Did you mean 'threshold'?" in str(exc_info.value)

    def test_product_blinders_typo_is_rejected(self, enum_egat_llpath_llrefine):
        cfg = copy.deepcopy(enum_egat_llpath_llrefine)
        cfg["initialize"]["enumeration"]["pre_enum_filters"]["product_blinders"] = {
            "target_product": "CCO", "distance_metrik": "soergel",
        }

        with pytest.raises(ValueError) as exc_info:
            InputParser(cfg)

        assert "Did you mean 'distance_metric'?" in str(exc_info.value)

    def test_pre_characterize_filters_typo_is_rejected(self, enum_egat_llpath_llrefine):
        cfg = copy.deepcopy(enum_egat_llpath_llrefine)
        cfg["ll_path"]["pre_characterize_filters"] = {
            "type": "barrier", "source": "egat_rgd1", "threshold": 100.0, "sorce": "oops",
        }

        with pytest.raises(ValueError) as exc_info:
            InputParser(cfg)

        assert "sorce" in str(exc_info.value)


class TestPreOptWiring:
    """
    The xTB pre-optimization sits between the ML stage and conformer
    generation. Its two legs are NOT siblings: the product leg starts from a
    UFF patch of the *relaxed reactant*, so it cannot begin until the reactant
    leg has finished.
    """

    def _tasks(self, cfg):
        return InputParser(cfg).pipeline_tasks

    def test_both_legs_exist(self, enum_egat_llpath_llrefine):
        tasks = self._tasks(enum_egat_llpath_llrefine)

        assert tasks["ll_path.reactant_pre_opt"].task_type == "reactant_pre_opt"
        assert tasks["ll_path.product_pre_opt"].task_type == "product_pre_opt"

    def test_product_leg_waits_for_the_reactant_leg(self, enum_egat_llpath_llrefine):
        tasks = self._tasks(enum_egat_llpath_llrefine)

        assert tasks["ll_path.product_pre_opt"].depends_on == ["ll_path.reactant_pre_opt"]

    def test_conformers_wait_for_their_own_leg(self, enum_egat_llpath_llrefine):
        tasks = self._tasks(enum_egat_llpath_llrefine)

        assert tasks["ll_path.reactant_conformer"].depends_on == ["ll_path.reactant_pre_opt"]
        assert tasks["ll_path.product_conformer"].depends_on == ["ll_path.product_pre_opt"]

    def test_reactant_leg_still_waits_for_the_ml_stage(self, enum_egat_llpath_llrefine):
        tasks = self._tasks(enum_egat_llpath_llrefine)

        assert tasks["ll_path.reactant_pre_opt"].depends_on == ["egat.ml_predict"]

    def test_ts_guess_is_unchanged(self, enum_egat_llpath_llrefine):
        """Pre-opt is inserted upstream; the TS guess still hangs off the conformers."""
        tasks = self._tasks(enum_egat_llpath_llrefine)

        assert tasks["ll_path.ts_guess"].depends_on == [
            "ll_path.reactant_conformer", "ll_path.product_conformer",
        ]

    def test_absent_block_runs_with_defaults(self, enum_egat_llpath_llrefine):
        """
        There is deliberately no way to switch the pre-optimization off, so an
        absent block means 'run with defaults', not 'skip'.
        """
        assert "pre_opt" not in enum_egat_llpath_llrefine["ll_path"]
        cfg = self._tasks(enum_egat_llpath_llrefine)["ll_path.reactant_pre_opt"].config

        assert cfg.software == "pysisyphus"
        assert cfg.lot == "xtb"
        assert cfg.opt_type == "lbfgs"

    def test_block_customises_the_stage(self, enum_egat_llpath_llrefine):
        cfg = copy.deepcopy(enum_egat_llpath_llrefine)
        cfg["ll_path"]["pre_opt"] = {"n_cpus": 4, "max_cycles": 50, "opt_type": "rfo"}
        parsed = self._tasks(cfg)["ll_path.reactant_pre_opt"].config

        assert (parsed.n_cpus, parsed.max_cycles, parsed.opt_type) == (4, 50, "rfo")

    def test_both_legs_share_one_config_object(self, enum_egat_llpath_llrefine):
        tasks = self._tasks(enum_egat_llpath_llrefine)

        assert tasks["ll_path.reactant_pre_opt"].config is tasks["ll_path.product_pre_opt"].config

    def test_optimizer_default_is_lbfgs_not_rfo(self, enum_egat_llpath_llrefine):
        """
        'rfo' cannot optimize a free diatomic, and products shedding H2 or O2
        are common, so the pre-opt default has to be lbfgs. The refine stage
        keeps rfo.
        """
        tasks = self._tasks(enum_egat_llpath_llrefine)

        assert tasks["ll_path.reactant_pre_opt"].config.opt_type == "lbfgs"
        assert tasks["ll_refine.reactant_optimization"].config.opt_type == "rfo"


class TestNoDuplicateDependencies:
    """
    Static `depends_on` and the `requires_data` ledger both appended, so
    `ts_guess` used to list both conformer tasks twice and `irc_validation`
    listed `ts_opt` twice.
    """

    @pytest.mark.parametrize("task_id", [
        "ll_path.ts_guess",
        "ll_refine.irc_validation",
        "ll_refine.reactant_optimization",
        "ll_refine.product_optimization",
        "ll_refine.transition_state_optimization",
    ])
    def test_dependencies_are_unique(self, enum_egat_llpath_llrefine, task_id):
        deps = InputParser(enum_egat_llpath_llrefine).pipeline_tasks[task_id].depends_on

        assert len(deps) == len(set(deps)), f"{task_id} has duplicate entries: {deps}"

    def test_ledger_still_links_what_it_should(self, enum_egat_llpath_llrefine):
        """Deduplicating must not drop a dependency that only the ledger knows about."""
        deps = InputParser(enum_egat_llpath_llrefine).pipeline_tasks["ll_refine.irc_validation"].depends_on

        assert set(deps) == {
            "ll_refine.transition_state_optimization",
            "ll_refine.reactant_optimization",
            "ll_refine.product_optimization",
        }


class TestMLPropDefaults:
    """
    `ml_rxn_prop` carries its settings flat in the stage body, so it used to be
    built from explicit `data.get()` calls instead of going through the shared
    construction path. Its `n_cpus` fallback there was 1, while the dataclass
    default is 8 and `__post_init__` rejects anything below 8 -- so omitting
    `n_cpus` failed while quoting a number the user never wrote.
    """

    def _ml_config(self, cfg):
        return InputParser(cfg).global_tasks["egat.ml_predict"].config

    def test_omitted_n_cpus_defaults_to_eight(self, enum_egat_llpath_llrefine):
        cfg = copy.deepcopy(enum_egat_llpath_llrefine)
        cfg["egat"].pop("n_cpus")

        assert self._ml_config(cfg).n_cpus == 8

    def test_explicit_n_cpus_is_honored(self, enum_egat_llpath_llrefine):
        cfg = copy.deepcopy(enum_egat_llpath_llrefine)
        cfg["egat"]["n_cpus"] = 16

        assert self._ml_config(cfg).n_cpus == 16

    def test_n_cpus_below_minimum_still_rejected(self, enum_egat_llpath_llrefine):
        """The floor still applies, and the error quotes what the user wrote."""
        cfg = copy.deepcopy(enum_egat_llpath_llrefine)
        cfg["egat"]["n_cpus"] = 4

        with pytest.raises(ValueError) as exc_info:
            InputParser(cfg)

        assert "Number selected: 4" in str(exc_info.value)

    def test_missing_model_keeps_its_readable_message(self, enum_egat_llpath_llrefine):
        """
        `model` carries a `None` default so an omitted value reaches the
        `__post_init__` check. Without it the dataclass raises a bare TypeError
        about a missing positional argument, which is no help to a user.
        """
        cfg = copy.deepcopy(enum_egat_llpath_llrefine)
        cfg["egat"].pop("model")

        with pytest.raises(ValueError) as exc_info:
            InputParser(cfg)

        assert str(exc_info.value) == (
            "Missing required key! Please provide 'model' when using ml_rxn_prop method!"
        )

    def test_other_defaults_unchanged(self, enum_egat_llpath_llrefine):
        cfg = copy.deepcopy(enum_egat_llpath_llrefine)
        cfg["egat"].pop("mem_per_cpu")
        ml = self._ml_config(cfg)

        assert ml.mem_per_cpu == 1000
        assert ml.max_runtime == "01:00:00"


class TestStrictnessDoesNotOverreach:
    """Guards against the strict check rejecting things it should accept."""

    def test_valid_multi_stage_config_still_parses(self, enum_egat_llpath_llrefine):
        inp = InputParser(enum_egat_llpath_llrefine)
        assert inp.stage_names == ["egat", "ll_path", "ll_refine"]
        assert "ll_path.ts_guess" in inp.pipeline_tasks
        assert "ll_refine.irc_validation" in inp.pipeline_tasks

    def test_unused_stage_block_is_still_allowed(self, enum_egat_llpath_llrefine):
        """
        Top-level keys stay permissive on purpose: keeping a stage block around
        that is not listed in `stages:` is a legitimate workflow.
        """
        cfg = copy.deepcopy(enum_egat_llpath_llrefine)
        cfg["hl_refine"] = {"method": "refine_rxn_path"}

        InputParser(cfg)  # must not raise

    def test_parser_owned_keys_are_still_accepted(self, enum_egat_llpath_llrefine):
        """
        `ON` and `initial_geom` are parser-owned fields that a user-written
        value never survives -- both are overwritten after construction. By
        decision they stay ACCEPTED rather than rejected, so this pins that
        choice: if someone later adds them to a blocklist, this test says so.
        """
        cfg = copy.deepcopy(enum_egat_llpath_llrefine)
        cfg["initialize"]["enumeration"]["ON"] = False
        cfg["ll_refine"]["rp_opt"]["initial_geom"] = {"reactant": "ignored"}

        inp = InputParser(cfg)

        # Accepted on the way in, and overwritten on the way out.
        assert inp.enum.ON is True
        assert inp.pipeline_tasks["ll_refine.reactant_optimization"].config.initial_geom.reactant.label == "conf_gen"