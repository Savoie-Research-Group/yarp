"""
Tests to ensure the input parser correctly reads YAML files and raises appropriate errors
"""
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