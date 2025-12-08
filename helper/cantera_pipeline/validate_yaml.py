#!/usr/bin/env python3
"""
Lightweight YAML validator that can be imported from main_cantera.py
or run directly from the command line.
"""

import argparse
import cantera as ct


def validate_yaml(yaml_path: str, phase: str = "gas") -> bool:
    """Load the YAML into Cantera and run a tiny reactor step."""
    gas = ct.Solution(yaml_path, phase)
    # Make sure kinetics are sane by advancing a small step
    gas.TPX = 1000.0, ct.one_atm, {gas.species_name(0): 1.0}
    r = ct.IdealGasConstPressureReactor(gas, energy="off")
    ct.ReactorNet([r]).advance(1e-6)
    return True