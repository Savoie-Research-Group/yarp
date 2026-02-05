#!/usr/bin/env python3
# validate_cantera.py
import sys, cantera as ct

if len(sys.argv) != 2:
    print("Usage: python validate_cantera.py reactions.yaml", file=sys.stderr)
    raise SystemExit(2)

yaml = sys.argv[1]
try:
    gas = ct.Solution(yaml, "gas")
    print(f"✅ Loaded: species={len(gas.species())}, reactions={gas.n_reactions}")
    # Quick touch to ensure kinetics are fine:
    gas.TPX = 1000.0, ct.one_atm, {gas.species_name(0): 1.0}
    r = ct.IdealGasConstPressureReactor(gas, energy="off")
    ct.ReactorNet([r]).advance(1e-6)
    print("✅ Reactor step OK.")
except Exception as e:
    print("❌ Cantera load error:\n", e)
    raise SystemExit(1)
