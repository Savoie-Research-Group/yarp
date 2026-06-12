#!/usr/bin/env python

import argparse
import glob
import os
import shutil

from ard_gsm.mol import MolGraph, SanitizationError
from ard_gsm.qchem import QChem, QChemError


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for logfile in glob.iglob(os.path.join(args.qlog_dir, '*.log')):
        try:
            log = QChem(logfile=logfile)
        except QChemError as e:
            print(e)
            continue

        try:
            freqs = log.get_frequencies()
        except QChemError as e:
            print(e)
            continue
        else:
            if any(freq < 0.0 for freq in freqs):
                print(f'Imaginary frequency in {logfile}')
                continue

        symbols, coords = log.get_geometry(first=True)
        mol_preopt = MolGraph(symbols=symbols, coords=coords)
        mol_preopt.infer_connections()
        symbols, coords = log.get_geometry()
        mol_postopt = MolGraph(symbols=symbols, coords=coords)
        mol_postopt.infer_connections()
        if not mol_postopt.has_same_connectivity(mol_preopt):
            print(f'Changed connectivity in {logfile}')
            continue

        if args.check_smiles:
            try:
                _ = mol_postopt.assign_atom_map_numbers_to_smiles(log.get_comment())
            except QChemError:
                print(f'WARNING: Missing SMILES in {logfile}')
            except SanitizationError:
                print(f'Incorrect SMILES in {logfile}')
                continue

        shutil.copy(logfile, args.out_dir)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description='Remove all geometry optimizations that had errors, '
                                                 'imaginary frequencies, or changed connectivity during optimization.')
    parser.add_argument('qlog_dir', metavar='QDIR', help='Path to directory containing geometry optimization outputs')
    parser.add_argument('out_dir', metavar='ODIR', help='Path to output directory')
    parser.add_argument('--check_smiles', action='store_true', help='Check that the SMILES from the comment matches')
    return parser.parse_args()


if __name__ == '__main__':
    main()
