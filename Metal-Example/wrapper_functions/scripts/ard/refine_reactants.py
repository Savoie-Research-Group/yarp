#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import glob
import os

from ard_gsm.qchem import QChem, QChemError


def main():
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for logfile in glob.iglob(os.path.join(args.qlog_dir, '*.log')):
        try:
            log = QChem(logfile=logfile)
        except QChemError as e:
            print(e)
            continue

        # Check frequencies
        try:
            freqs = log.get_frequencies()
        except QChemError as e:
            if 'not found' in str(e):
                print(f'Warning: Frequencies could not be found in {logfile}')
            else:
                raise
        else:
            if any(freq < 0.0 for freq in freqs):
                raise Exception(f'Negative frequency in {logfile}! Not optimized')

        symbols, coords = log.get_geometry()
        charge = log.get_charge()
        mult = log.get_multiplicity()
        fname = os.path.splitext(os.path.basename(logfile))[0] + '.in'
        path = os.path.join(args.out_dir, fname)

        q = QChem(config_file=args.config)
        q.make_input_from_coords(path, symbols, coords, charge=charge, multiplicity=mult, mem=args.mem)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('qlog_dir', help='Directory containing geometry optimization outputs')
    parser.add_argument('out_dir', help='Output directory')
    parser.add_argument('--mem', type=int, metavar='MEM', help='Q-Chem memory')
    parser.add_argument(
        '--config',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'config', 'qchem.opt_freq_high'),
        help='Configuration file for Q-Chem input files'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
