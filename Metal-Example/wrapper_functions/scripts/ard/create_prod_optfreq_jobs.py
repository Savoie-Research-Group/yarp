#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import glob
import os
import re

from ard_gsm.qchem import QChem
from ard_gsm.util import iter_sub_dirs, read_xyz_file


def main():
    args = parse_args()
    num_regex = re.compile(r'\d+')
    maxnum = float('inf') if args.maxnum is None else args.maxnum

    for gsm_sub_dir in iter_sub_dirs(args.gsm_dir, pattern=r'gsm\d+'):
        gsm_num = int(num_regex.search(os.path.basename(gsm_sub_dir)).group(0))
        if gsm_num > maxnum:
            continue

        out_dir = os.path.join(args.out_dir, os.path.basename(gsm_sub_dir))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        elif not args.overwrite:
            continue

        qstart_file = os.path.join(gsm_sub_dir, 'qstart')
        qtmp = QChem(logfile=qstart_file)
        charge, multiplicity = qtmp.get_charge(), qtmp.get_multiplicity()

        print(f'Extracting from {gsm_sub_dir}...')
        for gsm_log in glob.iglob(os.path.join(gsm_sub_dir, 'gsm*.out')):
            num = int(num_regex.search(os.path.basename(gsm_log)).group(0))
            string_file = os.path.join(gsm_sub_dir, f'stringfile.xyz{num:04}')

            if not (os.path.isfile(string_file) and os.path.getsize(string_file) > 0):
                continue
            if args.ignore_errors and has_error(gsm_log):
                continue

            if args.ignore_errors or is_successful(gsm_log):
                # Optimize van-der-Waals wells instead of separated products
                # Also check if product optimization during GSM failed
                xyzs = read_xyz_file(string_file, with_energy=True)
                last_energy = xyzs[-1][-1]
                second_to_last_energy = xyzs[-2][-1]
                if last_energy > second_to_last_energy:  # Something went wrong in product optimization
                    continue
                path = os.path.join(out_dir, f'prod_optfreq{num:04}.in')
                q = QChem(config_file=args.config)
                q.make_input_from_coords(path, *xyzs[-1][:-1], charge=charge, multiplicity=multiplicity, mem=args.mem)


def is_successful(gsm_log):
    """
    Success is defined as having converged to a transition state.
    """
    with open(gsm_log) as f:
        for line in reversed(f.readlines()):
            if '-XTS-' in line or '-TS-' in line:
                return True
    return False


def has_error(gsm_log):
    """
    Check if last node is high in energy or if the path is dissociative.
    """
    with open(gsm_log) as f:
        for line in reversed(f.readlines()):
            if 'high energy' in line and '-exit early-' in line:
                return True
            if 'terminating due to dissociation' in line:
                return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('gsm_dir', metavar='GSMDIR', help='Path to directory containing GSM folders')
    parser.add_argument('out_dir', metavar='ODIR', help='Path to output directory')
    parser.add_argument('--mem', type=int, metavar='MEM', help='Q-Chem memory')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite input files in existing directories')
    parser.add_argument('--maxnum', type=int, metavar='NUM', help='Only make jobs from GSM folders up to this number')
    parser.add_argument('--ignore_errors', action='store_true',
                        help='Extract from all GSM calculations ignoring (most) errors')
    parser.add_argument(
        '--config', metavar='FILE',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'config', 'qchem.opt_freq'),
        help='Configuration file for product optfreq jobs in Q-Chem'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
