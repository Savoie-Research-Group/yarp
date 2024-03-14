#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import glob
import os
import re
import shutil

from ard_gsm.qchem import QChem, QChemError, insert_into_qcinput
from ard_gsm.mol import MolGraph
from ard_gsm.driving_coords import generate_driving_coords
from config.limits import connection_limits


def main():
    args = parse_args()
    config_qchem_start = args.config_qchem
    config_qchem_end = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'config', 'qchem.gsm.end')
    config_gsm = args.config_gsm
    config_gscreate = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'config', 'gsm.gscreate')

    pdir = args.out_dir
    if not os.path.exists(pdir):
        os.makedirs(pdir)
    num_regex = re.compile(r'\d+')

    with open(os.path.join(pdir, 'params.log'), 'w') as f:
        if args.check_limits:
            f.write('Connection limits:\n')
            for symbol in connection_limits:
                ll = connection_limits[symbol][0]
                ul = connection_limits[symbol][1]
                f.write(f'  {symbol}: {ll}, {ul}\n')
        f.write(f'maxbreak = {args.maxbreak}\n')
        f.write(f'maxform = {args.maxform}\n')
        f.write(f'maxchange = {args.maxchange}\n')
        f.write(f'single_change = {not args.ignore_single_change}\n')
        f.write(f'equiv_Hs = {args.equiv_Hs}\n')
        f.write(f'minbreak = {args.minbreak}\n')
        f.write(f'minform = {args.minform}\n')
        f.write(f'minchange = {args.minchange}\n')

    with open(config_qchem_start) as f:
        config_qchem_start = f.readlines()
    if args.mem is not None:
        config_qchem_start = insert_into_qcinput(
            config_qchem_start, f'MEM_TOTAL                 {args.mem:d}\n', '$rem'
        )

    for log_idx, logfile in enumerate(glob.iglob(os.path.join(args.qlog_dir, '*.log'))):
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
        mol = MolGraph(symbols=symbols, coords=coords)
        mol.infer_connections()

        print(f'Making driving coordinates from {logfile}')
        driving_coords_set = generate_driving_coords(
            mol,
            maxbreak=args.maxbreak,
            maxform=args.maxform,
            maxchange=args.maxchange,
            single_change=not args.ignore_single_change,
            equiv_Hs=args.equiv_Hs,
            minbreak=args.minbreak,
            minform=args.minform,
            minchange=args.minchange,
            check_limits=args.check_limits
        )

        try:
            num = int(num_regex.search(os.path.basename(logfile)).group(0))
        except AttributeError:
            # Couldn't find number in filename
            num = log_idx

        out_dir = os.path.join(pdir, f'gsm{num}')
        scr_dir = os.path.join(out_dir, 'scratch')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if not os.path.exists(scr_dir):
            os.mkdir(scr_dir)

        # Use charge and multiplicity from reactant job
        config_qchem_start_tmp = insert_into_qcinput(
            config_qchem_start, f'{log.get_charge()} {log.get_multiplicity()}\n', '$molecule'
        )
        with open(os.path.join(out_dir, 'qstart'), 'w') as f:
            f.writelines(config_qchem_start_tmp)

        shutil.copy(config_qchem_end, os.path.join(out_dir, 'qend'))
        shutil.copy(config_gsm, os.path.join(out_dir, 'inpfileq'))

        gscreate_path = os.path.join(out_dir, 'gscreate')
        shutil.copy(config_gscreate, gscreate_path)
        os.chmod(gscreate_path, 0o755)  # Make executable

        for idx, driving_coords in enumerate(driving_coords_set):
            isomers_file = os.path.join(scr_dir, f'ISOMERS{idx:04}')
            initial_file = os.path.join(scr_dir, f'initial{idx:04}.xyz')
            with open(isomers_file, 'w') as f:
                f.write(str(driving_coords))
            with open(initial_file, 'w') as f:
                f.write(str(len(symbols)) + '\n')
                f.write('\n')
                for symbol, xyz in zip(symbols, coords):
                    f.write(f'{symbol}  {xyz[0]: .10f}  {xyz[1]: .10f}  {xyz[2]: .10f}\n')


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('qlog_dir', metavar='QDIR', help='Path to directory containing geometry optimization outputs')
    parser.add_argument('out_dir', metavar='ODIR', help='Path to output directory')
    parser.add_argument('--maxbreak', type=int, default=3, metavar='B', help='Maximum number of connections to break')
    parser.add_argument('--maxform', type=int, default=3, metavar='F', help='Maximum number of connections to form')
    parser.add_argument('--maxchange', type=int, default=5, metavar='C', help='Maximum number of connections to change')
    parser.add_argument('--ignore_single_change', action='store_true',
                        help='Do not consider single connection changes (e.g., nbreak=1, nform=0)')
    parser.add_argument('--consider_equivalent_hydrogens', action='store_true', dest='equiv_Hs',
                        help='Create equivalent driving coordinates for the same reaction with different but '
                             'equivalent hydrogens, i.e., hydrogens in methyl groups')
    parser.add_argument('--minbreak', type=int, default=0, metavar='B', help='Minimum number of connections to break')
    parser.add_argument('--minform', type=int, default=0, metavar='F', help='Minimum number of connections to form')
    parser.add_argument('--minchange', type=int, default=1, metavar='F', help='Minimum number of connections to change')
    parser.add_argument('--check_limits', action='store_true', help='Check valencies of expected products')
    parser.add_argument('--mem', type=int, metavar='MEM', help='Q-Chem memory')
    parser.add_argument(
        '--config_qchem', metavar='FILE',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'config', 'qchem.gsm.start'),
        help='Configuration file for Q-Chem calls in GSM'
    )
    parser.add_argument(
        '--config_gsm', metavar='FILE',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'config', 'gsm.inpfileq'),
        help='Settings for GSM calculations'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
