#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import glob
import os
import re

from ard_gsm.extract import qchem2molgraph, parse_reaction, remove_duplicates
from ard_gsm.qchem import QChem
from ard_gsm.util import iter_sub_dirs


def main():
    args = parse_args()
    num_regex = re.compile(r'\d+')
    maxnum = float('inf') if args.maxnum is None else args.maxnum

    for ts_sub_dir in iter_sub_dirs(args.ts_dir, pattern=r'gsm\d+'):
        sub_dir_name = os.path.basename(ts_sub_dir)
        gsm_num = int(num_regex.search(sub_dir_name).group(0))
        if gsm_num > maxnum:
            continue

        ts_sub_out_dir = os.path.join(args.ts_out_dir, sub_dir_name)
        if os.path.exists(ts_sub_out_dir) and not args.overwrite:
            continue

        print(f'Extracting from {sub_dir_name}...')

        reactant_num = int(num_regex.search(sub_dir_name).group(0))
        reactant_file = os.path.join(args.reac_dir, f'molopt{reactant_num}.log')
        reactant, qr = qchem2molgraph(reactant_file, return_qobj=True, freq_only=True, print_msg=False)
        if reactant is None:
            raise Exception(f'Negative frequency for reactant in {reactant_file}!')
        charge, multiplicity = qr.get_charge(), qr.get_multiplicity()

        reactions = {}
        for ts_file in glob.iglob(os.path.join(ts_sub_dir, 'ts_optfreq*.out')):
            num = int(num_regex.search(os.path.basename(ts_file)).group(0))
            prod_file = os.path.join(args.prod_dir, sub_dir_name, f'prod_optfreq{num:04}.out')

            rxn = parse_reaction(
                reactant,
                prod_file,
                ts_file,
                keep_isomorphic=args.keep_isomorphic_reactions,
                edist_max=args.edist,
                gdist_max=args.gdist,
                normal_mode_check=args.check_normal_mode,
                soft_check=args.soft_check
            )
            if rxn is not None:
                rxn.reactant_file = reactant_file
                reactions[num] = rxn

        reactions = remove_duplicates(
            reactions,
            ndup=args.ndup,
            group_by_connection_changes=args.group_by_connection_changes,
            set_smiles=False
        )

        prod_sub_out_dir = os.path.join(args.prod_out_dir, sub_dir_name)
        os.makedirs(ts_sub_out_dir, exist_ok=True)
        os.makedirs(prod_sub_out_dir, exist_ok=True)

        for num, rxn in reactions.items():
            ts_file = os.path.join(ts_sub_out_dir, f'ts_optfreq{num:04}.in')
            prod_file = os.path.join(prod_sub_out_dir, f'prod_optfreq{num:04}.in')
            qts = QChem(mol=rxn.ts, config_file=args.config_ts)
            qp = QChem(mol=rxn.product, config_file=args.config_prod)
            qts.make_input(ts_file, charge=charge, multiplicity=multiplicity, mem=args.mem)
            qp.make_input(prod_file, charge=charge, multiplicity=multiplicity, mem=args.mem)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('reac_dir', help='Directory containing optimized reactant structures (same level as products)')
    parser.add_argument('prod_dir', help='Directory containing optimized product structures')
    parser.add_argument('ts_dir', help='Directory containing optimized TS structures')
    parser.add_argument('prod_out_dir', help='Output directory for product jobs')
    parser.add_argument('ts_out_dir', help='Output directory for TS jobs')
    parser.add_argument('--maxnum', type=int, metavar='NUM', help='Only make jobs from GSM folders up to this number')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite input files in existing directories')
    parser.add_argument('--mem', type=int, metavar='MEM', help='Q-Chem memory')
    parser.add_argument('--ndup', type=int, default=1,
                        help='Number of duplicate reactions of the same type to extract (sorted by lowest barrier)')
    parser.add_argument('--edist', type=float, default=5.0,
                        help='Ignore TS files with energy differences (kcal/mol) larger than this')
    parser.add_argument('--gdist', type=float, default=1.0,
                        help='Ignore TS files with Cartesian RMSD (Angstrom) between first and last geometries'
                             ' larger than this')
    parser.add_argument('--check_normal_mode', action='store_true',
                        help='Perform a normal mode analysis to identify if the TS structure is correct (make sure to'
                             ' check the warnings in the normal_mode_analysis function before using this option')
    parser.add_argument('--soft_check', action='store_true',
                        help='If checking normal modes, only perform a soft check, i.e., only check that the largest'
                             ' TS variation is the largest overall')
    parser.add_argument('--group_by_connection_changes', action='store_true',
                        help='Use connection changes instead of product identities to distinguish reactions')
    parser.add_argument('--keep_isomorphic_reactions', action='store_true',
                        help='Consider reactions where the product is isomorphic with the reactant')
    parser.add_argument(
        '--config_prod',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'config', 'qchem.opt_freq_high'),
        help='Configuration file for Q-Chem product optfreq jobs'
    )
    parser.add_argument(
        '--config_ts',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'config', 'qchem.ts_opt_freq_high'),
        help='Configuration file for Q-Chem TS optfreq jobs'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
