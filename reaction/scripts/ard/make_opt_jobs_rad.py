#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import os

import pandas as pd
from rdkit import Chem

from ard_gsm.qchem import QChem


def main():
    args = parse_args()
    seed = 7

    print('Loading data...')
    data = pd.read_csv(args.csv)

    if args.rad_only:
        data = data[data['type'] == 'fragment']

    if args.max_heavy > 0:
        data = data[data['heavy_atoms'] <= args.max_heavy]
    if args.min_heavy > 0:
        data = data[data['heavy_atoms'] >= args.min_heavy]

    if args.num > 0:
        if args.random:
            data = data.sample(n=args.num, random_state=seed).reset_index(drop=True)
        else:
            data = data[:args.num]
    elif args.random:
        data = data.sample(frac=1, random_state=seed).reset_index(drop=True)

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    print(f'Writing {len(data)} input files...')
    for i, (smi, mol_block, mol_type) in enumerate(zip(data['smiles'], data['mol'], data['type'])):
        mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
        multiplicity = 1 if mol_type == 'molecule' else 2

        q = QChem(mol, config_file=args.config)
        q.make_input(os.path.join(args.out_dir, f'molopt{i}.in'), multiplicity=multiplicity, mem=args.mem, comment=smi)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('csv', metavar='FILE',
                        help='CSV file containing columns with SMILES, mol block, and type (molecule or fragment)')
    parser.add_argument('out_dir', metavar='DIR', help='Path to output directory')
    parser.add_argument('--max_heavy', type=int, default=-1, metavar='MAXH', help='Maximum number of heavy atoms')
    parser.add_argument('--min_heavy', type=int, default=-1, metavar='MINH', help='Minimum number of heavy atoms')
    parser.add_argument('--rad_only', action='store_true', help='Only select radicals')
    parser.add_argument('--num', type=int, default=-1, metavar='N', help='Number of molecules to choose from mol_data')
    parser.add_argument('--not_random', action='store_false', dest='random',
                        help='Select molecules in order instead of randomly')
    parser.add_argument('--mem', type=int, metavar='MEM', help='Q-Chem memory')
    parser.add_argument(
        '--config', metavar='FILE',
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'config', 'qchem.opt_freq'),
        help='Configuration file for Q-Chem input files.'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
