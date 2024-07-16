#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import csv
import glob
import os
import re

from ard_gsm.mol import SanitizationError
from ard_gsm.extract import qchem2molgraph, parse_reaction, remove_duplicates, rxn2xyzfile
from ard_gsm.util import iter_sub_dirs


def main():
    args = parse_args()
    num_regex = re.compile(r'\d+')
    out_file = open(args.out_file, 'w')

    if args.xyz_dir is not None:
        if not os.path.exists(args.xyz_dir):
            os.makedirs(args.xyz_dir)

    writer = csv.writer(out_file)
    header = ['rsmi', 'psmi', 'ea', 'dh']
    if args.write_file_info:
        header.extend(['rfile', 'pfile', 'tsfile'])
    writer.writerow(header)

    if args.all_ts:
        # This will not filter any reactions other than incorrect frequencies
        parsing_settings = dict(
            keep_isomorphic=True,
            edist_max=float('inf'),
            gdist_max=float('inf'),
            normal_mode_check=False,
            soft_check=False,
            negative_barrier_check=False
        )
    else:
        parsing_settings = dict(
            keep_isomorphic=args.keep_isomorphic_reactions,
            edist_max=args.edist,
            gdist_max=args.gdist,
            normal_mode_check=args.check_normal_mode,
            soft_check=args.soft_check,
            negative_barrier_check=True
        )

    rxn_num = 0
    for ts_sub_dir in iter_sub_dirs(args.ts_dir, pattern=r'gsm\d+'):
        sub_dir_name = os.path.basename(ts_sub_dir)
        print(f'Extracting from {sub_dir_name}...')
        reactant_num = int(num_regex.search(sub_dir_name).group(0))
        reactant_file = os.path.abspath(os.path.join(args.reac_dir, f'molopt{reactant_num}.log'))

        reactant, qr = qchem2molgraph(reactant_file, freq_only=True, print_msg=False, return_qobj=True)
        if reactant is None:
            raise Exception(f'Negative frequency for reactant in {reactant_file}!')

        if args.reactant_smiles_from_comment:
            reactant_smiles = qr.get_comment()
            if args.atommap:
                try:
                    reactant_smiles = reactant.assign_atom_map_numbers_to_smiles(reactant_smiles)
                except SanitizationError:
                    print(f'Error during Smiles conversion in {reactant_file}')
                    raise
        else:
            try:
                reactant_smiles = reactant.perceive_smiles(atommap=args.atommap)
            except SanitizationError:
                print(f'Error during Smiles conversion in {reactant_file}')
                raise

        reactions = {}
        for ts_file in glob.iglob(os.path.join(ts_sub_dir, 'ts_optfreq*.out')):
            ts_file = os.path.abspath(ts_file)
            num = int(num_regex.search(os.path.basename(ts_file)).group(0))
            prod_file = os.path.abspath(os.path.join(args.prod_dir, sub_dir_name, f'prod_optfreq{num:04}.out'))

            rxn = parse_reaction(
                reactant,
                prod_file,
                ts_file,
                **parsing_settings
            )
            if rxn is not None:
                rxn.reactant_smiles = reactant_smiles
                rxn.reactant_file = reactant_file
                reactions[num] = rxn

        if not args.all_ts:
            reactions = remove_duplicates(
                reactions,
                group_by_connection_changes=args.group_by_connection_changes,
                atommap=args.atommap
            )

        for num, rxn in reactions.items():
            row = [rxn.reactant_smiles, rxn.product_smiles, rxn.barrier, rxn.enthalpy]
            if args.write_file_info:
                row.extend([rxn.reactant_file, rxn.product_file, rxn.ts_file])
            writer.writerow(row)
            if args.xyz_dir is not None:
                path = os.path.join(args.xyz_dir, f'rxn{rxn_num:06}.xyz')
                rxn2xyzfile(rxn, path)
            rxn_num += 1

            if args.include_reverse:
                # For reverse reactions, it's technically possible that some of
                # them are the same as already extracted reactions in a different
                # sub dir, but it's unlikely
                rxn = rxn.reverse()
                row = [rxn.reactant_smiles, rxn.product_smiles, rxn.barrier, rxn.enthalpy]
                if args.write_file_info:
                    row.extend([rxn.reactant_file, rxn.product_file, rxn.ts_file])
                writer.writerow(row)
                if args.xyz_dir is not None:
                    path = os.path.join(args.xyz_dir, f'rxn{rxn_num:06}.xyz')
                    rxn2xyzfile(rxn, path)
                rxn_num += 1

    print(f'Wrote {rxn_num} reactions to {args.out_file}.')
    out_file.close()


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('reac_dir', help='Path to directory containing optimized reactant structures')
    parser.add_argument('prod_dir', help='Path to directory containing optimized product structures')
    parser.add_argument('ts_dir', help='Path to directory containing optimized TS structures')
    parser.add_argument('out_file', help='Path to output file')
    parser.add_argument('--all_ts', action='store_true',
                        help='Do not filter reactions or remove duplicates and store only TS file info')
    parser.add_argument('--reactant_smiles_from_comment', action='store_true',
                        help='Use the SMILES in the reactant geometry optimization comment instead of inferring it')
    parser.add_argument('--xyz_dir', help='If specified, write the geometries for each reaction to this directory')
    parser.add_argument('--include_reverse', action='store_true', help='Also extract reverse reactions')
    parser.add_argument('--write_file_info', action='store_true', help='Write file paths to output file')
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
    parser.add_argument('--no_atommap', action='store_false', dest='atommap',
                        help='Do not include atom mapping in parsed SMILES')
    return parser.parse_args()


if __name__ == '__main__':
    main()
