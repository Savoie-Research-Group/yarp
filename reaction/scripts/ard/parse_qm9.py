#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import glob
import os

from ard_gsm.mol import MolData
from ard_gsm.util import pickle_dump


def main():
    args = parse_args()

    ignore = set()
    if args.ignore_file is not None:
        with open(args.ignore_file) as f:
            for line in f:
                try:
                    idx = int(line.split()[0])
                except (IndexError, ValueError):
                    continue
                else:
                    ignore.add(idx)

    print('Parsing files...')
    files = glob.iglob(os.path.join(args.data_dir, '*.xyz'))
    data = []

    for path in files:
        d = MolData(path=path)
        if d.index in ignore:
            continue
        elif not args.fluorine and d.contains_element('F'):
            continue
        else:
            data.append(d)

    out_dir = os.path.dirname(os.path.abspath(args.out_file))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    pickle_dump(args.out_file, data, compress=True)
    print(f'Dumped {len(data)} molecules to {args.out_file}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', metavar='DIR', help='Path to 134k data directory')
    parser.add_argument('out_file', metavar='FILE', help='Path to output file')
    parser.add_argument('--ignore', metavar='FILE', dest='ignore_file',
                        help='Path to file containing list of indices to ignore. Indices should be in the first column')
    parser.add_argument('--no_fluorine', action='store_false', dest='fluorine',
                        help='Ignore molecules containing fluorine')
    return parser.parse_args()


if __name__ == '__main__':
    main()
