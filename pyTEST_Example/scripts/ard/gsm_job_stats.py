#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import csv
import glob
import os
import re

from ard_gsm.mol import MolGraph
from ard_gsm.driving_coords import DrivingCoords
from ard_gsm.util import read_xyz_file


def main():
    args = parse_args()

    gsm_dir = args.gsm_dir
    scr_dir = os.path.join(gsm_dir, 'scratch')
    num_regex = re.compile(r'\d+')

    results = []

    for gsm_log in glob.iglob(os.path.join(gsm_dir, 'gsm*.out')):
        num = int(num_regex.search(os.path.basename(gsm_log)).group(0))

        slurm_log = os.path.join(gsm_dir, f'{num}.log')
        string_file = os.path.join(gsm_dir, f'stringfile.xyz{num:04}')
        isomers_file = os.path.join(scr_dir, f'ISOMERS{num:04}')
        ts_file = os.path.join(scr_dir, f'tsq{num:04}.xyz')

        try:
            (niter,
             ngrad,
             overlap,
             ts_type,
             scf_error,
             too_many_nodes,
             high_energy,
             geometry_error,
             dissociative,
             bad_spacings,
             growth_limit) = get_gsm_stats(gsm_log)
        except:
            print(f'Error in {gsm_log}:')
            raise
        time_limit, bus_error = check_for_slurm_error(slurm_log)

        error = None
        if time_limit:
            error = 'time'
        elif bus_error:
            error = 'bus'
        elif too_many_nodes:
            error = 'nodes'
        elif high_energy:
            error = 'highE'
        elif geometry_error:
            error = 'geometry'
        elif dissociative:
            error = 'dissociative'
        elif scf_error:
            error = 'scf'
        elif bad_spacings:
            error = 'spacing'
        elif growth_limit:
            error = 'growth_limit'
        # ##### Temporary #####
        elif ts_type != '-FL-' and not os.path.exists(ts_file):
            raise Exception(f'Other error in {gsm_log}!')
        #####

        if ts_type in {'-XTS-', '-TS-'} and os.path.exists(string_file):
            xyzs = read_xyz_file(string_file, with_energy=True)
            energies = [xyz[2] for xyz in xyzs]
            barrier = max(energies[1:-1]) - energies[0]
            rxn_energy = energies[-1] - energies[0]
            # If DFT fails during final geometry optimization then the final node might have an invalid energy
            if rxn_energy >= barrier:
                rxn_energy = energies[-2] - energies[0]
                if rxn_energy >= barrier:
                    print(f'Warning: Ignored {gsm_log} because of invalid reaction energy')
                    continue
            intended = check_bond_changes(isomers_file, xyzs)
            stats = Stats(num=num,
                          niter=niter,
                          ngrad=ngrad,
                          error=error,
                          ts_type=ts_type,
                          barrier=barrier,
                          rxn_energy=rxn_energy,
                          overlap=overlap,
                          intended=intended)
            results.append(stats)
        else:
            stats = Stats(num=num,
                          niter=niter,
                          ngrad=ngrad,
                          error=error,
                          ts_type=ts_type)
            results.append(stats)

    with open(args.out_file, 'w') as csvfile:
        stats_writer = csv.writer(csvfile, quoting=csv.QUOTE_NONNUMERIC)
        header = ['num', 'niter', 'ngrad', 'error', 'ts_type', 'barrier', 'rxn_energy', 'overlap', 'intended']
        stats_writer.writerow(header)

        results.sort(key=lambda s: s.num)
        for stats in results:
            stats_writer.writerow([getattr(stats, name, None) for name in header])

    # Print summary statistics (ignoring bus errors)
    ngrads = [stats.ngrad for stats in results if stats.error != 'bus']
    ngrads_success = [stats.ngrad for stats in results if stats.ts_type in {'-XTS-', '-TS-'}]
    avg_grad = sum(ngrads) / len(ngrads)
    max_grad = max(ngrads)
    avg_grad_success = sum(ngrads_success) / len(ngrads_success)
    max_grad_success = max(ngrads_success)
    frac_success = len(ngrads_success) / len(ngrads)
    frac_intended = (sum(1 for stats in results if stats.ts_type in {'-XTS-', '-TS-'} and stats.intended)
                     / len(ngrads_success))
    print(f'Average number of gradients: {avg_grad:.0f}')
    print(f'Maximum number of gradients: {max_grad}')
    print(f'Average number of gradients in successful jobs: {avg_grad_success:.0f}')
    print(f'Maximum number of gradients in successful jobs: {max_grad_success}')
    print(f'Fraction of jobs that succeeded: {frac_success:.4f}')
    print(f'Fraction of successful jobs that were intended: {frac_intended:.4f}')
    print('Lowest barriers:')
    results_with_barrier = [stats for stats in results if hasattr(stats, 'barrier')]
    results_with_barrier.sort(key=lambda s: s.barrier)
    for i in range(min(10, len(results_with_barrier))):
        print(f'{results_with_barrier[i].num}: {results_with_barrier[i].barrier:.2f}')


class Stats(object):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def get_gsm_stats(gsm_log):
    """
    Returns a tuple of the number of iterations, number of gradients, overlap
    between TS and Hessian vector (None if not found), type of TS/reaction path
    (-XTS-, -TS-, -multistep-, -FL-, '-diss growth', None if not found),
    whether an SCF error occurred, whether there are too many nodes, whether
    the maximum reaction energy was exceeded, whether a geometry error
    occurred, and whether the profile is dissociative.
    """
    niter = 0
    ngrad = 0
    overlap = None
    ts_type = None
    scf_error = False
    too_many_nodes = False
    high_energy = False
    geometry_error = False
    dissociative = False
    bad_spacings = False
    growth_limit = False

    with open(gsm_log) as f:
        for line in f:
            if 'totalgrad' in line:
                if 'tgrads:' in line:
                    line_split = line.split()
                    idx = line_split.index('tgrads:')
                    ngrad = int(line_split[idx + 1])
                if 'opt_iter:' in line:
                    niter += 1
                elif 'opt_iters over:' in line:
                    line_split = line.split()
                    for ol_num in range(100):
                        try:
                            ol_idx = line_split.index(f'ol({ol_num}):')
                        except ValueError as e:
                            ol_error = e
                            continue
                        else:
                            break
                    else:
                        raise ol_error
                    overlap = float(line_split[ol_idx+1])
                    ts_type = line_split[-1]
                    if ts_type == 'growth-':
                        ts_type = '-diss growth-'
                    # Handle cases where nothing was printed at the end of the line
                    if len(ts_type) == 1:
                        ts_type = None
            elif 'SCF failed' in line:
                scf_error = True
            elif 'cannot add node' in line:
                too_many_nodes = True
            elif 'high energy' in line and '-exit early-' in line:
                high_energy = True
            elif 'ERROR: Geometry contains NaN' in line:
                geometry_error = True
            elif 'terminating due to dissociation' in line:
                dissociative = True
            elif 'ERROR: bad spacings' in line:
                bad_spacings = True
            elif 'at limit of growth' in line:
                growth_limit = True

    return (niter, ngrad, overlap, ts_type,
            scf_error, too_many_nodes, high_energy, geometry_error, dissociative, bad_spacings, growth_limit)


def check_for_slurm_error(slurm_log):
    """
    Returns a tuple of flags indicating whether SLURM reached the time limit or
    encountered a bus error.
    """
    time_limit = bus_error = False

    with open(slurm_log) as f:
        for line in reversed(f.readlines()):
            if 'TIME LIMIT' in line:
                time_limit = True
            elif 'Bus error' in line:
                bus_error = True

    return time_limit, bus_error


def check_bond_changes(isomers_file, string):
    """
    Check if desired bond changes were obtained.
    """
    with open(isomers_file) as f:
        intended_changes = DrivingCoords()
        intended_changes.reconstruct_from_str(f.read())

    r_symbols, r_coords = string[0][:2]
    p_symbols, p_coords = string[-1][:2]
    reactant = MolGraph(symbols=r_symbols, coords=r_coords)
    product = MolGraph(symbols=p_symbols, coords=p_coords)
    reactant.infer_connections()
    product.infer_connections()

    # Extract connection changes
    r_connections = reactant.get_all_connections()
    p_connections = product.get_all_connections()
    break_idxs, form_idxs = [], []
    for connection in r_connections:
        if connection not in p_connections:
            idxs = (connection.atom1.idx, connection.atom2.idx)
            break_idxs.append(idxs)
    for connection in p_connections:
        if connection not in r_connections:
            idxs = (connection.atom1.idx, connection.atom2.idx)
            form_idxs.append(idxs)
    actual_changes = DrivingCoords(break_idxs=break_idxs, form_idxs=form_idxs)

    if intended_changes.is_subset(actual_changes):
        return True
    else:
        return False


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('gsm_dir', metavar='DIR', help='Path to GSM directory')
    parser.add_argument('out_file', metavar='FILE', help='Path to output file')
    return parser.parse_args()


if __name__ == '__main__':
    main()
