'''
run_bc_approach.py
=============================================================================
Orchestrates the full BC pipeline for a given exercise:
  1. MLP  — train → test (FK plot)
  2. GRU  — train → test (FK plot)

The bc_dataset.csv must already exist (run build_dataset.py first).

Usage:
  py -m bc_approach.run_bc_approach --exercise 1
  py -m bc_approach.run_bc_approach --exercise 1 --runs 3
  py -m bc_approach.run_bc_approach --exercise 1 --steps 200
  py -m bc_approach.run_bc_approach --exercise 1 --mlp-only
  py -m bc_approach.run_bc_approach --exercise 1 --gru-only
'''

import argparse
import subprocess
import sys
from pathlib import Path


def run_module(module: str, args: list):
    """Runs a Python module as a subprocess with the current interpreter."""
    cmd = [sys.executable, '-m', module] + [str(a) for a in args]
    print(f'\n{"="*60}')
    print(f'Running: {" ".join(cmd)}')
    print('='*60)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f'\n[ERROR] {module} failed with exit code {result.returncode}')
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description='Run MLP, GRU and Transformer BC pipelines for one exercise.')
    parser.add_argument('--exercise', type=int, required=True,
                        help='Exercise number (e.g. 1)')
    parser.add_argument('--runs',     type=int, default=1,
                        help='Number of autoregressive runs for test plots (default: 1)')
    parser.add_argument('--steps',    type=int, default=None,
                        help='Override number of inference steps (default: from baseline/canonical)')
    parser.add_argument('--mlp-only',         action='store_true',
                        help='Run only the MLP pipeline')
    parser.add_argument('--gru-only',         action='store_true',
                        help='Run only the GRU pipeline')
    parser.add_argument('--transformer-only', action='store_true',
                        help='Run only the Transformer pipeline')
    args = parser.parse_args()

    ex          = args.exercise
    test_args   = ['--exercise', ex, '--runs', args.runs]
    if args.steps is not None:
        test_args += ['--steps', args.steps]

    run_mlp         = not args.gru_only and not args.transformer_only
    run_gru         = not args.mlp_only and not args.transformer_only
    run_transformer = not args.mlp_only and not args.gru_only

    if run_mlp:
        print('\n' + '#'*60)
        print('# MLP — TRAIN')
        print('#'*60)
        run_module('bc_approach.MLP.train_bc', [ex])

        print('\n' + '#'*60)
        print('# MLP — TEST  (offline FK plot)')
        print('#'*60)
        run_module('bc_approach.MLP.test_bc', test_args)

    if run_gru:
        print('\n' + '#'*60)
        print('# GRU — TRAIN')
        print('#'*60)
        run_module('bc_approach.GRU.train_bc', [ex])

        print('\n' + '#'*60)
        print('# GRU — TEST  (offline FK plot)')
        print('#'*60)
        run_module('bc_approach.GRU.test_bc', test_args)

    if run_transformer:
        print('\n' + '#'*60)
        print('# TRANSFORMER — TRAIN')
        print('#'*60)
        run_module('bc_approach.Transformer.train_bc', [ex])

        print('\n' + '#'*60)
        print('# TRANSFORMER — TEST  (offline FK plot)')
        print('#'*60)
        run_module('bc_approach.Transformer.test_bc', test_args)

    print('\n' + '='*60)
    print(f'BC approach complete for exercise {ex:03d}.')
    ex_name = f'exercise_{ex:03d}'
    print(f'Results in: data/dataset/{ex_name}/{{MLP,GRU,Transformer}}/')
    print(f'Plots in  : data/dataset/{ex_name}/plot/')
    print('='*60)


if __name__ == '__main__':
    main()