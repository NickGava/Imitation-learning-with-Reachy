'''
run_bc_approach.py
=============================================================================
Orchestrates the full BC pipeline for one or more exercises and one split:
  1. build_dataset.py  — builds n_XX/bc_dataset.csv
  2. MLP  — train (n_XX/MLP/) → test (n_XX/plots/, n_XX/plots_joints/)
  3. GRU
  4. Transformer

Usage:
  py run_bc_approach.py                              # prompted
  py run_bc_approach.py --exercise 1                 # single exercise, all demos
  py run_bc_approach.py --exercise 1 --n-demos 10    # split n_10
  py run_bc_approach.py --exercise 1 --n-demos 10 --runs 3
  py run_bc_approach.py --exercise a --n-demos 25    # all exercises, split n_25
  py run_bc_approach.py --mlp-only
'''

import argparse
import subprocess
import sys
import time


def _run(module: str, args: list) -> bool:
    """Runs a Python module. Returns True if successful, False otherwise."""
    cmd = [sys.executable, '-m', module] + [str(a) for a in args]
    print(f'\n{"="*60}')
    print(f'Running: {" ".join(cmd)}')
    print('='*60)
    result = subprocess.run(cmd)
    return result.returncode == 0


def _run_exercise(ex: int, runs: int, steps,
                  run_mlp: bool, run_gru: bool, run_transformer: bool,
                  n_demos: int = 55) -> bool:
    """
    Runs build_dataset + selected model pipelines for a single exercise and split.
    """
    test_args = ['--exercise', ex, '--runs', runs, '--n-demos', n_demos]
    if steps is not None:
        test_args += ['--steps', steps]

    print(f'\n{"#"*60}')
    print(f'# build_dataset — exercise {ex:03d}  split n_{n_demos:02d}')
    print(f'{"#"*60}')
    if not _run('bc_approach.build_dataset', [ex, '--n-demos', n_demos]):
        print(f'  [ERROR] build_dataset failed for exercise {ex:03d} — skipping.')
        return False

    if run_mlp:
        print(f'\n{"#"*60}\n# MLP — TRAIN\n{"#"*60}')
        if not _run('bc_approach.MLP.train_bc', [ex, '--n-demos', n_demos]):
            return False
        print(f'\n{"#"*60}\n# MLP — TEST\n{"#"*60}')
        if not _run('bc_approach.MLP.test_bc', test_args):
            return False

    if run_gru:
        print(f'\n{"#"*60}\n# GRU — TRAIN\n{"#"*60}')
        if not _run('bc_approach.GRU.train_bc', [ex, '--n-demos', n_demos]):
            return False
        print(f'\n{"#"*60}\n# GRU — TEST\n{"#"*60}')
        if not _run('bc_approach.GRU.test_bc', test_args):
            return False

    if run_transformer:
        print(f'\n{"#"*60}\n# TRANSFORMER — TRAIN\n{"#"*60}')
        if not _run('bc_approach.Transformer.train_bc', [ex, '--n-demos', n_demos]):
            return False
        print(f'\n{"#"*60}\n# TRANSFORMER — TEST\n{"#"*60}')
        if not _run('bc_approach.Transformer.test_bc', test_args):
            return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Run BC pipeline for one or all exercises.')
    parser.add_argument('--exercise', type=str, default=None,
                        help='Exercise number, or "a" for all 1-20. If omitted, you will be prompted.')
    parser.add_argument('--start',   type=int, default=1,
                        help='First exercise when running a range (default: 1).')
    parser.add_argument('--end',     type=int, default=26,
                        help='Last exercise when running a range (default: 26).')
    parser.add_argument('--runs',    type=int, default=1,
                        help='Number of autoregressive runs for test plots (default: 1)')
    parser.add_argument('--steps',   type=int, default=None,
                        help='Override number of inference steps.')
    parser.add_argument('--n-demos', type=int, default=55, choices=[10, 25, 55],
                        help='Number of demos split to use (default: 55 = all subjects).')
    parser.add_argument('--mlp-only',         action='store_true')
    parser.add_argument('--gru-only',         action='store_true')
    parser.add_argument('--transformer-only', action='store_true')
    args = parser.parse_args()

    # Resolve exercise list
    exercise_arg = args.exercise
    if exercise_arg is None:
        exercise_arg = input("Exercise number (or 'a' for all): ").strip()

    if exercise_arg.lower() == 'a':
        exercises = list(range(args.start, args.end + 1))
    else:
        try:
            exercises = [int(exercise_arg)]
        except ValueError:
            print(f"[ERROR] Invalid exercise value: '{exercise_arg}'. Use a number or 'a'.")
            sys.exit(1)

    run_mlp         = not args.gru_only and not args.transformer_only
    run_gru         = not args.mlp_only and not args.transformer_only
    run_transformer = not args.mlp_only and not args.gru_only

    # Run
    t_total = time.time()
    ok, fail = [], []

    for ex in exercises:
        print(f'\n{"="*60}')
        print(f'  Exercise {ex:03d}  |  split n_{args.n_demos:02d}')
        print(f'{"="*60}')
        if _run_exercise(ex, args.runs, args.steps,
                         run_mlp, run_gru, run_transformer,
                         n_demos=args.n_demos):
            ok.append(ex)
        else:
            fail.append(ex)

    # Summary
    elapsed = time.time() - t_total
    print(f'\n{"="*60}')
    print(f'SUMMARY  (total: {elapsed:.0f}s)  split n_{args.n_demos:02d}')
    print(f'{"="*60}')
    print(f'  OK   : {ok}')
    if fail:
        print(f'  FAIL : {fail}')
    print()


if __name__ == '__main__':
    main()