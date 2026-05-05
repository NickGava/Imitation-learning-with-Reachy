'''
run_bc_all.py
=============================================================================
Runs the full BC pipeline for exercises 1 to 20:
  1. build_dataset.py  for all exercises 1-20
  2. run_bc_approach   for all exercises 1-20

Usage:
  py run_bc_all.py
  py run_bc_all.py --mlp-only
  py run_bc_all.py --gru-only
  py run_bc_all.py --start 5 --end 15
'''

import argparse
import subprocess
import sys
import time

EXERCISES = list(range(1, 21))  # 1 to 20 inclusive


def run(module: str, args: list) -> bool:
    """Runs a Python module. Returns True if successful, False otherwise."""
    cmd = [sys.executable, '-m', module] + [str(a) for a in args]
    print(f'  > {" ".join(cmd)}')
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run BC pipeline for exercises 1-20.')
    parser.add_argument('--mlp-only', action='store_true')
    parser.add_argument('--gru-only', action='store_true')
    parser.add_argument('--transformer-only', action='store_true')
    parser.add_argument('--start', type=int, default=1,  help='First exercise (default: 1)')
    parser.add_argument('--end',   type=int, default=20, help='Last exercise (default: 20)')
    args = parser.parse_args()

    exercises = list(range(args.start, args.end + 1))

    t_total   = time.time()
    ok_build  = []
    fail_build = []
    ok_bc     = []
    fail_bc   = []

    # ------------------------------------------------------------------
    # Phase 1 — build_dataset for all exercises
    # ------------------------------------------------------------------
    print('\n' + '='*60)
    print('PHASE 1 — build_dataset')
    print('='*60)

    for ex in exercises:
        print(f'\n--- Exercise {ex:03d} ---')
        ok = run('bc_approach.build_dataset', [ex])
        if ok:
            ok_build.append(ex)
        else:
            fail_build.append(ex)
            print(f'  [WARN] build_dataset failed for exercise {ex:03d} — skipping BC')

    # ------------------------------------------------------------------
    # Phase 2 — run_bc_approach for all exercises
    # ------------------------------------------------------------------
    print('\n' + '='*60)
    print('PHASE 2 — run_bc_approach')
    print('='*60)

    bc_args = []
    if args.mlp_only:
        bc_args.append('--mlp-only')
    if args.gru_only:
        bc_args.append('--gru-only')
    if args.transformer_only:
        bc_args.append('--transformer-only')

    for ex in exercises:
        if ex in fail_build:
            print(f'\n[SKIP] Exercise {ex:03d} — build_dataset failed')
            continue
        print(f'\n--- Exercise {ex:03d} ---')
        ok = run('run_bc_approach', ['--exercise', ex] + bc_args)
        if ok:
            ok_bc.append(ex)
        else:
            fail_bc.append(ex)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_total
    print(f'\n{"="*60}')
    print(f'SUMMARY  (total: {elapsed:.0f}s)')
    print(f'{"="*60}')
    print(f'  build_dataset  OK   : {ok_build}')
    if fail_build:
        print(f'  build_dataset  FAIL : {fail_build}')
    print(f'  run_bc_approach OK   : {ok_bc}')
    if fail_bc:
        print(f'  run_bc_approach FAIL : {fail_bc}')
    print()


if __name__ == '__main__':
    main()