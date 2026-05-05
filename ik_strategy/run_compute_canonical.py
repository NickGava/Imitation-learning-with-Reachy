'''
run_compute_canonical.py
=============================================================================
Interactive script to compute canonical trajectories for one or more
exercises using DTW Barycenter Averaging (DBA).

Usage:
  py run_compute_canonical.py
  py run_compute_canonical.py --exercise 1
  py run_compute_canonical.py --exercise 1 --subject 3

When prompted, type 'a' instead of a number to process all exercises.
'''

import argparse
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from utilities.config import DATA_ROOT
from canonical_approach.compute_canonical import _process_exercise
from utilities.plot_baseline_canonical import _fk_figure, _load_trajectory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ask(prompt: str) -> str:
    return input(prompt).strip().lower()


def _all_exercise_numbers(landmarks_root: Path) -> list:
    """Scans landmarks/ to find all exercise numbers present."""
    found = set()
    for subj_dir in landmarks_root.glob('subject_*'):
        for exer_dir in subj_dir.glob('exercise_*'):
            try:
                found.add(int(exer_dir.name.split('_')[1]))
            except ValueError:
                pass
    return sorted(found)


def _parse_inputs(args) -> list:
    """
    Returns a list of exercise numbers to process.
    Accepts CLI args or interactive prompt. 'a' means all.
    """
    if args.exercise is not None:
        return [args.exercise]

    raw = _ask("Exercise number (or 'a' for all): ")

    if raw == 'a':
        landmarks_root = DATA_ROOT / 'landmarks'
        numbers = _all_exercise_numbers(landmarks_root)
        if not numbers:
            print('No exercises found in landmarks/.')
            sys.exit(1)
        return numbers

    try:
        return [int(raw)]
    except ValueError:
        print(f"Invalid input: '{raw}'. Enter a number or 'a'.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute canonical trajectories via DBA. Use 'a' for all exercises.")
    parser.add_argument('--exercise', type=int, default=None,
                        help='Exercise number (e.g. 1). Omit to be prompted.')
    parser.add_argument('--subject',  type=int, default=None,
                        help='Filter to a single subject (optional).')
    args = parser.parse_args()

    exercise_numbers = _parse_inputs(args)

    landmarks_root = DATA_ROOT / 'landmarks'
    dataset_root   = DATA_ROOT / 'dataset'

    print(f'\nExercises to process : {exercise_numbers}')
    if args.subject is not None:
        print(f'Subject filter       : {args.subject}')
    print()

    t_total  = time.time()
    ok_list  = []
    fail_list = []

    for ex_num in exercise_numbers:
        ex_name = f'exercise_{ex_num:03d}'
        print(f'\n{"="*55}')
        print(f'  Processing {ex_name}')
        print(f'{"="*55}')
        t0 = time.time()
        try:
            _process_exercise(ex_num, landmarks_root, dataset_root, args.subject)
            print(f'  [OK] {ex_name} completed in {time.time() - t0:.1f}s')
            ok_list.append(ex_name)

            # --- Plot canonical FK ---
            canonical_path = dataset_root / ex_name / 'canonical.csv'
            plot_dir       = dataset_root / ex_name / 'plots'
            plot_dir.mkdir(parents=True, exist_ok=True)
            try:
                trajectory = _load_trajectory(canonical_path)
                fig        = _fk_figure(trajectory, ex_num, 'canonical')
                out_path   = plot_dir / 'canonical_fk.png'
                fig.savefig(out_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f'  Plot saved → {out_path.relative_to(DATA_ROOT)}')
            except Exception as e:
                print(f'  [WARN] Plot failed for {ex_name}: {e}')
        except Exception as e:
            print(f'  [FAIL] {ex_name}: {e}')
            fail_list.append(ex_name)

    print(f'\n{"="*55}')
    print(f'  SUMMARY  (total: {time.time() - t_total:.1f}s)')
    print(f'{"="*55}')
    for name in ok_list:
        print(f'  [OK]   {name}')
    for name in fail_list:
        print(f'  [FAIL] {name}')
    print()


if __name__ == '__main__':
    main()