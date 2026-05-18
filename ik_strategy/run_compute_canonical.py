'''
run_compute_canonical.py
=============================================================================
Interactive script to compute canonical trajectories for one or more
exercises using ShapeDBA. Saves results in the appropriate split subfolder.

Usage:
  py run_compute_canonical.py
  py run_compute_canonical.py --exercise 1
  py run_compute_canonical.py --exercise 1 --n-demos 10
  py run_compute_canonical.py --exercise 1 --n-demos 25

When prompted, type 'a' instead of a number to process all exercises.
n-demos controls which subjects are used (10->2 subjects, 25->5, 55->11).
'''

import argparse
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from utilities.config import DATA_ROOT
from utilities.split_utils import split_name, select_subjects, N_DEMOS_SPLITS
from canonical_approach.compute_canonical import _process_exercise
from utilities.plot_baseline_canonical import _fk_figure, _load_trajectory, _joints_figure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ask(prompt: str) -> str:
    return input(prompt).strip().lower()


def _all_exercise_numbers(landmarks_root: Path) -> list:
    found = set()
    for subj_dir in landmarks_root.glob('subject_*'):
        for exer_dir in subj_dir.glob('exercise_*'):
            try:
                found.add(int(exer_dir.name.split('_')[1]))
            except ValueError:
                pass
    return sorted(found)


def _parse_inputs(args) -> list:
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


def _selected_subject_nums(landmarks_root: Path, n_demos: int) -> list:
    """Returns sorted list of subject numbers to use for this split."""
    all_subj_dirs = sorted(landmarks_root.glob('subject_*'))
    selected = select_subjects(all_subj_dirs, n_demos)
    return [int(d.name.split('_')[1]) for d in selected]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute canonical trajectories via ShapeDBA.")
    parser.add_argument('--exercise', type=int, default=None,
                        help='Exercise number (e.g. 1). Omit to be prompted.')
    parser.add_argument('--n-demos', type=int, default=55, choices=N_DEMOS_SPLITS,
                        help='Split da usare: 10=2 soggetti, 25=5, 55=11 (default: 55).')
    args = parser.parse_args()

    exercise_numbers = _parse_inputs(args)
    landmarks_root   = DATA_ROOT / 'landmarks'
    dataset_root     = DATA_ROOT / 'dataset'

    # Seleziona i soggetti per questo split
    selected_nums = _selected_subject_nums(landmarks_root, args.n_demos)

    print(f'\nExercises     : {exercise_numbers}')
    print(f'Split         : n_{args.n_demos:02d}')
    print(f'Subjects used : {selected_nums}')
    print()

    t_total   = time.time()
    ok_list   = []
    fail_list = []

    for ex_num in exercise_numbers:
        ex_name   = f'exercise_{ex_num:03d}'
        spl_name  = split_name(args.n_demos)
        split_dir = dataset_root / ex_name / spl_name

        print(f'\n{"="*55}')
        print(f'  Processing {ex_name}  [{spl_name}]')
        print(f'{"="*55}')
        t0 = time.time()

        try:
            _process_exercise(
                exercise_num    = ex_num,
                landmarks_root  = landmarks_root,
                dataset_root    = dataset_root,
                filter_subject  = None,
                n_demos         = args.n_demos,
            )
            print(f'  [OK] {ex_name} completed in {time.time() - t0:.1f}s')
            ok_list.append(ex_name)

            # --- Plot canonical FK (salvato nel split_dir) ---
            canonical_path = split_dir / 'canonical.csv'
            plot_dir       = split_dir / 'plots'
            joints_dir     = split_dir / 'plots_joints'
            plot_dir.mkdir(parents=True, exist_ok=True)
            joints_dir.mkdir(parents=True, exist_ok=True)

            try:
                trajectory = _load_trajectory(canonical_path)

                fig      = _fk_figure(trajectory, ex_num, 'canonical')
                out_path = plot_dir / 'canonical_fk.png'
                fig.savefig(out_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f'  Plot saved -> {out_path.relative_to(DATA_ROOT)}')

                fig_j      = _joints_figure(trajectory, ex_num, 'canonical')
                out_path_j = joints_dir / 'joints_canonical.png'
                fig_j.savefig(out_path_j, dpi=150, bbox_inches='tight')
                plt.close(fig_j)
                print(f'  Saved -> {out_path_j.relative_to(DATA_ROOT)}')

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