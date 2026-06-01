'''
aggregate_runs.py
=============================================================================
Aggregates results from multiple training runs for each BC architecture.

For each architecture (MLP, GRU, Transformer):
  1. Finds all run_*/bc_trajectory.csv files in the arch/ folder
  2. Resamples trajectories to the median length
  3. Computes the mean trajectory  -> arch/bc_trajectory_mean.csv
  4. Computes per-run metrics vs baseline -> arch/runs_metrics.csv

The mean trajectory is then used by evaluate_exercise.py
(load_bc_trajectory prefers bc_trajectory_mean.csv if available).
The metric std is shown in the heatmap and degradation chain plot.

Usage:
  py -m bc_approach.aggregate_runs --exercise 1
  py -m bc_approach.aggregate_runs --exercise 1 --n-demos 10
  py -m bc_approach.aggregate_runs --exercise 1 --n-demos 10 25 55
'''

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

from utilities.config import DATA_ROOT, JOINT_COLS
from utilities.split_utils import split_name, N_DEMOS_SPLITS
from evaluation_and_comparison._config import ARCHITECTURES
from evaluation_and_comparison._metrics import compute_metrics, compute_cartesian_metrics

# Scalar metrics to save in runs_metrics.csv
_SCALAR_JOINT_KEYS = [
    'dtw_distance', 'rmse_mean', 'peak_error_mean', 'pearson_mean', 'smoothness',
]
_SCALAR_CART_KEYS = [
    'cart_dtw',
    'cart_rmse_r_wrist', 'cart_rmse_l_wrist',
    'cart_rmse_r_elbow', 'cart_rmse_l_elbow',
    'cart_peak_r_wrist', 'cart_peak_l_wrist',
    'cart_pearson_mean', 'cart_smoothness',
]


def _resample(arr: np.ndarray, target_len: int) -> np.ndarray:
    '''Resample (T, D) to (target_len, D) using linear interpolation.'''
    if len(arr) == target_len:
        return arr
    t_old = np.linspace(0, 1, len(arr))
    t_new = np.linspace(0, 1, target_len)
    return interp1d(t_old, arr, axis=0, kind='linear')(t_new)


def aggregate_arch(arch_dir: Path, baseline: np.ndarray, active_side: str) -> bool:
    '''
    Aggregates all run_*/bc_trajectory.csv found in arch_dir.

    Saves:
      arch_dir/bc_trajectory_mean.csv  - mean frame-per-frame (resampled)
      arch_dir/runs_metrics.csv        - metrics per-run vs baseline
    '''
    run_dirs   = sorted(arch_dir.glob('run_*/'))
    traj_paths = [d / 'bc_trajectory.csv' for d in run_dirs if (d / 'bc_trajectory.csv').exists()]

    if len(traj_paths) < 2:
        print(f'  [{arch_dir.name}] Less than 2 runs found - skip aggregation.')
        return False

    n_runs = len(traj_paths)
    print(f'\n  [{arch_dir.name}] Aggregating {n_runs} run(s) ...')

    # Load trajectories
    trajs = []
    for p in traj_paths:
        df  = pd.read_csv(p)
        cols = [c for c in JOINT_COLS if c in df.columns]
        arr  = df[cols].dropna().values.astype(float)
        trajs.append(arr)
        print(f'    {p.parent.name}: {len(arr)} frame')

    # Resample to median length
    median_len = int(np.median([len(t) for t in trajs]))
    resampled  = [_resample(t, median_len) for t in trajs]

    # ── Mean trajectory ──────────────────────────────────────────────────
    mean_traj = np.mean(resampled, axis=0)   # (median_len, 16)
    mean_path = arch_dir / 'bc_trajectory_mean.csv'
    pd.DataFrame(mean_traj, columns=JOINT_COLS).to_csv(mean_path, index=False)
    print(f'    -> bc_trajectory_mean.csv  ({median_len} frame)')

    # ── Metrics per-run ──────────────────────────────────────────────────
    rows = []
    for i, traj in enumerate(resampled):
        jm  = compute_metrics(traj, baseline)
        cm  = compute_cartesian_metrics(traj, baseline, active_side=active_side)
        row = {'run': i + 1}
        for k in _SCALAR_JOINT_KEYS:
            row[k] = jm.get(k, float('nan'))
        for k in _SCALAR_CART_KEYS:
            row[k] = cm.get(k, float('nan'))
        rows.append(row)
        print(f'    run_{i+1}: DTW={jm["dtw_distance"]:.2f}°  '
              f'cart_DTW={cm["cart_dtw"]:.4f}m  '
              f'smooth={jm["smoothness"]:.4f}')

    runs_df   = pd.DataFrame(rows)
    runs_path = arch_dir / 'runs_metrics.csv'
    runs_df.to_csv(runs_path, index=False)
    print(f'    -> runs_metrics.csv  ({n_runs} run)')

    # ── Summary of variance ─────────────────────────────────────────────────
    print(f'    Std across {n_runs} runs:')
    for col in ['dtw_distance', 'cart_dtw', 'rmse_mean', 'smoothness']:
        if col in runs_df.columns:
            print(f'      {col:<24}: ±{runs_df[col].std():.4f}')

    return True


def run_aggregation(exercise_num: int, n_demos: int = 55) -> None:
    '''Aggregates all architectures for one exercise and split.'''
    exercise_name = f'exercise_{exercise_num:03d}'
    dataset_dir   = DATA_ROOT / 'dataset' / exercise_name
    split_dir     = dataset_dir / split_name(n_demos)

    print(f'\n{"="*60}')
    print(f'  Aggregating runs - {exercise_name}  [{split_name(n_demos)}]')
    print(f'{"="*60}')

    # Baseline
    baseline_path = dataset_dir / 'baseline.csv'
    if not baseline_path.exists():
        print(f'  [!] baseline.csv not found - impossible to compute metrics.')
        return
    baseline_df = pd.read_csv(baseline_path)
    baseline    = baseline_df[JOINT_COLS].dropna().values.astype(float)

    # Active arm (same logic as evaluate_exercise.py)
    stds         = np.std(baseline, axis=0)
    active_joint = int(np.argmax(stds))
    active_side  = 'right' if active_joint < 8 else 'left'
    print(f'  Active side: {active_side}')

    results = {}
    for arch_name in ARCHITECTURES:
        arch_dir = split_dir / arch_name
        if not arch_dir.is_dir():
            print(f'\n  [{arch_name}] Folder not found - skip.')
            continue
        ok = aggregate_arch(arch_dir, baseline, active_side)
        results[arch_name] = 'OK' if ok else 'SKIP'

    print(f'\n  {"─"*40}')
    for arch, status in results.items():
        print(f'  [{status}] {arch}')


def main():
    parser = argparse.ArgumentParser(description='Aggregate BC training runs for each architecture.')
    parser.add_argument('--exercise', type=int, nargs='+', required=True, metavar='N', help='Exercise number(s).')
    parser.add_argument('--n-demos', type=int, nargs='+', default=[55], choices=N_DEMOS_SPLITS, help='Split(s) to aggregate (default: 55).')
    args = parser.parse_args()

    for n in args.n_demos:
        for ex in args.exercise:
            run_aggregation(ex, n)

    print('\nDone.')


if __name__ == '__main__':
    main()
