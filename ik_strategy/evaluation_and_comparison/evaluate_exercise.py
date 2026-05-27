'''
evaluate_exercise.py
=============================================================================
Complete evaluation of a single exercise

Output (in data/dataset/exercise_NNN/n_XX/evaluation/):
    results_summary.csv
    results_per_joint.csv
    results_per_endpoint.csv
    plot_degradation_chain.png
    plot_rmse_per_joint.png
    plot_pearson_per_joint.png
    plot_velocity_<joint>.png
    plot_cartesian_*.png
    plot_summary_heatmap.png
    plot_spider_chart.png

Standalone usage:
    py -m evaluation_and_comparison.evaluate_exercise --exercise 1
    py -m evaluation_and_comparison.evaluate_exercise --exercise 1 --n-demos 10
'''

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from utilities.config import DATA_ROOT
from utilities.split_utils import split_name, N_DEMOS_SPLITS

from evaluation_and_comparison._config  import ARCHITECTURES, get_modality
from evaluation_and_comparison._io      import (
    load_baseline, load_canonical, load_canonical_shape,
    load_human_demos, load_bc_trajectory, load_bc_runs_variance, save_results_csv,
)
from evaluation_and_comparison._metrics import (
    compute_metrics, compute_cartesian_metrics, aggregate_metrics, print_summary,
)
from evaluation_and_comparison._plots   import (
    plot_degradation_chain, plot_velocity_profile,
    plot_cartesian_trajectories,
    plot_cartesian_velocity,
    plot_3d_trajectories, plot_summary_heatmap, plot_spider_chart,
)


def run_exercise_evaluation(exercise_num: int, n_demos: int = 55, n_steps: Optional[int] = None) -> Dict[str, Dict]:
    '''
    Evaluate a single exercise with all available architectures.

    Returns the results dict (method → metrics) for optional reuse
    in the modality analysis (avoids recomputing twice).
    '''
    exercise_name  = f'exercise_{exercise_num:03d}'
    dataset_dir    = DATA_ROOT / 'dataset' / exercise_name
    split_dir      = dataset_dir / split_name(n_demos)
    landmarks_root = DATA_ROOT / 'landmarks'
    output_dir     = split_dir / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    modality = get_modality(exercise_num)

    print(f'\n{"="*65}')
    print(f'  Exercise {exercise_num:03d}  [{modality}]  split={split_name(n_demos)}')
    print(f'{"="*65}')

    # _____ Loading data _____
    baseline = load_baseline(dataset_dir)       # baseline stays in the root folder
    if baseline is None:
        print('  SKIP: baseline.csv missing.')
        return {}

    # Choose active arm - needed for metrics and plots
    stds         = np.std(baseline, axis=0)
    active_joint = int(np.argmax(stds))
    active_side  = 'right' if active_joint < 8 else 'left'
    print(f'  Active side: {active_side}')

    canonical        = load_canonical(split_dir)        # DBA standard
    canonical_shape  = load_canonical_shape(split_dir)  # ShapeDBA
    human_demos      = load_human_demos(landmarks_root, exercise_num, n_demos=n_demos)

    # _____ Loading BC trajectories pre-generated _____
    print('\nLoading trajectories BC ...')
    bc_trajs: Dict[str, Optional[object]] = {}
    for arch_name in ARCHITECTURES:
        bc_trajs[arch_name] = load_bc_trajectory(split_dir, arch_name)

    # _____ Metrics computation _____
    print('\nMetrics computation...')
    results: Dict[str, Dict] = {}

    # Human demos - used only as reference for bounds, not shown in plots
    # Compute per-demo metrics to extract individual best/worst for each metric
    _LOWER_IS_BETTER_KEYS = {
        'dtw_distance', 'rmse_mean', 'peak_error_mean',
        'cart_dtw',
        'cart_rmse_r_wrist', 'cart_rmse_l_wrist',
        'cart_rmse_r_elbow', 'cart_rmse_l_elbow',
        'cart_peak_r_wrist', 'cart_peak_l_wrist',
    }
    human_bounds: Dict[str, tuple] = {}   # {metric_key: (best_val, worst_val)}

    if human_demos:
        joint_list = [compute_metrics(d, baseline) for d in human_demos]
        cart_list  = [compute_cartesian_metrics(d, baseline, active_side=active_side)
                      for d in human_demos]
        merged = [{**j, **c} for j, c in zip(joint_list, cart_list)]
        results['Human demos'] = aggregate_metrics(merged)
        r = results['Human demos']
        print(f'  [Human demos (ref)]  DTW={r["dtw_distance"]:>9.2f}  '
              f'RMSE={r["rmse_mean"]:>6.2f}  DTW_cart={r["cart_dtw"]:.4f}m  '
              f'Smooth={r["smoothness"]:.4f}')

        # Collect per-demo values and compute (best, worst)
        per_demo_vals: Dict[str, List[float]] = {}
        for combo in merged:
            for key, val in combo.items():
                if isinstance(val, float) and not np.isnan(val):
                    per_demo_vals.setdefault(key, []).append(val)
        for key, vals in per_demo_vals.items():
            if not vals:
                continue
            if key in _LOWER_IS_BETTER_KEYS:
                human_bounds[key] = (min(vals), max(vals))   # (best=min, worst=max)
            else:
                human_bounds[key] = (max(vals), min(vals))   # (best=max, worst=min)

    if canonical is not None:
        jm = compute_metrics(canonical, baseline, label='Canonical')
        cm = compute_cartesian_metrics(canonical, baseline,
                                       active_side=active_side, label='Canonical [cart]')
        results['Canonical'] = {**jm, **cm}

    if canonical_shape is not None:
        jm = compute_metrics(canonical_shape, baseline, label='CanonicalShape')
        cm = compute_cartesian_metrics(canonical_shape, baseline,
                                       active_side=active_side, label='CanonicalShape [cart]')
        results['CanonicalShape'] = {**jm, **cm}

    for arch_name, traj in bc_trajs.items():
        if traj is not None:
            jm = compute_metrics(traj, baseline, label=arch_name)
            cm = compute_cartesian_metrics(traj, baseline,
                                           active_side=active_side,
                                           label=f'{arch_name} [cart]')
            results[arch_name] = {**jm, **cm}

    if not results:
        print('  No result - check canonical.csv and models.')
        return {}

    # ── Variance across training runs (None if single-run) ──────────────────
    arch_variance: Dict[str, Optional[Dict]] = {}
    for arch_name in ARCHITECTURES:
        arch_variance[arch_name] = load_bc_runs_variance(split_dir, arch_name)
    has_variance = any(v is not None for v in arch_variance.values())
    if has_variance:
        print('\nVarianza inter-run caricata:')
        for arch, var in arch_variance.items():
            if var:
                print(f'  {arch}: DTW=±{var.get("dtw_distance", float("nan")):.4f}  '
                      f'cart_DTW=±{var.get("cart_dtw", float("nan")):.4f}m')

    # --- Save and plot ------------------------------------------------------
    print('\nSalvataggio CSV ...')
    save_results_csv(results, output_dir)

    print('\nGenerazione plot ...')
    suffix = f' -- Exercise {exercise_num:03d} [{modality}]'

    all_trajs = {
        'Baseline'      : baseline,
        'Canonical'     : canonical,
        'CanonicalShape': canonical_shape,
        **{k: v for k, v in bc_trajs.items() if v is not None},
    }
    # remove None values (canonical/canonicalShape missing)
    all_trajs = {k: v for k, v in all_trajs.items() if v is not None}

    plot_degradation_chain(results, output_dir, exercise_num, modality,
                           arch_variance=arch_variance)
    plot_velocity_profile(all_trajs, output_dir, baseline=baseline)

    # Cartesian plot 
    plot_cartesian_trajectories(all_trajs, output_dir, suffix, active_side=active_side)
    plot_cartesian_velocity(all_trajs, output_dir, baseline=baseline)
    plot_3d_trajectories(all_trajs, output_dir, active_side=active_side)

    # Final summary heatmap + spider chart
    plot_summary_heatmap(results, output_dir, suffix,
                         active_side=active_side, human_bounds=human_bounds,
                         arch_variance=arch_variance)
    plot_spider_chart(results, output_dir, suffix,
                      active_side=active_side, human_bounds=human_bounds)

    print_summary(results, f'Exercise {exercise_num:03d} [{modality}]')
    print(f'\n  Output → {output_dir}')
    return results


# ============================================================================
# Standalone
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Single exercise evaluation (degradation chain + metrics).')
    parser.add_argument('--exercise', type=int, nargs='+', required=True, metavar='N')
    parser.add_argument('--n-demos', type=int, default=55, choices=N_DEMOS_SPLITS, help='Split to be evaluated (default: 55).')
    parser.add_argument('--steps', type=int, default=None)
    args = parser.parse_args()

    for ex in args.exercise:
        run_exercise_evaluation(ex, args.n_demos, args.steps)
    print('\nDone.')


if __name__ == '__main__':
    main()