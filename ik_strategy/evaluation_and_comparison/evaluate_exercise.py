'''
evaluate_exercise.py
=============================================================================
Valutazione completa di un singolo esercizio.

Per ogni esercizio carica baseline, canonical e demo umane, genera le
traiettorie BC con tutte e tre le architetture, calcola le metriche e
produce i plot.

Output (in data/dataset/exercise_NNN/evaluation/):
    results_summary.csv
    results_per_joint.csv
    plot_degradation_chain.png
    plot_rmse_per_joint.png
    plot_pearson_per_joint.png
    plot_smoothness.png
    plot_velocity_<joint>.png

Uso standalone:
    py -m evaluation_and_comparison.evaluate_exercise --exercise 1
    py -m evaluation_and_comparison.evaluate_exercise --exercise 1 11 21
'''

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from utilities.config import DATA_ROOT

from evaluation_and_comparison._config  import ARCHITECTURES, get_modality
from evaluation_and_comparison._io      import (
    load_baseline, load_canonical, load_human_demos,
    load_bc_trajectory, save_results_csv,
)
from evaluation_and_comparison._metrics import (
    compute_metrics, compute_cartesian_metrics, aggregate_metrics, print_summary,
)
from evaluation_and_comparison._plots   import (
    plot_degradation_chain, plot_rmse_per_joint, plot_pearson_per_joint,
    plot_velocity_profile,
    plot_cartesian_trajectories, plot_cartesian_rmse,
    plot_cartesian_velocity, plot_cartesian_pearson,
    plot_summary_heatmap,
)


def run_exercise_evaluation(exercise_num: int,
                             n_steps: Optional[int] = None) -> Dict[str, Dict]:
    '''
    Valuta un singolo esercizio con tutte le architetture disponibili.

    Ritorna il dict results (method → metrics) per eventuale riutilizzo
    nell'analisi modality (evita di ricalcolare due volte).
    '''
    exercise_name  = f'exercise_{exercise_num:03d}'
    dataset_dir    = DATA_ROOT / 'dataset' / exercise_name
    landmarks_root = DATA_ROOT / 'landmarks'
    output_dir     = dataset_dir / 'evaluation'
    output_dir.mkdir(parents=True, exist_ok=True)
    modality = get_modality(exercise_num)

    print(f'\n{"="*65}')
    print(f'  Exercise {exercise_num:03d}  [{modality}]')
    print(f'{"="*65}')

    # --- Caricamento dati ---------------------------------------------------
    baseline = load_baseline(dataset_dir)
    if baseline is None:
        print('  SKIP: baseline.csv mancante.')
        return {}

    canonical  = load_canonical(dataset_dir)

    print('\nCaricamento demo umane ...')
    human_demos = load_human_demos(landmarks_root, exercise_num)

    # --- Caricamento traiettorie BC pre-generate ----------------------------
    # Genera prima con: py -m bc_approach.{MLP,GRU,Transformer}.test_bc --exercise N
    print('\nCaricamento traiettorie BC ...')
    bc_trajs: Dict[str, Optional[object]] = {}
    for arch_name in ARCHITECTURES:
        bc_trajs[arch_name] = load_bc_trajectory(dataset_dir, arch_name)

    # --- Calcolo metriche ---------------------------------------------------
    print('\nCalcolo metriche ...')
    results: Dict[str, Dict] = {}

    if human_demos:
        joint_list = [compute_metrics(d, baseline) for d in human_demos]
        cart_list  = [compute_cartesian_metrics(d, baseline) for d in human_demos]
        merged     = [{**j, **c} for j, c in zip(joint_list, cart_list)]
        results['Human demos'] = aggregate_metrics(merged)
        r = results['Human demos']
        print(f'  [Human demos      ]  '
              f'DTW={r["dtw_distance"]:>9.2f}°  RMSE={r["rmse_mean"]:>6.2f}°  '
              f'DTW_cart={r["cart_dtw"]:.4f}m  '
              f'Rw={r["cart_rmse_r_wrist"]:.4f}m  Lw={r["cart_rmse_l_wrist"]:.4f}m')

    if canonical is not None:
        jm = compute_metrics(canonical, baseline, label='Canonical')
        cm = compute_cartesian_metrics(canonical, baseline, label='Canonical [cart]')
        results['Canonical'] = {**jm, **cm}

    for arch_name, traj in bc_trajs.items():
        if traj is not None:
            jm = compute_metrics(traj, baseline, label=arch_name)
            cm = compute_cartesian_metrics(traj, baseline, label=f'{arch_name} [cart]')
            results[arch_name] = {**jm, **cm}

    if not results:
        print('  Nessun risultato — verificare canonical.csv e modelli.')
        return {}

    # --- Salvataggio e plot -------------------------------------------------
    print('\nSalvataggio CSV ...')
    save_results_csv(results, output_dir)

    print('\nGenerazione plot ...')
    suffix = f' -- Exercise {exercise_num:03d} [{modality}]'

    # Determina braccio attivo dalla baseline (joint con std massima)
    stds        = np.std(baseline, axis=0)
    active_joint = int(np.argmax(stds))
    active_side  = 'right' if active_joint < 8 else 'left'

    all_trajs = {'Baseline': baseline, 'Canonical': canonical, **bc_trajs}

    plot_degradation_chain(results, output_dir, exercise_num, modality)
    plot_rmse_per_joint(results, output_dir, suffix, active_side=active_side)
    plot_pearson_per_joint(results, output_dir, suffix, active_side=active_side)
    plot_velocity_profile(all_trajs, output_dir, baseline=baseline)

    # Plot cartesiani
    plot_cartesian_trajectories(all_trajs, output_dir, suffix, active_side=active_side)
    plot_cartesian_rmse(results, output_dir, suffix, active_side=active_side)
    plot_cartesian_pearson(results, output_dir, suffix, active_side=active_side)
    plot_cartesian_velocity(all_trajs, output_dir, baseline=baseline)

    # Heatmap riassuntiva finale
    plot_summary_heatmap(results, output_dir, suffix, active_side=active_side)

    print_summary(results, f'Exercise {exercise_num:03d} [{modality}]')
    print(f'\n  Output → {output_dir}')
    return results


# ============================================================================
# Standalone
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Valutazione singolo esercizio (degradation chain + metriche).')
    parser.add_argument(
        '--exercise', type=int, nargs='+', required=True, metavar='N',
        help='Numero/i esercizio (es. --exercise 1  oppure  --exercise 1 11 21).')
    parser.add_argument(
        '--steps', type=int, default=None,
        help='Override numero di step BC. Default: lunghezza baseline.csv.')
    args = parser.parse_args()

    for ex in args.exercise:
        run_exercise_evaluation(ex, args.steps)

    print('\nDone.')


if __name__ == '__main__':
    main()