"""
evaluate_demos.py
=============================================================================
Analisi della sensibilita al numero di dimostrazioni disponibili.

Risponde alla domanda: quante demo (soggetti) bastano?
Il sistema degrada significativamente passando da 55 a 25 o a 10 demo?

Prerequisito: aver gia eseguito per ogni split:
  py run_bc_approach.py --exercise X --n-demos 10
  py run_bc_approach.py --exercise X --n-demos 25
  py run_bc_approach.py --exercise X --n-demos 55
  py -m evaluation_and_comparison.evaluate --exercise X --n-demos 10 25 55

Il file legge i results_summary.csv gia calcolati — non riesegue inference.

Output PER ESERCIZIO (in dataset/exercise_NNN/evaluation_demos/):
  results_by_n_demos.csv
  plot_cart_dtw_vs_demos.png  (e altri per metrica)

Output GLOBALE (in data/evaluation_demos/):
  results_all_exercises.csv
  results_aggregated.csv
  plot_cart_dtw_vs_demos.png  (media su tutti gli esercizi)

Usage:
  py -m evaluation_and_comparison.evaluate_demos --exercise 21
  py -m evaluation_and_comparison.evaluate_demos --exercise 21 22 23
  py -m evaluation_and_comparison.evaluate_demos --all
"""

import argparse
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utilities.config import DATA_ROOT
from utilities.split_utils import N_DEMOS_SPLITS, split_name
from evaluation_and_comparison._config import PALETTE


# Candidati metriche — solo quelle con valori non-NaN nel CSV vengono plottate
# (automaticamente filtra il braccio inattivo)
METRICS_CANDIDATES = [
    ('cart_dtw',          'DTW (m)',          True),
    ('cart_rmse_wrist',   'RMSE wrist (m)',   True),
    ('cart_rmse_elbow',   'RMSE elbow (m)',   True),
    ('cart_rmse_l_wrist', 'RMSE Lw (m)',      True),
    ('cart_rmse_r_wrist', 'RMSE Rw (m)',      True),
    ('cart_pearson_mean', 'Pearson cart',     False),
    ('rmse_mean',         'RMSE joint (deg)', True),
]

METHOD_ORDER = ['Human demos', 'Canonical', 'MLP', 'GRU', 'Transformer']


def _load_split_results(exercise_dir: Path, n_demos: int) -> Optional[pd.DataFrame]:
    csv_path = exercise_dir / split_name(n_demos) / 'evaluation' / 'results_summary.csv'
    if not csv_path.exists():
        print(f'  [!] Mancante: {split_name(n_demos)}/evaluation/results_summary.csv')
        return None
    df = pd.read_csv(csv_path)
    df['n_demos']      = n_demos
    df['n_subjects']   = n_demos // 5
    df['exercise_num'] = int(exercise_dir.name.split('_')[1])
    return df


def _plot_metrics_vs_demos(df: pd.DataFrame, output_dir: Path,
                            title_suffix: str = '') -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x_ticks = sorted(df['n_demos'].unique())
    methods  = [m for m in METHOD_ORDER
                if m in df['method'].values and m != 'Human demos']

    for metric_key, ylabel, lower_is_better in METRICS_CANDIDATES:
        if metric_key not in df.columns or df[metric_key].isna().all():
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        plotted = False
        for method in methods:
            sub = df[df['method'] == method].sort_values('n_demos')
            if sub.empty or sub[metric_key].isna().all():
                continue
            ax.plot(sub['n_demos'], sub[metric_key],
                    marker='o', linewidth=2, markersize=8,
                    label=method, color=PALETTE.get(method, '#333'), alpha=0.9)
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        direction = '(lower=better)' if lower_is_better else '(higher=better)'
        ax.set_xlabel('N dimostrazioni', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{n}\n({n//5} sogg.)' for n in x_ticks], fontsize=10)
        ax.set_title(f'{ylabel} vs N demos{title_suffix}\n{direction}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        fname = f'plot_{metric_key}_vs_demos.png'
        fig.savefig(output_dir / fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved -> {fname}')


def run_exercise_demos_analysis(exercise_num: int) -> Optional[pd.DataFrame]:
    exercise_dir = DATA_ROOT / 'dataset' / f'exercise_{exercise_num:03d}'
    if not exercise_dir.is_dir():
        print(f'  [!] exercise_{exercise_num:03d} non trovato.')
        return None

    print(f'\n{"="*60}')
    print(f'  Exercise {exercise_num:03d} — demos sensitivity')
    print(f'{"="*60}')

    dfs = [_load_split_results(exercise_dir, n) for n in N_DEMOS_SPLITS]
    dfs = [d for d in dfs if d is not None]

    if len(dfs) < 2:
        print('  Meno di 2 split — skip.')
        return None

    all_data = pd.concat(dfs, ignore_index=True)
    out_dir  = exercise_dir / 'evaluation_demos'
    out_dir.mkdir(parents=True, exist_ok=True)

    p = out_dir / 'results_by_n_demos.csv'
    all_data.to_csv(p, index=False)
    print(f'  Saved -> {p.name}')

    _plot_metrics_vs_demos(all_data, out_dir,
                           title_suffix=f' -- Exercise {exercise_num:03d}')
    return all_data


def run_demos_analysis(exercise_nums: List[int]) -> None:
    all_results = [run_exercise_demos_analysis(ex) for ex in exercise_nums]
    all_results = [r for r in all_results if r is not None]

    if not all_results:
        print('\nNessun risultato. Eseguire prima run_bc_approach + evaluate.')
        return

    global_df = pd.concat(all_results, ignore_index=True)
    out_dir   = DATA_ROOT / 'evaluation_demos'
    out_dir.mkdir(parents=True, exist_ok=True)

    p = out_dir / 'results_all_exercises.csv'
    global_df.to_csv(p, index=False)
    print(f'\nSaved -> {p}')

    metric_cols = [m[0] for m in METRICS_CANDIDATES if m[0] in global_df.columns]
    agg = (global_df.groupby(['n_demos', 'method'])[metric_cols]
                    .mean().reset_index())
    p = out_dir / 'results_aggregated.csv'
    agg.to_csv(p, index=False)
    print(f'Saved -> {p.name}')

    print('\nPlot globali (media su tutti gli esercizi) ...')
    _plot_metrics_vs_demos(agg, out_dir,
                           title_suffix=' -- All exercises (mean)')
    print(f'\nOutput globale -> {out_dir}')


def main():
    parser = argparse.ArgumentParser(
        description='Ablation study: sensibilita al numero di dimostrazioni.')
    parser.add_argument('--exercise', type=int, nargs='+', metavar='N')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all:
        dataset_root  = DATA_ROOT / 'dataset'
        exercise_nums = sorted(int(d.name.split('_')[1])
                               for d in dataset_root.glob('exercise_???')
                               if d.is_dir())
    elif args.exercise:
        exercise_nums = args.exercise
    else:
        parser.error('Specificare --exercise N [N ...] oppure --all.')

    run_demos_analysis(exercise_nums)
    print('\nDone.')


if __name__ == '__main__':
    main()