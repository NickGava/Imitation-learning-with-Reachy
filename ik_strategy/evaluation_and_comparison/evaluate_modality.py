"""
evaluate_modality.py
=============================================================================
Analisi comparativa tra modalita di acquisizione: Stereo vs Mixed vs Mono.

Risponde alla domanda: la pipeline stereo porta un beneficio misurabile
rispetto alla mono? Le due sorgenti possono essere combinate?

Approccio: legge i results_summary.csv gia calcolati da evaluate_exercise.py
per ogni esercizio del split richiesto. Non ricalcola metriche ne' inference.

Prerequisito:
  py -m evaluation_and_comparison.evaluate --all --n-demos 55
  (o il n-demos desiderato)

Output (in data/evaluation_modality/):
  results_by_exercise.csv     <- (esercizio x metodo)
  results_aggregated.csv      <- (modalita x metodo), media sui 5 esercizi
  plot_dtw_modality.png
  plot_rmse_modality.png
  plot_pearson_modality.png
  plot_rmse_heatmap_MLP.png
  plot_rmse_heatmap_GRU.png
  plot_rmse_heatmap_Transformer.png

Usage:
  py -m evaluation_and_comparison.evaluate_modality
  py -m evaluation_and_comparison.evaluate_modality --n-demos 55
  py -m evaluation_and_comparison.evaluate_modality --n-demos 10
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utilities.config import DATA_ROOT
from utilities.split_utils import split_name, N_DEMOS_SPLITS
from evaluation_and_comparison._config import (
    ARCHITECTURES, MODALITY_GROUPS, PALETTE, ACTIVE_LABELS,
    get_exercise_type,
)


METHOD_ORDER = ['Canonical', 'CanonicalShape', 'MLP', 'GRU', 'Transformer']


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _load_exercise_results(exercise_num: int, n_demos: int) -> Optional[pd.DataFrame]:
    """Legge results_summary.csv dal split corrispondente."""
    path = (DATA_ROOT / 'dataset' / f'exercise_{exercise_num:03d}'
            / split_name(n_demos) / 'evaluation' / 'results_summary.csv')
    if not path.exists():
        print(f'  [!] Mancante: exercise_{exercise_num:03d}/{split_name(n_demos)}/evaluation/results_summary.csv')
        return None
    df = pd.read_csv(path)
    df['exercise_num']  = exercise_num
    df['exercise_type'] = get_exercise_type(exercise_num)
    df['modality']      = next(
        mod for mod, nums in MODALITY_GROUPS.items() if exercise_num in nums
    )
    return df


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _grouped_bar_modality(agg: pd.DataFrame, metric: str,
                           ylabel: str, title: str,
                           output_path: Path) -> None:
    """Grouped bar: asse X = architetture, colori = modalita."""
    methods  = [m for m in ['Canonical', 'CanonicalShape'] + list(ARCHITECTURES.keys())
                if m in agg['method'].values]
    mods     = list(MODALITY_GROUPS.keys())
    x        = np.arange(len(methods))
    width    = 0.25
    off0     = -(len(mods) - 1) / 2 * width

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, mod in enumerate(mods):
        subset = agg[agg['modality'] == mod].set_index('method')
        vals   = [subset.loc[m, metric] if m in subset.index else np.nan
                  for m in methods]
        bars = ax.bar(x + off0 + i * width, vals, width,
                      label=mod, color=PALETTE.get(mod, f'C{i}'),
                      edgecolor='white', linewidth=1.0, alpha=0.88)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + abs(val) * 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(title='Modality', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {output_path.name}')


def _rmse_heatmap(agg: pd.DataFrame, arch: str, output_path: Path) -> None:
    """Heatmap RMSE joint per (modalita x joint) per una singola architettura."""
    # Per questa heatmap serve results_per_joint.csv, non results_summary.
    # Se non disponibile, skip silenzioso.
    print(f'  [skip] heatmap RMSE per joint non disponibile da results_summary '
          f'(serve results_per_joint.csv) — {arch}')


def _line_per_exercise(df: pd.DataFrame, metric: str,
                        ylabel: str, title: str,
                        output_path: Path) -> None:
    """Line plot: asse X = tipo esercizio (1-5), linee per (metodo x modalita)."""
    ex_types = sorted(df['exercise_type'].unique())
    methods  = [m for m in ['Canonical', 'CanonicalShape'] + list(ARCHITECTURES.keys())
                if m in df['method'].values]
    ls_map   = {'Stereo': '-', 'Mixed': '--', 'Mono': ':'}

    fig, ax = plt.subplots(figsize=(10, 5))
    for mod in MODALITY_GROUPS:
        mod_df = df[df['modality'] == mod]
        for method in methods:
            sub  = mod_df[mod_df['method'] == method]
            vals = [sub[sub['exercise_type'] == et][metric].mean()
                    if not sub[sub['exercise_type'] == et].empty else np.nan
                    for et in ex_types]
            ax.plot(ex_types, vals,
                    linestyle=ls_map[mod],
                    color=PALETTE.get(method, '#333'),
                    marker='o', linewidth=1.8, markersize=6,
                    label=f'{method} [{mod}]', alpha=0.85)

    ax.set_xticks(ex_types)
    ax.set_xticklabels([f'Type {t}' for t in ex_types])
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=7.5, ncol=3, loc='best')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {output_path.name}')


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_modality_analysis(n_demos: int = 55) -> None:
    """
    Legge i results_summary.csv gia calcolati per ogni esercizio
    del split n_demos, aggrega per modalita e produce i plot.
    """
    out_dir = DATA_ROOT / 'evaluation_modality'
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*65}')
    print(f'  ANALISI MODALITY  --  split {split_name(n_demos)}')
    print(f'{"="*65}')

    # Carica tutti i risultati disponibili
    all_dfs = []
    for mod, ex_nums in MODALITY_GROUPS.items():
        print(f'\n  {mod}:')
        for ex_num in ex_nums:
            df = _load_exercise_results(ex_num, n_demos)
            if df is not None:
                all_dfs.append(df)
                print(f'    exercise_{ex_num:03d}: {len(df)} metodi')

    if not all_dfs:
        print('\nNessun risultato trovato. Eseguire prima:')
        print(f'  py -m evaluation_and_comparison.evaluate --all --n-demos {n_demos}')
        return

    by_ex = pd.concat(all_dfs, ignore_index=True)

    # Colonna derivata: RMSE del braccio attivo (sempre definita,
    # compatibile con CSV generati prima dell'aggiornamento di _io.py)
    if 'cart_rmse_wrist' not in by_ex.columns:
        by_ex['cart_rmse_wrist'] = (by_ex.get('cart_rmse_l_wrist', float('nan'))
                                         .fillna(by_ex.get('cart_rmse_r_wrist',
                                                            float('nan'))))

    # Salva CSV raw
    p = out_dir / 'results_by_exercise.csv'
    by_ex.to_csv(p, index=False)
    print(f'\n  Saved -> {p.name}')

    # Aggrega per (modalita x metodo) — media sui 5 esercizi
    metric_cols = [c for c in by_ex.columns
                   if c not in ('method', 'modality', 'exercise_num', 'exercise_type')]
    agg = (by_ex.groupby(['modality', 'method'])[metric_cols]
                 .mean()
                 .reset_index())

    if 'cart_rmse_wrist' not in agg.columns:
        agg['cart_rmse_wrist'] = (agg.get('cart_rmse_l_wrist', float('nan'))
                                      .fillna(agg.get('cart_rmse_r_wrist', float('nan'))))
    p = out_dir / 'results_aggregated.csv'
    agg.to_csv(p, index=False)
    print(f'  Saved -> {p.name}')

    # Plot
    print('\n  Generazione plot ...')
    PLOT_CANDIDATES = [
        ('cart_dtw',          'DTW (m)  (lower=better)',      'plot_dtw_modality.png'),
        ('cart_rmse_l_wrist', 'RMSE Lw (m)  (lower=better)', 'plot_rmse_lw_modality.png'),
        ('cart_rmse_r_wrist', 'RMSE Rw (m)  (lower=better)', 'plot_rmse_rw_modality.png'),
        ('cart_rmse_l_elbow', 'RMSE Le (m)  (lower=better)', 'plot_rmse_le_modality.png'),
        ('cart_rmse_r_elbow', 'RMSE Re (m)  (lower=better)', 'plot_rmse_re_modality.png'),
        ('cart_pearson_mean', 'Pearson cart  (higher=better)','plot_pearson_modality.png'),
        ('rmse_mean',         'RMSE joint (deg)(lower=better)','plot_rmse_joint_modality.png'),
    ]
    for metric, ylabel, fname in PLOT_CANDIDATES:
        if metric not in agg.columns or agg[metric].isna().all():
            continue
        _grouped_bar_modality(
            agg, metric, ylabel=ylabel,
            title=f'Modality Analysis -- {ylabel}',
            output_path=out_dir / fname)

    LINE_CANDIDATES = [
        ('cart_dtw',         'DTW (m)',           'plot_dtw_per_exercise.png'),
        ('cart_rmse_wrist',  'RMSE wrist (m)',    'plot_rmse_wrist_per_exercise.png'),
        ('cart_pearson_mean','Pearson cart',      'plot_pearson_per_exercise.png'),
    ]
    for metric, ylabel, fname in LINE_CANDIDATES:
        if metric not in by_ex.columns or by_ex[metric].isna().all():
            continue
        _line_per_exercise(
            by_ex, metric, ylabel,
            title=f'Per-Exercise -- {ylabel}  [all modalities x methods]',
            output_path=out_dir / fname)

    # Riepilogo a terminale
    print(f'\n{"="*72}')
    print(f'  RIEPILOGO -- split {split_name(n_demos)}')
    print(f'{"="*72}')
    print(f'  {"Modality":<8} {"Method":<16}  {"DTW(m/f)":>9}  '
          f'{"RMSE Lw":>8}  {"Pearson":>8}')
    print(f'  {"-"*60}')
    for mod in MODALITY_GROUPS:
        for method in ['Canonical', 'CanonicalShape'] + list(ARCHITECTURES.keys()):
            row = agg[(agg['modality'] == mod) & (agg['method'] == method)]
            if row.empty:
                continue
            r = row.iloc[0]
            dtw  = r.get('cart_dtw_norm',    np.nan)
            rmse = r.get('cart_rmse_l_wrist', np.nan)
            pear = r.get('cart_pearson_mean', np.nan)
            print(f'  {mod:<8} {method:<16}  {dtw:>9.4f}  '
                  f'{rmse:>8.4f}  {pear:>8.3f}')

    print(f'\n  Output -> {out_dir}')


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Analisi comparativa Stereo vs Mixed vs Mono.')
    parser.add_argument('--n-demos', type=int, default=55, choices=N_DEMOS_SPLITS,
                        help='Split da usare (default: 55).')
    args = parser.parse_args()
    run_modality_analysis(args.n_demos)
    print('\nDone.')


if __name__ == '__main__':
    main()