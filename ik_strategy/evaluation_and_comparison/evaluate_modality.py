'''
evaluate_modality.py
=============================================================================
Analisi comparativa tra modalità di acquisizione: Stereo vs Mixed vs Mono.

Risponde alla domanda: la complessità aggiuntiva della pipeline stereo porta
un beneficio misurabile? Le due sorgenti (stereo + mono) possono essere
combinate senza introdurre inconsistenze?

Per ogni modalità (Stereo: 001–005, Mixed: 011–015, Mono: 021–025) e per
ogni architettura BC, calcola le metriche su tutti gli esercizi disponibili,
aggrega per modalità e produce i plot di confronto.

Output (in data/evaluation_modality/):
    results_by_exercise.csv       — una riga per (esercizio × architettura)
    results_aggregated.csv        — una riga per (modalità × architettura)
    plot_dtw_modality.png
    plot_rmse_modality.png
    plot_pearson_modality.png
    plot_smoothness_modality.png
    plot_rmse_heatmap_<arch>.png  — uno per architettura
    plot_dtw_per_exercise.png
    plot_rmse_per_exercise.png
    plot_pearson_per_exercise.png

Uso standalone:
    py -m evaluation_and_comparison.evaluate_modality
    py -m evaluation_and_comparison.evaluate_modality --steps 200
'''

import argparse
from typing import Dict, List

import numpy as np
import pandas as pd

from utilities.config import DATA_ROOT

from evaluation_and_comparison._config  import (
    ARCHITECTURES, MODALITY_GROUPS, get_exercise_type,
)
from evaluation_and_comparison._io      import (
    load_baseline, load_canonical, load_bc_trajectory,
)
from evaluation_and_comparison._metrics import compute_metrics, aggregate_metrics
from evaluation_and_comparison._plots   import (
    plot_modality_grouped_bar, plot_modality_rmse_heatmap, plot_per_exercise_lines,
)


def run_modality_analysis() -> None:
    '''
    Per ogni modalità e ogni architettura, carica le traiettorie BC pre-generate
    da test_bc.py, calcola le metriche su tutti gli esercizi disponibili,
    aggrega per modalità e salva i risultati.

    Prerequisito: eseguire prima test_bc.py per ogni architettura e esercizio,
    in modo che bc_trajectory.csv sia presente in dataset/exercise_NNN/{arch}/.
    '''
    output_dir = DATA_ROOT / 'evaluation_modality'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*65}')
    print('  ANALISI MODALITY — Stereo vs Mixed vs Mono')
    print(f'{"="*65}')

    # ex_results[exercise_num][arch] = metrics
    ex_results: Dict[int, Dict[str, Dict]] = {}
    # agg[modality][arch] = metrics aggregati (media sui 5 esercizi della modalità)
    agg: Dict[str, Dict[str, Dict]] = {mod: {} for mod in MODALITY_GROUPS}
    csv_rows: List[Dict] = []

    for mod, ex_nums in MODALITY_GROUPS.items():
        print(f'\n{"─"*50}')
        print(f'  Modalità: {mod}')
        print(f'{"─"*50}')
        arch_all: Dict[str, List[Dict]] = {a: [] for a in ARCHITECTURES}

        for ex_num in ex_nums:
            exercise_name = f'exercise_{ex_num:03d}'
            dataset_dir   = DATA_ROOT / 'dataset' / exercise_name

            if not dataset_dir.is_dir():
                print(f'  [!] {exercise_name} non trovato, skip.')
                continue

            print(f'\n  Exercise {ex_num:03d}')
            baseline = load_baseline(dataset_dir)
            if baseline is None:
                continue

            canonical  = load_canonical(dataset_dir)
            ex_results.setdefault(ex_num, {})

            for arch_name in ARCHITECTURES:
                print(f'    [{arch_name}]', end=' ', flush=True)
                traj = load_bc_trajectory(dataset_dir, arch_name)
                if traj is None:
                    continue
                met = compute_metrics(traj, baseline, label=f'{arch_name}/{ex_num:03d}')
                ex_results[ex_num][arch_name] = met
                arch_all[arch_name].append(met)
                csv_rows.append({
                    'modality'     : mod,
                    'exercise_num' : ex_num,
                    'exercise_type': get_exercise_type(ex_num),
                    'architecture' : arch_name,
                    **{k: v for k, v in met.items() if not hasattr(v, '__len__')},
                })

        for arch_name, met_list in arch_all.items():
            if met_list:
                agg[mod][arch_name] = aggregate_metrics(met_list)

    # --- Salvataggio CSV ----------------------------------------------------
    if csv_rows:
        p = output_dir / 'results_by_exercise.csv'
        pd.DataFrame(csv_rows).to_csv(p, index=False)
        print(f'\n  Saved -> {p.name}')

    agg_rows = [
        {'modality': mod, 'architecture': arch,
         **{k: v for k, v in met.items() if not hasattr(v, '__len__')}}
        for mod, arch_mets in agg.items()
        for arch, met in arch_mets.items()
    ]
    if agg_rows:
        p = output_dir / 'results_aggregated.csv'
        pd.DataFrame(agg_rows).to_csv(p, index=False)
        print(f'  Saved -> {p.name}')

    # --- Plot aggregati per modalità ----------------------------------------
    print('\nGenerazione plot analisi modality ...')
    for metric, ylabel, fname in [
        ('dtw_distance', 'DTW distance (↓ meglio)',            'plot_dtw_modality.png'),
        ('rmse_mean',    'RMSE medio su active joints (°)',     'plot_rmse_modality.png'),
        ('pearson_mean', 'Pearson r medio  (↑ meglio)',         'plot_pearson_modality.png'),
        ('smoothness',   'Smoothness −mean(jerk²) (↑ meglio)', 'plot_smoothness_modality.png'),
    ]:
        plot_modality_grouped_bar(
            agg, metric, ylabel=ylabel,
            title=f'Modality Analysis — {ylabel}',
            output_path=output_dir / fname)

    # Heatmap RMSE per joint, una per architettura
    for arch_name in ARCHITECTURES:
        plot_modality_rmse_heatmap(
            agg, arch_name,
            output_dir / f'plot_rmse_heatmap_{arch_name}.png')

    # Line plot per-exercise (andamento al variare del tipo di esercizio)
    if ex_results:
        for metric, ylabel, fname in [
            ('dtw_distance', 'DTW distance',  'plot_dtw_per_exercise.png'),
            ('rmse_mean',    'RMSE (°)',       'plot_rmse_per_exercise.png'),
            ('pearson_mean', 'Pearson r',      'plot_pearson_per_exercise.png'),
        ]:
            plot_per_exercise_lines(
                ex_results, metric, ylabel,
                title=f'Per-Exercise — {ylabel}  [all modalities × architectures]',
                output_path=output_dir / fname)

    # --- Riepilogo a terminale ----------------------------------------------
    print(f'\n{"="*72}')
    print('  RIEPILOGO AGGREGATO PER MODALITÀ')
    print(f'{"="*72}')
    print(f'  {"Modality":<8} {"Arch":<14}  {"DTW":>8}  {"RMSE(°)":>8}  '
          f'{"Pearson":>8}  {"Smooth":>10}')
    print(f'  {"-"*66}')
    for mod in MODALITY_GROUPS:
        for arch in ARCHITECTURES:
            m = agg.get(mod, {}).get(arch)
            if m:
                print(f'  {mod:<8} {arch:<14}  '
                      f'{m["dtw_distance"]:>8.2f}  {m["rmse_mean"]:>8.2f}  '
                      f'{m["pearson_mean"]:>8.3f}  {m["smoothness"]:>10.4f}')
    print(f'{"="*72}')
    print(f'\n  Output → {output_dir}')


# ============================================================================
# Standalone
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Analisi comparativa Stereo vs Mixed vs Mono.')
    args = parser.parse_args()
    run_modality_analysis()
    print('\nDone.')


if __name__ == '__main__':
    main()