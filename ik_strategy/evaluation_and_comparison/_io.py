'''
_io.py
=============================================================================
I/O del modulo evaluation_and_comparison — solo lettura/scrittura CSV.

Le traiettorie BC sono generate da bc_approach/*/test_bc.py e salvate
come bc_trajectory.csv prima di eseguire la valutazione.

Funzioni pubbliche:
    load_baseline(dataset_dir)                   → np.ndarray | None
    load_canonical(dataset_dir)                  → np.ndarray | None
    load_bc_trajectory(dataset_dir, arch)        → np.ndarray | None
    load_human_demos(landmarks_root, exercise)   → list[np.ndarray]
    start_pose_from_canonical(canonical)         → np.ndarray
    save_results_csv(results, output_dir)
'''

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utilities.config import JOINT_COLS
from evaluation_and_comparison._config import N_JOINTS


# ============================================================================
# Helpers interni
# ============================================================================

def _load_joint_csv(path: Path, label: str = '') -> Optional[np.ndarray]:
    '''
    Carica un CSV con colonne JOINT_COLS e restituisce un array (T, 16).
    Fa padding con zero se alcune colonne mancano.
    '''
    if not path.exists():
        if label:
            print(f'  [!] {label}: {path.name} non trovato')
        return None
    df   = pd.read_csv(path)
    cols = [c for c in JOINT_COLS if c in df.columns]
    arr  = df[cols].dropna().values.astype(float)
    if len(cols) < N_JOINTS:
        padded = np.zeros((len(arr), N_JOINTS))
        for i, c in enumerate(cols):
            padded[:, JOINT_COLS.index(c)] = arr[:, i]
        arr = padded
    if label:
        print(f'  {label}: {len(arr)} frame')
    return arr


# ============================================================================
# Caricamento traiettorie
# ============================================================================

def load_baseline(dataset_dir: Path) -> Optional[np.ndarray]:
    return _load_joint_csv(dataset_dir / 'baseline.csv', 'Baseline')


def load_canonical(dataset_dir: Path) -> Optional[np.ndarray]:
    return _load_joint_csv(dataset_dir / 'canonical.csv', 'Canonical')


def load_bc_trajectory(dataset_dir: Path, arch: str) -> Optional[np.ndarray]:
    '''
    Carica la traiettoria BC pre-generata da test_bc.py.

    Path atteso: dataset_dir / arch / 'bc_trajectory.csv'
    Esempio:     data/dataset/exercise_021/MLP/bc_trajectory.csv

    Se il file non esiste, stampa un avviso e restituisce None.
    Esegui prima il test_bc.py corrispondente per generarlo.
    '''
    path = dataset_dir / arch / 'bc_trajectory.csv'
    return _load_joint_csv(path, label=arch)


def load_human_demos(landmarks_root: Path,
                      exercise_num: int,
                      n_demos: int = 55) -> List[np.ndarray]:
    '''
    Carica i joint_ik.csv delle demo umane per un dato esercizio,
    limitandosi ai primi K soggetti (stessi usati per il training dello split).

    n_demos : 10 -> 2 soggetti, 25 -> 5, 55 -> tutti 11
    '''
    from utilities.split_utils import select_subjects

    exercise_name = f'exercise_{exercise_num:03d}'
    all_subj_dirs = sorted(landmarks_root.glob('subject_*'))
    selected      = select_subjects(all_subj_dirs, n_demos)
    selected_set  = {d.name for d in selected}

    demos = []
    for subj_dir in sorted(landmarks_root.glob('subject_*')):
        if subj_dir.name not in selected_set:
            continue
        exer_dir = subj_dir / exercise_name
        if not exer_dir.is_dir():
            continue
        for video_dir in sorted(exer_dir.glob('video_*')):
            arr = _load_joint_csv(video_dir / 'joint_ik.csv')
            if arr is not None and len(arr) >= 2:
                demos.append(arr)
    print(f'  Human demos caricate: {len(demos)} '
          f'({len(selected)} soggetti, split n_{n_demos:02d})')
    return demos


def start_pose_from_canonical(canonical: Optional[np.ndarray]) -> np.ndarray:
    '''Primo frame della canonical come start pose, o array di zeri se assente.'''
    if canonical is not None and len(canonical) > 0:
        return canonical[0].copy().astype(np.float32)
    return np.zeros(N_JOINTS, dtype=np.float32)


# ============================================================================
# Salvataggio risultati
# ============================================================================

def save_results_csv(results: Dict, output_dir: Path) -> None:
    '''
    Salva tre CSV nella cartella output_dir:
      results_summary.csv      — una riga per metodo, tutte le metriche scalari
      results_per_joint.csv    — una riga per (metodo × joint), metriche joint-space
      results_per_endpoint.csv — una riga per (metodo × endpoint), metriche cartesiane
    '''
    output_dir.mkdir(parents=True, exist_ok=True)

    ALL_CART_SCALAR_KEYS = [
        'cart_dtw',
        'cart_rmse_r_wrist', 'cart_rmse_l_wrist',
        'cart_rmse_r_elbow', 'cart_rmse_l_elbow',
        'cart_peak_r_wrist', 'cart_peak_l_wrist',
        'cart_pearson_mean', 'cart_smoothness',
    ]
    # Salva solo le chiavi con almeno un valore non-NaN tra i metodi
    import math
    CART_SCALAR_KEYS = [
        k for k in ALL_CART_SCALAR_KEYS
        if any(not math.isnan(r.get(k, float('nan'))) for r in results.values())
    ]

    # --- results_summary.csv ------------------------------------------------
    summary = []
    for m, r in results.items():
        rw = r.get('cart_rmse_r_wrist', float('nan'))
        lw = r.get('cart_rmse_l_wrist', float('nan'))
        re = r.get('cart_rmse_r_elbow', float('nan'))
        le = r.get('cart_rmse_l_elbow', float('nan'))
        row = {
            'method'           : m,
            'dtw_distance'     : r.get('dtw_distance',     float('nan')),
            'rmse_mean_deg'    : r.get('rmse_mean',        float('nan')),
            'peak_error_mean'  : r.get('peak_error_mean',  float('nan')),
            'pearson_mean'     : r.get('pearson_mean',     float('nan')),
            'smoothness'       : r.get('smoothness',       float('nan')),
            # cart metriche braccio attivo (combinato)
            'cart_rmse_wrist'  : lw if not math.isnan(lw) else rw,
            'cart_rmse_elbow'  : le if not math.isnan(le) else re,
        }
        for k in CART_SCALAR_KEYS:
            row[k] = r.get(k, float('nan'))
        summary.append(row)
    p = output_dir / 'results_summary.csv'
    pd.DataFrame(summary).to_csv(p, index=False)
    print(f'  Saved -> {p.name}')

    # --- results_per_joint.csv ----------------------------------------------
    per_joint = [
        {
            'method'    : m,
            'joint'     : jname,
            'rmse_deg'  : r['rmse_per_joint'][j],
            'peak_error': r['peak_error_per_joint'][j],
            'pearson'   : r['pearson_per_joint'][j],
        }
        for m, r in results.items()
        if 'rmse_per_joint' in r
        for j, jname in enumerate(JOINT_COLS)
    ]
    if per_joint:
        p = output_dir / 'results_per_joint.csv'
        pd.DataFrame(per_joint).to_csv(p, index=False)
        print(f'  Saved -> {p.name}')

    # --- results_per_endpoint.csv -------------------------------------------
    ENDPOINTS = {
        'r_wrist': ('cart_pearson_rw', 'cart_rmse_r_wrist', 'cart_peak_r_wrist'),
        'l_wrist': ('cart_pearson_lw', 'cart_rmse_l_wrist', 'cart_peak_l_wrist'),
    }
    COORD_LABELS = ['x', 'y', 'z']
    per_ep = []
    for m, r in results.items():
        if 'cart_dtw' not in r:
            continue
        for ep, (pearson_key, rmse_key, peak_key) in ENDPOINTS.items():
            pearson_vec = r.get(pearson_key, np.full(3, float('nan')))
            for c, coord in enumerate(COORD_LABELS):
                per_ep.append({
                    'method'   : m,
                    'endpoint' : ep,
                    'coord'    : coord,
                    'pearson'  : pearson_vec[c],
                    'rmse_m'   : r.get(rmse_key,  float('nan')),
                    'peak_m'   : r.get(peak_key,  float('nan')),
                })
    if per_ep:
        p = output_dir / 'results_per_endpoint.csv'
        pd.DataFrame(per_ep).to_csv(p, index=False)
        print(f'  Saved -> {p.name}')