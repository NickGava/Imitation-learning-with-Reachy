'''
evaluate.py
=============================================================================
Valutazione completa del sistema di imitazione (LfD) su Reachy.

Architetture valutate: MLP · GRU · Transformer
Modalità di acquisizione codificate nel numero di esercizio:
    001–005  →  Stereo
    011–015  →  Mixed  (stereo + mono)
    021–025  →  Mono

Tutte le metriche sono calcolate rispetto alla baseline (hardcoded ground-truth):
    DTW distance        : distanza totale dopo allineamento temporale
    RMSE per joint      : errore medio in gradi su active joints, dopo DTW
    Peak angle error    : |max(baseline) − max(generated)| per joint
    Pearson correlation : struttura temporale (−1 to 1), indipendente da scala
    Smoothness          : −mean(jerk²), jerk = derivata seconda (higher = smoother)
    Velocity profile    : analisi visiva (bell-shape = moto naturale umano)

Modalità di esecuzione
──────────────────────
  # Valutazione singolo/i esercizio/i (tutte e 3 le architetture)
  python evaluate.py --exercise 1
  python evaluate.py --exercise 1 11 21

  # Analisi modality: confronto Stereo vs Mixed vs Mono
  python evaluate.py --analysis modality

  # Tutto: tutti gli esercizi trovati + analisi modality
  python evaluate.py --all

Output
──────
  _data/dataset/exercise_NNN/evaluation/
      results_summary.csv
      results_per_joint.csv
      plot_degradation_chain.png
      plot_rmse_per_joint.png
      plot_pearson_per_joint.png
      plot_smoothness.png
      plot_velocity_<joint>.png

  _data/evaluation_modality/
      results_by_exercise_type.csv
      results_aggregated.csv
      plot_dtw_modality.png
      plot_rmse_modality.png
      plot_pearson_modality.png
      plot_smoothness_modality.png
      plot_rmse_heatmap_<arch>.png
      plot_dtw_per_exercise.png
      plot_rmse_per_exercise.png
      plot_pearson_per_exercise.png
'''

import argparse
import pickle
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from tslearn.metrics import dtw_path

from utilities.config import DATA_ROOT, JOINT_COLS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_JOINTS     = len(JOINT_COLS)   # 16
VELOCITY_LAG = 5
VEL_CLIP     = 10.0

# Architetture e relative sottocartelle nel dataset
ARCHITECTURES: Dict[str, str] = {
    'MLP'        : 'MLP',
    'GRU'        : 'GRU',
    'Transformer': 'Transformer',
}

# Raggruppamento esercizi per modalità di acquisizione
MODALITY_GROUPS: Dict[str, List[int]] = {
    'Stereo': list(range(1,  6)),   # 001–005
    'Mixed' : list(range(11, 16)),  # 011–015
    'Mono'  : list(range(21, 26)),  # 021–025
}


def get_modality(exercise_num: int) -> str:
    if  1 <= exercise_num <=  5: return 'Stereo'
    if 11 <= exercise_num <= 15: return 'Mixed'
    if 21 <= exercise_num <= 25: return 'Mono'
    return 'Unknown'


def get_exercise_type(exercise_num: int) -> int:
    '''Tipo di esercizio (1–5), indipendente dalla modalità.'''
    if  1 <= exercise_num <=  5: return exercise_num
    if 11 <= exercise_num <= 15: return exercise_num - 10
    if 21 <= exercise_num <= 25: return exercise_num - 20
    return exercise_num


# Short labels per i 16 joint (ordine = JOINT_COLS)
JOINT_LABELS = [
    'r_sh_p', 'r_sh_r', 'r_aw', 'r_el_p', 'r_fw_y', 'r_wr_p', 'r_wr_r', 'r_gr',
    'l_sh_p', 'l_sh_r', 'l_aw', 'l_el_p', 'l_fw_y', 'l_wr_p', 'l_wr_r', 'l_gr',
]

# Active joints: shoulder_pitch/roll, arm_yaw, elbow_pitch per braccio.
# Wrist e gripper sono fissi a zero, esclusi dalle medie scalari e dai plot.
ACTIVE_IDX    = [0, 1, 2, 3, 8, 9, 10, 11]
ACTIVE_LABELS = ['r_sh_p', 'r_sh_r', 'r_aw', 'r_el_p',
                 'l_sh_p', 'l_sh_r', 'l_aw', 'l_el_p']

# Joint scelti per il velocity profile
VELOCITY_JOINTS = {'r_sh_p': 0, 'l_sh_r': 9}

# Palette colori
PALETTE = {
    'Human demos': '#e74c3c',
    'Canonical'  : '#e67e22',
    'MLP'        : '#3498db',
    'GRU'        : '#9b59b6',
    'Transformer': '#1abc9c',
    'Stereo'     : '#2ecc71',
    'Mixed'      : '#f39c12',
    'Mono'       : '#8e44ad',
}

# ============================================================================
# Definizioni dei modelli (devono corrispondere ai training script)
# ============================================================================

class BCPolicyMLP(nn.Module):
    '''MLP policy: [q, dq] (32) → Δq (16). Stateless.'''
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_size: int = 256, n_layers: int = 2, **_):
        super().__init__()
        layers, in_f = [], input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_f, hidden_size), nn.ReLU()]
            in_f = hidden_size
        layers.append(nn.Linear(in_f, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BCPolicyGRU(nn.Module):
    '''
    GRU policy: processa [q, dq] online, un frame per volta, mantenendo h.
    Input : (1, 1, input_dim)  →  Output: (1, output_dim), h aggiornato.
    '''
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_size: int = 256, n_layers: int = 2, **_):
        super().__init__()
        self.gru         = nn.GRU(input_dim, hidden_size, n_layers, batch_first=True)
        self.fc          = nn.Linear(hidden_size, output_dim)
        self.n_layers    = n_layers
        self.hidden_size = hidden_size

    def forward(self, x: torch.Tensor,
                h: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        out, h = self.gru(x, h)
        return self.fc(out[:, -1, :]), h


class BCPolicyTransformer(nn.Module):
    '''
    Transformer policy con finestra causale di stati passati.
    Input : (1, window_size, input_dim)  →  Output: (1, output_dim).
    '''
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_size: int = 256, n_layers: int = 2,
                 n_heads: int = 4, window_size: int = 10,
                 dropout: float = 0.1, **_):
        super().__init__()
        self.window_size = window_size
        self.input_proj  = nn.Linear(input_dim, hidden_size)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.fc          = nn.Linear(hidden_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


_MODEL_CLASSES = {
    'MLP'        : BCPolicyMLP,
    'GRU'        : BCPolicyGRU,
    'Transformer': BCPolicyTransformer,
}


def _resolve_ckpt_dim(ckpt: dict, keys: List[str], default: int) -> int:
    '''Cerca la dimensione provando più nomi di chiave comuni, con fallback su default.'''
    for k in keys:
        if k in ckpt:
            return int(ckpt[k])
    return default


def _infer_dims_from_state_dict(ckpt: dict, model_type: str
                                 ) -> Tuple[int, int, int, int]:
    '''
    Ricava (input_dim, output_dim, hidden_size, n_layers) direttamente dai
    tensori nel model_state, quando le chiavi esplicite non sono presenti.
    Funziona per MLP, GRU e Transformer con le architetture definite sopra.
    '''
    sd = ckpt.get('model_state', {})

    if model_type == 'GRU':
        # Prima weight_ih_l0: (3*hidden, input)
        wih = sd.get('gru.weight_ih_l0')
        whh = sd.get('gru.weight_hh_l0')
        fc  = sd.get('fc.weight')
        hidden_size = int(wih.shape[0] / 3)  if wih is not None else 256
        input_dim   = int(wih.shape[1])       if wih is not None else 32
        output_dim  = int(fc.shape[0])        if fc  is not None else N_JOINTS
        # Conta i layer: weight_ih_lN presente per N = 0 … n_layers-1
        n_layers = sum(1 for k in sd if k.startswith('gru.weight_ih_l'))
        return input_dim, output_dim, hidden_size, max(n_layers, 1)

    elif model_type == 'Transformer':
        proj = sd.get('input_proj.weight')
        fc   = sd.get('fc.weight')
        input_dim   = int(proj.shape[1]) if proj is not None else 32
        hidden_size = int(proj.shape[0]) if proj is not None else 256
        output_dim  = int(fc.shape[0])   if fc   is not None else N_JOINTS
        n_layers = sum(1 for k in sd
                       if k.startswith('transformer.layers.') and k.endswith('.self_attn.in_proj_weight'))
        return input_dim, output_dim, hidden_size, max(n_layers, 1)

    else:  # MLP — net.0.weight = (hidden, input), net[-1].weight = (output, hidden)
        first_w = next((v for k, v in sd.items()
                        if k.endswith('.weight') and 'net' in k), None)
        last_w  = None
        for k, v in sd.items():
            if k.endswith('.weight') and 'net' in k:
                last_w = v
        input_dim   = int(first_w.shape[1]) if first_w is not None else 32
        hidden_size = int(first_w.shape[0]) if first_w is not None else 256
        output_dim  = int(last_w.shape[0])  if last_w  is not None else N_JOINTS
        n_layers = sum(1 for k in sd if k.endswith('.weight') and 'net' in k) - 1
        return input_dim, output_dim, hidden_size, max(n_layers, 1)


def _build_model(ckpt: dict) -> nn.Module:
    '''
    Istanzia e carica il modello dal checkpoint.

    Strategia (in ordine):
    1. Legge le chiavi esplicite nel checkpoint (state_dim / input_dim / input_size,
       action_dim / output_dim / output_size, ecc.) — nomi diversi usati da
       training script diversi.
    2. Se mancano, inferisce le dimensioni direttamente dai tensori nel model_state.
    3. Stampa un riepilogo per facilitare il debug.
    '''
    model_type = ckpt.get('model_type', 'MLP')

    # --- Dimensioni: prova nomi alternativi, poi fallback sull'inferenza ---
    input_dim = _resolve_ckpt_dim(
        ckpt, ['state_dim', 'input_dim', 'input_size'], default=-1)
    output_dim = _resolve_ckpt_dim(
        ckpt, ['action_dim', 'output_dim', 'output_size'], default=-1)
    hidden_size = _resolve_ckpt_dim(
        ckpt, ['hidden_size', 'hidden_dim'], default=-1)
    n_layers = _resolve_ckpt_dim(
        ckpt, ['n_layers', 'num_layers'], default=-1)

    if input_dim < 0 or output_dim < 0:
        # Inferenza dai tensori
        i, o, h, l = _infer_dims_from_state_dict(ckpt, model_type)
        if input_dim  < 0: input_dim  = i
        if output_dim < 0: output_dim = o
        if hidden_size < 0: hidden_size = h
        if n_layers   < 0: n_layers   = l
        print(f'      [dim inferred from weights] '
              f'in={input_dim} out={output_dim} '
              f'hidden={hidden_size} layers={n_layers}')
    else:
        if hidden_size < 0: hidden_size = 256
        if n_layers    < 0: n_layers    = 2

    cls = _MODEL_CLASSES.get(model_type, BCPolicyMLP)
    model = cls(
        input_dim   = input_dim,
        output_dim  = output_dim,
        hidden_size = hidden_size,
        n_layers    = n_layers,
        n_heads     = ckpt.get('n_heads',     4),
        window_size = ckpt.get('window_size', 10),
        dropout     = ckpt.get('dropout',     0.1),
    )
    try:
        model.load_state_dict(ckpt['model_state'])
    except RuntimeError as e:
        # Mismatch architettura: stampa info utili e rilancia
        print(f'\n  [!] load_state_dict fallito per {model_type}.')
        print(f'      Chiavi checkpoint (non-tensor): '
              f'{ {k: v for k, v in ckpt.items() if not hasattr(v, "shape")} }')
        print(f'      Chiavi model_state: {list(ckpt.get("model_state", {}).keys())[:8]} ...')
        raise RuntimeError(
            f'Architettura {model_type} non compatibile con il checkpoint. '
            f'Verifica che i model class in evaluate.py corrispondano ai '
            f'training script. Dettaglio originale:\n{e}'
        ) from e
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Caricamento ensemble k-fold
# ---------------------------------------------------------------------------
def load_ensemble(model_dir: Path) -> List[Tuple]:
    '''
    Carica tutti i fold (bc_model_fold_N.pth + scaler_fold_N.pkl) da model_dir.
    Ritorna lista di (model, scaler, model_type, ckpt).
    '''
    fold_idx, ensemble = 0, []
    while True:
        mp = model_dir / f'bc_model_fold_{fold_idx}.pth'
        sp = model_dir / f'scaler_fold_{fold_idx}.pkl'
        if not mp.exists():
            break
        ckpt  = torch.load(mp, map_location='cpu', weights_only=False)

        # Debug: mostra chiavi non-tensor al primo fold
        if fold_idx == 0:
            meta = {k: v for k, v in ckpt.items()
                    if not hasattr(v, 'shape') and k != 'model_state'}
            print(f'    Checkpoint keys (fold 0): {meta}')

        model = _build_model(ckpt)
        with open(sp, 'rb') as f:
            scaler = pickle.load(f)
        mtype = ckpt.get('model_type', 'MLP')
        ensemble.append((model, scaler, mtype, ckpt))
        print(f'    Fold {fold_idx} [{mtype}]  '
              f'val_loss={ckpt.get("val_loss", float("nan")):.6f}')
        fold_idx += 1
    return ensemble


# ============================================================================
# Generazione traiettoria BC (autoregressive, ensemble k-fold)
# ============================================================================
def generate_bc_trajectory(model_dir: Path,
                            start_pose: np.ndarray,
                            n_steps: int) -> Optional[np.ndarray]:
    '''
    Loop autoregressivo con ensemble k-fold, compatibile con MLP / GRU / Transformer.

    - MLP        : stateless, state = [q, dq]
    - GRU        : mantiene hidden state h per fold
    - Transformer: usa rolling window di stati normalizzati

    Returns (n_steps, 16) float32 oppure None se nessun modello trovato.
    '''
    if not model_dir.is_dir():
        print(f'    [BC] Cartella non trovata: {model_dir.name}')
        return None

    ensemble = load_ensemble(model_dir)
    if not ensemble:
        print(f'    [BC] Nessun fold trovato in {model_dir.name}')
        return None

    model_type  = ensemble[0][2]
    window_size = ensemble[0][3].get('window_size', 10)
    print(f'    [BC/{model_type}] {len(ensemble)} fold(s)')

    start_q   = start_pose.astype(np.float32)
    q_history = [start_q.copy() for _ in range(VELOCITY_LAG)]
    q_current = start_q.copy()
    q_traj    = np.zeros((n_steps, N_JOINTS), dtype=np.float32)

    gru_hiddens   = [None] * len(ensemble)
    trans_windows = [deque(maxlen=window_size) for _ in ensemble]

    for step in range(n_steps):
        velocity = np.clip(q_current - q_history[0], -VEL_CLIP, VEL_CLIP)
        state    = np.concatenate([q_current, velocity]).astype(np.float32)

        deltas = []
        for idx, (model, scaler, mtype, _) in enumerate(ensemble):
            state_norm = scaler.transform([state])   # (1, state_dim)

            if mtype == 'GRU':
                x = torch.tensor(state_norm, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    delta_t, gru_hiddens[idx] = model(x, gru_hiddens[idx])
                delta = delta_t.squeeze(0).numpy()

            elif mtype == 'Transformer':
                trans_windows[idx].append(state_norm[0])
                win = list(trans_windows[idx])
                if len(win) < window_size:
                    win = [win[0]] * (window_size - len(win)) + win
                x = torch.tensor([win], dtype=torch.float32)
                with torch.no_grad():
                    delta = model(x).numpy()[0]

            else:  # MLP
                x = torch.tensor(state_norm, dtype=torch.float32)
                with torch.no_grad():
                    delta = model(x).numpy()[0]

            deltas.append(delta)

        q_new = q_current + np.mean(deltas, axis=0)
        q_history.pop(0)
        q_history.append(q_current.copy())
        q_current    = q_new
        q_traj[step] = q_new

    return q_traj


# ============================================================================
# Caricamento dati
# ============================================================================
def _load_joint_csv(path: Path, label: str = '') -> Optional[np.ndarray]:
    if not path.exists():
        if label:
            print(f'  [!] {label}: {path.name} non trovato')
        return None
    df   = pd.read_csv(path)
    cols = [c for c in JOINT_COLS if c in df.columns]
    arr  = df[cols].dropna().values.astype(float)
    if len(cols) < N_JOINTS:          # padding con zero per joint mancanti
        padded = np.zeros((len(arr), N_JOINTS))
        for i, c in enumerate(cols):
            padded[:, JOINT_COLS.index(c)] = arr[:, i]
        arr = padded
    if label:
        print(f'  {label}: {len(arr)} frame')
    return arr


def load_baseline(dataset_dir: Path) -> Optional[np.ndarray]:
    return _load_joint_csv(dataset_dir / 'baseline.csv', 'Baseline')


def load_canonical(dataset_dir: Path) -> Optional[np.ndarray]:
    return _load_joint_csv(dataset_dir / 'canonical.csv', 'Canonical')


def load_human_demos(landmarks_root: Path, exercise_num: int) -> List[np.ndarray]:
    exercise_name = f'exercise_{exercise_num:03d}'
    demos = []
    for subj_dir in sorted(landmarks_root.glob('subject_*')):
        exer_dir = subj_dir / exercise_name
        if not exer_dir.is_dir():
            continue
        for video_dir in sorted(exer_dir.glob('video_*')):
            arr = _load_joint_csv(video_dir / 'joint_ik.csv')
            if arr is not None and len(arr) >= 2:
                demos.append(arr)
    print(f'  Human demos: {len(demos)}')
    return demos


def _start_pose_from_canonical(canonical: Optional[np.ndarray]) -> np.ndarray:
    if canonical is not None and len(canonical) > 0:
        return canonical[0].copy().astype(np.float32)
    return np.zeros(N_JOINTS, dtype=np.float32)


# ============================================================================
# Calcolo metriche
# ============================================================================
def compute_metrics(generated: np.ndarray, reference: np.ndarray,
                    label: str = '') -> Dict:
    '''
    Tutte le metriche tra traiettoria generata e reference (baseline).
    Le medie scalari sono calcolate su ACTIVE_IDX (8 joint mobili).
    '''
    path, dtw_dist = dtw_path(generated, reference)
    gen_al = generated[[p[0] for p in path]]
    ref_al = reference[[p[1] for p in path]]

    rmse_pj      = np.sqrt(np.mean((gen_al - ref_al) ** 2, axis=0))
    rmse_mean    = float(np.mean(rmse_pj[ACTIVE_IDX]))

    peak_pj      = np.abs(np.max(reference, axis=0) - np.max(generated, axis=0))
    peak_mean    = float(np.mean(peak_pj[ACTIVE_IDX]))

    pearson_pj   = np.array([pearsonr(gen_al[:, j], ref_al[:, j])[0]
                              for j in range(N_JOINTS)])
    pearson_pj   = np.nan_to_num(pearson_pj, nan=0.0)
    pearson_mean = float(np.mean(pearson_pj[ACTIVE_IDX]))

    if len(generated) > 2:
        jerk       = np.diff(generated[:, ACTIVE_IDX], n=2, axis=0)
        smoothness = float(-np.mean(jerk ** 2))
    else:
        smoothness = float('nan')

    if label:
        print(f'  [{label:<16}]  DTW={dtw_dist:>9.2f}  '
              f'RMSE={rmse_mean:>6.2f}°  Peak={peak_mean:>6.2f}°  '
              f'r={pearson_mean:>6.3f}  Smooth={smoothness:>9.4f}')

    return {
        'dtw_distance'         : float(dtw_dist),
        'rmse_mean'            : rmse_mean,
        'rmse_per_joint'       : rmse_pj,
        'peak_error_mean'      : peak_mean,
        'peak_error_per_joint' : peak_pj,
        'pearson_mean'         : pearson_mean,
        'pearson_per_joint'    : pearson_pj,
        'smoothness'           : smoothness,
    }


def _aggregate_metrics(metrics_list: List[Dict]) -> Dict:
    scalar_keys = ['dtw_distance', 'rmse_mean', 'peak_error_mean',
                   'pearson_mean', 'smoothness']
    arr_keys    = ['rmse_per_joint', 'peak_error_per_joint', 'pearson_per_joint']
    out = {k: float(np.nanmean([m[k] for m in metrics_list])) for k in scalar_keys}
    out.update({k: np.nanmean([m[k] for m in metrics_list], axis=0) for k in arr_keys})
    return out


# ============================================================================
# Salvataggio CSV
# ============================================================================
def _save_results_csv(results: Dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = [{'method': m,
                'dtw_distance'   : r['dtw_distance'],
                'rmse_mean_deg'  : r['rmse_mean'],
                'peak_error_mean': r['peak_error_mean'],
                'pearson_mean'   : r['pearson_mean'],
                'smoothness'     : r['smoothness']}
               for m, r in results.items()]
    p = output_dir / 'results_summary.csv'
    pd.DataFrame(summary).to_csv(p, index=False)
    print(f'  Saved -> {p.name}')

    per_joint = [{'method': m, 'joint': jname,
                  'rmse_deg'  : r['rmse_per_joint'][j],
                  'peak_error': r['peak_error_per_joint'][j],
                  'pearson'   : r['pearson_per_joint'][j]}
                 for m, r in results.items()
                 for j, jname in enumerate(JOINT_COLS)]
    p = output_dir / 'results_per_joint.csv'
    pd.DataFrame(per_joint).to_csv(p, index=False)
    print(f'  Saved -> {p.name}')


# ============================================================================
# Plot — singolo esercizio
# ============================================================================
def _plot_degradation_chain(results: Dict, output_dir: Path,
                             exercise_num: int) -> None:
    CHAIN_ORDER = ['Human demos', 'Canonical', 'MLP', 'GRU', 'Transformer']
    methods  = [m for m in CHAIN_ORDER if m in results] + \
               [m for m in results   if m not in CHAIN_ORDER]
    dtw_vals = [results[m]['dtw_distance'] for m in methods]
    colors   = [PALETTE.get(m, '#95a5a6') for m in methods]

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.8), 5))
    bars = ax.bar(methods, dtw_vals, color=colors, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, dtw_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(dtw_vals) * 0.012,
                f'{val:.1f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.set_ylabel('DTW distance vs baseline  (↓ meglio)', fontsize=11)
    ax.set_title(f'Degradation Chain — Exercise {exercise_num:03d} '
                 f'[{get_modality(exercise_num)}]\n'
                 'Expected: MLP / GRU / Transformer ≤ Canonical < Human demos',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(dtw_vals) * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    p = output_dir / 'plot_degradation_chain.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {p.name}')


def _grouped_bar(ax, methods_dict: Dict, arr_key: str,
                 active_only: bool = True) -> None:
    '''Helper: bar chart raggruppato su active joints o tutti.'''
    idx    = ACTIVE_IDX    if active_only else list(range(N_JOINTS))
    labels = ACTIVE_LABELS if active_only else JOINT_LABELS
    x      = np.arange(len(labels))
    width  = 0.75 / max(len(methods_dict), 1)
    off0   = -(len(methods_dict) - 1) / 2 * width
    for i, (m, res) in enumerate(methods_dict.items()):
        ax.bar(x + off0 + i * width, res[arr_key][idx], width,
               label=m, color=PALETTE.get(m, f'C{i}'),
               edgecolor='white', linewidth=0.8, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)


def _plot_rmse_per_joint(results: Dict, output_dir: Path,
                          title_suffix: str = '') -> None:
    methods = {k: v for k, v in results.items() if k != 'Human demos'}
    if not methods:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    _grouped_bar(ax, methods, 'rmse_per_joint')
    ax.set_ylabel('RMSE (gradi)', fontsize=11)
    ax.set_title(f'RMSE per Joint vs Baseline{title_suffix}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    p = output_dir / 'plot_rmse_per_joint.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {p.name}')


def _plot_pearson_per_joint(results: Dict, output_dir: Path,
                             title_suffix: str = '') -> None:
    methods = {k: v for k, v in results.items() if k != 'Human demos'}
    if not methods:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    _grouped_bar(ax, methods, 'pearson_per_joint')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylim(-1.05, 1.15)
    ax.set_ylabel('Pearson r  (↑ meglio)', fontsize=11)
    ax.set_title(f'Pearson Correlation per Joint{title_suffix}\n'
                 '(struttura temporale, DTW-aligned)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    p = output_dir / 'plot_pearson_per_joint.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {p.name}')


def _plot_smoothness(results: Dict, output_dir: Path,
                     title_suffix: str = '') -> None:
    methods = list(results.keys())
    smooth  = [results[m]['smoothness'] for m in methods]
    colors  = [PALETTE.get(m, f'C{i}') for i, m in enumerate(methods)]
    ymax    = max((s for s in smooth if not np.isnan(s)), default=1.0)

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.8), 5))
    bars = ax.bar(methods, smooth, color=colors, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, smooth):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    max(val, 0) + abs(ymax) * 0.02,
                    f'{val:.4f}', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
    ax.set_ylabel('Smoothness: −mean(jerk²)  (↑ più fluido)', fontsize=10)
    ax.set_title(f'Smoothness{title_suffix}\n(misura intrinseca — non vs baseline)',
                 fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    p = output_dir / 'plot_smoothness.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {p.name}')


def _plot_velocity_profile(trajs: Dict[str, Optional[np.ndarray]],
                            output_dir: Path) -> None:
    for joint_short, joint_idx in VELOCITY_JOINTS.items():
        items = [(lbl, t) for lbl, t in trajs.items()
                 if t is not None and len(t) > 2]
        if not items:
            continue
        n = len(items)
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=False)
        if n == 1:
            axes = [axes]
        for ax, (lbl, traj) in zip(axes, items):
            vel = np.diff(traj[:, joint_idx])
            ax.plot(vel, color=PALETTE.get(lbl, '#7f8c8d'), linewidth=1.5)
            ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
            ax.set_title(lbl, fontsize=11, fontweight='bold')
            ax.set_xlabel('Frame', fontsize=9)
            ax.set_ylabel('Vel. angolare (°/frame)', fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(alpha=0.3)
        fig.suptitle(f'Velocity Profile — {joint_short}\n'
                     '(bell-shaped ≈ moto umano naturale; misura intrinseca)',
                     fontsize=11, fontweight='bold')
        fig.tight_layout()
        p = output_dir / f'plot_velocity_{joint_short}.png'
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved -> {p.name}')


# ============================================================================
# Plot — Analisi Modality
# ============================================================================
def _plot_modality_grouped_bar(agg: Dict[str, Dict[str, Dict]],
                                metric: str, ylabel: str, title: str,
                                output_path: Path) -> None:
    '''
    Grouped bar: asse X = architetture, colori = modalità.
    agg[modality][arch] = metrics_dict
    '''
    archs   = list(ARCHITECTURES.keys())
    mods    = list(MODALITY_GROUPS.keys())
    x       = np.arange(len(archs))
    width   = 0.25
    off0    = -(len(mods) - 1) / 2 * width

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, mod in enumerate(mods):
        vals = [agg.get(mod, {}).get(a, {}).get(metric, np.nan) for a in archs]
        bars = ax.bar(x + off0 + i * width, vals, width,
                      label=mod, color=PALETTE.get(mod, f'C{i}'),
                      edgecolor='white', linewidth=1.0, alpha=0.88)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + abs(val) * 0.02,
                        f'{val:.2f}', ha='center', va='bottom', fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(archs, fontsize=11)
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


def _plot_modality_rmse_heatmap(agg: Dict[str, Dict[str, Dict]],
                                 arch: str, output_path: Path) -> None:
    mods = list(MODALITY_GROUPS.keys())
    data = np.array([
        agg.get(mod, {}).get(arch, {}).get(
            'rmse_per_joint', np.full(N_JOINTS, np.nan)
        )[ACTIVE_IDX]
        for mod in mods
    ])
    finite = data[np.isfinite(data)]
    vmax   = float(np.nanmax(finite)) if len(finite) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(10, 2.0 + len(mods) * 0.7))
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label='RMSE (gradi)')
    ax.set_xticks(range(len(ACTIVE_LABELS)))
    ax.set_xticklabels(ACTIVE_LABELS, rotation=40, ha='right', fontsize=9)
    ax.set_yticks(range(len(mods)))
    ax.set_yticklabels(mods, fontsize=10)
    for i in range(len(mods)):
        for j in range(len(ACTIVE_IDX)):
            val = data[i, j]
            txt = f'{val:.1f}' if not np.isnan(val) else '–'
            ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                    color='white' if val > vmax * 0.6 else 'black')
    ax.set_title(f'RMSE per Joint — {arch} — Modality Comparison',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {output_path.name}')


def _plot_per_exercise_lines(ex_results: Dict[int, Dict[str, Dict]],
                              metric: str, ylabel: str, title: str,
                              output_path: Path) -> None:
    '''
    Line plot: asse X = tipo esercizio (1–5), una linea per (arch × modalità).
    '''
    ex_types = sorted({get_exercise_type(n) for n in ex_results})
    fig, ax  = plt.subplots(figsize=(10, 5))

    ls_map = {'Stereo': '-', 'Mixed': '--', 'Mono': ':'}
    for mod, ex_nums in MODALITY_GROUPS.items():
        for arch in ARCHITECTURES:
            vals = []
            for et in ex_types:
                num = next((n for n in ex_nums
                            if get_exercise_type(n) == et), None)
                m   = ex_results.get(num, {}).get(arch)
                vals.append(m[metric] if m else np.nan)
            ax.plot(ex_types, vals,
                    linestyle=ls_map[mod],
                    color=PALETTE.get(arch, '#333'),
                    marker='o', linewidth=1.8, markersize=6,
                    label=f'{arch} [{mod}]', alpha=0.85)

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


# ============================================================================
# Stampa tabella riassuntiva
# ============================================================================
def _print_summary(results: Dict, title: str) -> None:
    w = 85
    print(f'\n{"="*w}\n  {title}\n{"="*w}')
    print(f'  {"Metodo":<20}  {"DTW":>9}  {"RMSE(°)":>8}  '
          f'{"Peak(°)":>8}  {"Pearson":>8}  {"Smooth":>10}')
    print(f'  {"-"*73}')
    for m, r in results.items():
        print(f'  {m:<20}  {r["dtw_distance"]:>9.2f}  {r["rmse_mean"]:>8.2f}  '
              f'{r["peak_error_mean"]:>8.2f}  {r["pearson_mean"]:>8.3f}  '
              f'{r["smoothness"]:>10.4f}')
    print(f'{"="*w}')


# ============================================================================
# MODALITÀ 1 — Valutazione singolo esercizio
# ============================================================================
def run_exercise_evaluation(exercise_num: int,
                             n_steps: Optional[int] = None) -> Dict[str, Dict]:
    '''
    Valuta un singolo esercizio con tutte le architetture disponibili.
    Ritorna il dict results per eventuale riutilizzo nell'analisi modality.
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

    baseline = load_baseline(dataset_dir)
    if baseline is None:
        print('  SKIP: baseline.csv mancante.')
        return {}

    canonical  = load_canonical(dataset_dir)
    n          = n_steps if n_steps is not None else len(baseline)
    start_pose = _start_pose_from_canonical(canonical)

    print('\nCaricamento demo umane ...')
    human_demos = load_human_demos(landmarks_root, exercise_num)

    bc_trajs: Dict[str, Optional[np.ndarray]] = {}
    for arch_name, arch_dir in ARCHITECTURES.items():
        print(f'\nGenerazione BC [{arch_name}] ({n} step) ...')
        bc_trajs[arch_name] = generate_bc_trajectory(
            dataset_dir / arch_dir, start_pose, n)

    print('\nCalcolo metriche ...')
    results: Dict[str, Dict] = {}

    if human_demos:
        agg = _aggregate_metrics([compute_metrics(d, baseline) for d in human_demos])
        results['Human demos'] = agg
        print(f'  [Human demos      ]  DTW={agg["dtw_distance"]:>9.2f}  '
              f'RMSE={agg["rmse_mean"]:>6.2f}°  r={agg["pearson_mean"]:>6.3f}  '
              f'Smooth={agg["smoothness"]:>9.4f}')

    if canonical is not None:
        results['Canonical'] = compute_metrics(canonical, baseline, label='Canonical')

    for arch_name, traj in bc_trajs.items():
        if traj is not None:
            results[arch_name] = compute_metrics(traj, baseline, label=arch_name)

    if not results:
        print('  Nessun risultato — verificare canonical.csv e modelli.')
        return {}

    print('\nSalvataggio CSV ...')
    _save_results_csv(results, output_dir)

    print('\nGenerazione plot ...')
    suffix = f' — Exercise {exercise_num:03d} [{modality}]'
    _plot_degradation_chain(results, output_dir, exercise_num)
    _plot_rmse_per_joint(results, output_dir, suffix)
    _plot_pearson_per_joint(results, output_dir, suffix)
    _plot_smoothness(results, output_dir, suffix)
    _plot_velocity_profile(
        {'Baseline': baseline, 'Canonical': canonical, **bc_trajs},
        output_dir)

    _print_summary(results, f'Exercise {exercise_num:03d} [{modality}]')
    print(f'\n  Output → {output_dir}')
    return results


# ============================================================================
# MODALITÀ 2 — Analisi modality (Stereo vs Mixed vs Mono)
# ============================================================================
def run_modality_analysis(n_steps: Optional[int] = None) -> None:
    '''
    Per ogni modalità e ogni tipo di esercizio, calcola metriche BC di tutte
    le architetture. Aggrega per modalità e genera plot di confronto.
    '''
    landmarks_root = DATA_ROOT / 'landmarks'
    output_dir     = DATA_ROOT / 'evaluation_modality'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*65}')
    print('  ANALISI MODALITY — Stereo vs Mixed vs Mono')
    print(f'{"="*65}')

    # ex_results[exercise_num][arch] = metrics
    ex_results:  Dict[int,  Dict[str, Dict]] = {}
    # agg[modality][arch] = metrics aggregati
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
            baseline  = load_baseline(dataset_dir)
            if baseline is None:
                continue
            canonical  = load_canonical(dataset_dir)
            n          = n_steps if n_steps is not None else len(baseline)
            start_pose = _start_pose_from_canonical(canonical)
            ex_results.setdefault(ex_num, {})

            for arch_name, arch_dir in ARCHITECTURES.items():
                print(f'    [{arch_name}]', end=' ', flush=True)
                traj = generate_bc_trajectory(
                    dataset_dir / arch_dir, start_pose, n)
                if traj is None:
                    continue
                met = compute_metrics(traj, baseline,
                                      label=f'{arch_name}/{ex_num:03d}')
                ex_results[ex_num][arch_name] = met
                arch_all[arch_name].append(met)
                csv_rows.append({
                    'modality'     : mod,
                    'exercise_num' : ex_num,
                    'exercise_type': get_exercise_type(ex_num),
                    'architecture' : arch_name,
                    **{k: v for k, v in met.items()
                       if not isinstance(v, np.ndarray)},
                })

        for arch_name, met_list in arch_all.items():
            if met_list:
                agg[mod][arch_name] = _aggregate_metrics(met_list)

    # CSV
    if csv_rows:
        p = output_dir / 'results_by_exercise_type.csv'
        pd.DataFrame(csv_rows).to_csv(p, index=False)
        print(f'\n  Saved -> {p.name}')

    agg_rows = [
        {'modality': mod, 'architecture': arch,
         **{k: v for k, v in met.items() if not isinstance(v, np.ndarray)}}
        for mod, arch_mets in agg.items()
        for arch, met in arch_mets.items()
    ]
    if agg_rows:
        p = output_dir / 'results_aggregated.csv'
        pd.DataFrame(agg_rows).to_csv(p, index=False)
        print(f'  Saved -> {p.name}')

    # Plot aggregati per modalità
    print('\nGenerazione plot analisi modality ...')
    for metric, ylabel, fname in [
        ('dtw_distance', 'DTW distance (↓ meglio)',           'plot_dtw_modality.png'),
        ('rmse_mean',    'RMSE medio su active joints (°)',    'plot_rmse_modality.png'),
        ('pearson_mean', 'Pearson r medio  (↑ meglio)',        'plot_pearson_modality.png'),
        ('smoothness',   'Smoothness −mean(jerk²)  (↑ meglio)','plot_smoothness_modality.png'),
    ]:
        _plot_modality_grouped_bar(
            agg, metric, ylabel=ylabel,
            title=f'Modality Analysis — {ylabel}',
            output_path=output_dir / fname)

    # Heatmap RMSE per joint per ogni architettura
    for arch_name in ARCHITECTURES:
        _plot_modality_rmse_heatmap(
            agg, arch_name,
            output_dir / f'plot_rmse_heatmap_{arch_name}.png')

    # Line plot per-exercise
    if ex_results:
        for metric, ylabel, fname in [
            ('dtw_distance', 'DTW distance',  'plot_dtw_per_exercise.png'),
            ('rmse_mean',    'RMSE (°)',       'plot_rmse_per_exercise.png'),
            ('pearson_mean', 'Pearson r',      'plot_pearson_per_exercise.png'),
        ]:
            _plot_per_exercise_lines(
                ex_results, metric, ylabel,
                title=f'Per-Exercise — {ylabel}  [all modalities × architectures]',
                output_path=output_dir / fname)

    # Stampa riepilogo
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
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Valutazione del pipeline LfD su Reachy (MLP / GRU / Transformer).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Esempi:
  python evaluate.py --exercise 1
  python evaluate.py --exercise 1 11 21
  python evaluate.py --analysis modality
  python evaluate.py --all
        ''',
    )
    parser.add_argument(
        '--exercise', type=int, nargs='+', default=None, metavar='N',
        help='Numero/i esercizio da valutare (es. --exercise 1 11 21).'
    )
    parser.add_argument(
        '--analysis', type=str, default=None, choices=['modality'],
        help='"modality": confronto Stereo vs Mixed vs Mono.'
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Valuta tutti gli esercizi trovati + analisi modality.'
    )
    parser.add_argument(
        '--steps', type=int, default=None,
        help='Numero di step BC. Default: lunghezza baseline.csv.'
    )
    args = parser.parse_args()

    run_modal  = args.all or (args.analysis == 'modality')
    run_single = args.exercise is not None or args.all

    if args.all:
        dataset_root  = DATA_ROOT / 'dataset'
        exercise_nums = sorted(
            int(d.name.split('_')[1])
            for d in dataset_root.glob('exercise_???')
            if d.is_dir()
        )
    else:
        exercise_nums = args.exercise or []

    if run_single and exercise_nums:
        for ex_num in exercise_nums:
            run_exercise_evaluation(ex_num, args.steps)

    if run_modal:
        run_modality_analysis(args.steps)

    if not run_single and not run_modal:
        parser.print_help()

    print('\nDone.')


if __name__ == '__main__':
    main()