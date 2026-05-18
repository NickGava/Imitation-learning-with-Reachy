'''
_plots.py
=============================================================================
Tutte le funzioni di visualizzazione del modulo evaluation_and_comparison.

Sezioni:
  A. Helpers condivisi
  B. Plot per singolo esercizio  (chiamati da evaluate_exercise.py)
  C. Plot per analisi modality   (chiamati da evaluate_modality.py)
'''

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evaluation_and_comparison._config import (
    ACTIVE_IDX, ACTIVE_LABELS, JOINT_LABELS, PALETTE,
    VELOCITY_JOINTS, ARCHITECTURES, MODALITY_GROUPS,
    get_exercise_type,
)

# ============================================================================
# A. Helpers condivisi
# ============================================================================

def _grouped_bar(ax, methods_dict: Dict, arr_key: str,
                 active_only: bool = True) -> None:
    '''Bar chart raggruppato su active joints (o tutti i joint se active_only=False).'''
    idx    = ACTIVE_IDX    if active_only else list(range(len(JOINT_LABELS)))
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


# Active side helpers
_RIGHT_ACTIVE_IDX    = [0, 1, 2, 3]   # indices within ACTIVE_IDX for right arm
_LEFT_ACTIVE_IDX     = [4, 5, 6, 7]   # indices within ACTIVE_IDX for left arm
_RIGHT_ACTIVE_LABELS = ['r_sh_p', 'r_sh_r', 'r_aw', 'r_el_p']
_LEFT_ACTIVE_LABELS  = ['l_sh_p', 'l_sh_r', 'l_aw', 'l_el_p']

def _side_joint_filter(active_side):
    """Returns (idx_list, label_list) for the active side within ACTIVE_IDX."""
    if active_side == 'right':
        return list(range(4)),  _RIGHT_ACTIVE_LABELS
    elif active_side == 'left':
        return list(range(4, 8)), _LEFT_ACTIVE_LABELS
    return list(range(8)), ACTIVE_LABELS

# ============================================================================
# B. Plot per singolo esercizio
# ============================================================================

def plot_degradation_chain(results: Dict, output_dir: Path,
                           exercise_num: int, modality: str) -> None:
    '''
    Bar chart della DTW cartesiana (metri) per ogni metodo nella degradation chain.
    La DTW cartesiana  invariante alla ridondanza cinematica e interpretabile:
    rappresenta la distanza media (m) del polso dalla traiettoria di riferimento
    dopo allineamento temporale ottimale.

    Ordine atteso: Human demos  Canonical  MLP  GRU  Transformer
    '''
    CHAIN_ORDER = ['Human demos', 'Canonical', 'MLP', 'GRU', 'Transformer']

    # Usa cart_dtw_norm (m/frame) se disponibile, altrimenti fallback a dtw_distance (gradi)
    use_cart = any('cart_dtw_norm' in results.get(m, {}) for m in CHAIN_ORDER)
    metric   = 'cart_dtw_norm' if use_cart else 'dtw_distance'
    unit     = 'm/frame' if use_cart else 'deg'

    methods  = [m for m in CHAIN_ORDER if m in results] + \
               [m for m in results if m not in CHAIN_ORDER]
    dtw_vals = [results[m].get(metric, float('nan')) for m in methods]
    colors   = [PALETTE.get(m, '#95a5a6') for m in methods]

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.8), 5))
    bars = ax.bar(methods, dtw_vals, color=colors, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, dtw_vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(v for v in dtw_vals if not np.isnan(v)) * 0.012,
                    f'{val:.3f}{unit}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    ax.set_ylabel(f'DTW distance vs baseline  [{unit}]  ( meglio)', fontsize=11)
    ax.set_title(
        f'Degradation Chain  Exercise {exercise_num:03d} [{modality}]\n'
        f'DTW cartesiana  invariante alla ridondanza cinematica\n'
        'Expected: MLP / GRU / Transformer  Canonical < Human demos',
        fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(v for v in dtw_vals if not np.isnan(v)) * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    p = output_dir / 'plot_degradation_chain.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {p.name}')


def plot_rmse_per_joint(results: Dict, output_dir: Path,
                        title_suffix: str = '',
                        active_side: str = 'both') -> None:
    methods = {k: v for k, v in results.items() if k != 'Human demos'}
    if not methods:
        return
    a_idx, a_labels = _side_joint_filter(active_side)
    x     = np.arange(len(a_labels))
    width = 0.75 / max(len(methods), 1)
    off0  = -(len(methods) - 1) / 2 * width
    fig, ax = plt.subplots(figsize=(max(6, len(a_labels) * 1.8), 5))
    for i, (m, res) in enumerate(methods.items()):
        pj   = res.get('rmse_per_joint', np.zeros(16))
        vals = pj[ACTIVE_IDX][a_idx]
        ax.bar(x + off0 + i * width, vals, width,
               label=m, color=PALETTE.get(m, f'C{i}'),
               edgecolor='white', linewidth=0.8, alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(a_labels, rotation=45, ha='right', fontsize=9)
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


def plot_pearson_per_joint(results, output_dir, title_suffix='', active_side='both'):
    methods = {k: v for k, v in results.items() if k != 'Human demos'}
    if not methods:
        return

    # Filtra per braccio attivo, poi per joint non-costanti
    a_idx, a_labels = _side_joint_filter(active_side)
    side_idx    = [ACTIVE_IDX[i] for i in a_idx]

    active_mask = np.zeros(len(side_idx), dtype=bool)
    for res in methods.values():
        pj = res.get('pearson_per_joint', np.zeros(16))
        active_mask |= (np.abs(pj[side_idx]) > 0.05)

    visible_idx    = [side_idx[i]    for i, keep in enumerate(active_mask) if keep]
    visible_labels = [a_labels[i] for i, keep in enumerate(active_mask) if keep]

    if not visible_idx:
        print('  [skip] plot_pearson_per_joint: tutti i joint sono costanti nella baseline')
        return

    x     = np.arange(len(visible_labels))
    width = 0.75 / max(len(methods), 1)
    off0  = -(len(methods) - 1) / 2 * width

    fig, ax = plt.subplots(figsize=(max(7, len(visible_labels) * 1.8), 5))
    for i, (m, res) in enumerate(methods.items()):
        pj   = res.get('pearson_per_joint', np.zeros(16))
        vals = pj[visible_idx]
        ax.bar(x + off0 + i * width, vals, width,
               label=m, color=PALETTE.get(m, f'C{i}'),
               edgecolor='white', linewidth=0.8, alpha=0.88)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylim(-1.05, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(visible_labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Pearson r  (higher=better)', fontsize=11)
    ax.set_title(
        f'Pearson Correlation per Joint{title_suffix}\n'
        '(solo joint non-costanti nella baseline; struttura temporale DTW-aligned)',
        fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    p = output_dir / 'plot_pearson_per_joint.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {p.name}')



def plot_smoothness(results: Dict, output_dir: Path,
                    title_suffix: str = '') -> None:
    methods = list(results.keys())
    smooth  = [results[m]['smoothness'] for m in methods]
    colors  = [PALETTE.get(m, f'C{i}') for i, m in enumerate(methods)]
    ymax    = max((s for s in smooth if not np.isnan(s)), default=1.0)

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.8), 5))
    bars = ax.bar(methods, smooth, color=colors, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, smooth):
        if np.isnan(val):
            continue
        bar_h = bar.get_height()   # negativo
        bar_x = bar.get_x() + bar.get_width() / 2
        if abs(bar_h) > abs(ymax) * 0.08:
            # Barra abbastanza alta: etichetta al centro della barra
            ax.text(bar_x, bar_h / 2,
                    f'{val:.4f}', ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')
        else:
            # Barra quasi zero: etichetta appena sotto lo zero
            ax.text(bar_x, -abs(ymax) * 0.04,
                    f'{val:.4f}', ha='center', va='top',
                    fontsize=8, fontweight='bold')
    ax.set_ylabel('Smoothness: mean(jerk)  ( pi fluido)', fontsize=10)
    ax.set_title(
        f'Smoothness{title_suffix}\n(misura intrinseca  non vs baseline)',
        fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    p = output_dir / 'plot_smoothness.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {p.name}')


def plot_velocity_profile(trajs, output_dir, baseline=None):
    from evaluation_and_comparison._config import VELOCITY_JOINTS, JOINT_LABELS
    if baseline is not None and len(baseline) > 1:
        stds        = np.std(baseline, axis=0)
        joint_idx   = int(np.argmax(stds))
        joint_label = JOINT_LABELS[joint_idx]
        active_side = "right" if joint_idx < 8 else "left"
        joints_to_plot = {joint_label: joint_idx}
    else:
        joints_to_plot = VELOCITY_JOINTS
        active_side    = "both"

    for joint_short, j_idx in joints_to_plot.items():
        items = [(lbl, t) for lbl, t in trajs.items()
                 if t is not None and len(t) > 2]
        if not items:
            continue
        n = len(items)
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=False)
        if n == 1:
            axes = [axes]
        for ax, (lbl, traj) in zip(axes, items):
            vel = np.diff(traj[:, j_idx])
            ax.plot(vel, color=PALETTE.get(lbl, "#7f8c8d"), linewidth=1.5)
            ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
            ax.set_title(lbl, fontsize=11, fontweight="bold")
            ax.set_xlabel("Frame", fontsize=9)
            ax.set_ylabel("Vel. angolare (deg/frame)", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(alpha=0.3)
        fig.suptitle(
            f"Velocity Profile -- {joint_short}  [{active_side} arm]\n"
            "(bell-shaped ~ moto umano naturale; misura intrinseca)",
            fontsize=11, fontweight="bold")
        fig.tight_layout()
        p = output_dir / f"plot_velocity_{joint_short}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {p.name}")

# ============================================================================
# C. Plot per analisi modality
# ============================================================================

def plot_modality_grouped_bar(agg: Dict[str, Dict[str, Dict]],
                              metric: str, ylabel: str, title: str,
                              output_path: Path) -> None:
    '''
    Grouped bar: asse X = architetture, colori = modalit  (Stereo / Mixed / Mono).
    agg[modality][arch] = metrics_dict
    '''
    archs = list(ARCHITECTURES.keys())
    mods  = list(MODALITY_GROUPS.keys())
    x     = np.arange(len(archs))
    width = 0.25
    off0  = -(len(mods) - 1) / 2 * width

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


def plot_modality_rmse_heatmap(agg: Dict[str, Dict[str, Dict]],
                               arch: str, output_path: Path) -> None:
    '''Heatmap RMSE per joint  modalit  per una singola architettura.'''
    mods = list(MODALITY_GROUPS.keys())
    data = np.array([
        agg.get(mod, {}).get(arch, {}).get(
            'rmse_per_joint', np.full(len(ACTIVE_IDX), np.nan)
        )
        for mod in mods
    ])
    # Estrai solo gli active joint se i vettori sono lunghi 16
    if data.shape[1] == 16:
        data = data[:, ACTIVE_IDX]

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
        for j in range(len(ACTIVE_LABELS)):
            val = data[i, j]
            txt = f'{val:.1f}' if not np.isnan(val) else ''
            ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                    color='white' if val > vmax * 0.6 else 'black')
    ax.set_title(f'RMSE per Joint  {arch}  Modality Comparison',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {output_path.name}')


def plot_per_exercise_lines(ex_results: Dict[int, Dict[str, Dict]],
                            metric: str, ylabel: str, title: str,
                            output_path: Path) -> None:
    '''
    Line plot: asse X = tipo esercizio (15), una linea per (architettura  modalit ).
    ex_results[exercise_num][arch] = metrics_dict
    '''
    ex_types = sorted({get_exercise_type(n) for n in ex_results})
    fig, ax  = plt.subplots(figsize=(10, 5))

    ls_map = {'Stereo': '-', 'Mixed': '--', 'Mono': ':'}
    for mod, ex_nums in MODALITY_GROUPS.items():
        for arch in ARCHITECTURES:
            vals = []
            for et in ex_types:
                num = next((n for n in ex_nums if get_exercise_type(n) == et), None)
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
# D. Plot cartesiani (per singolo esercizio)
# ============================================================================

def plot_cartesian_pearson(results: Dict, output_dir: Path,
                            title_suffix: str = '',
                            active_side: str = 'both') -> None:
    '''
    Bar chart della Pearson correlation per coordinata cartesiana del polso.
    6 coordinate: r_wrist_X, r_wrist_Y, r_wrist_Z, l_wrist_X, l_wrist_Y, l_wrist_Z.

    A differenza della versione joint-space, qui tutte le coordinate hanno
    valori significativi (il polso si muove sempre in 3D).
    '''
    methods = {k: v for k, v in results.items()
               if k != 'Human demos' and 'cart_pearson_rw' in v}
    if not methods:
        return

    if active_side == 'right':
        COORD_LABELS = ['Rw X', 'Rw Y', 'Rw Z']
        all_vals = np.array([res['cart_pearson_rw'] for res in methods.values()])
    elif active_side == 'left':
        COORD_LABELS = ['Lw X', 'Lw Y', 'Lw Z']
        all_vals = np.array([res['cart_pearson_lw'] for res in methods.values()])
    else:
        COORD_LABELS = ['Rw X', 'Rw Y', 'Rw Z', 'Lw X', 'Lw Y', 'Lw Z']
        all_vals = np.array([
            np.concatenate([res['cart_pearson_rw'], res['cart_pearson_lw']])
            for res in methods.values()
        ])

    # Mostra solo le coordinate dove almeno un metodo ha |pearson| > 0.05
    visible_mask   = np.any(np.abs(all_vals) > 0.05, axis=0)
    visible_coords = [l for l, v in zip(COORD_LABELS, visible_mask) if v]
    visible_all    = all_vals[:, visible_mask]

    if not visible_coords:
        print('  [skip] plot_cartesian_pearson: tutte le coordinate sono costanti')
        return

    x     = np.arange(len(visible_coords))
    width = 0.75 / max(len(methods), 1)
    off0  = -(len(methods) - 1) / 2 * width

    fig, ax = plt.subplots(figsize=(max(6, len(visible_coords) * 1.8), 5))
    for i, (m, vals_row) in enumerate(zip(methods.keys(), visible_all)):
        ax.bar(x + off0 + i * width, vals_row, width,
               label=m, color=PALETTE.get(m, f'C{i}'),
               edgecolor='white', linewidth=0.8, alpha=0.88)

    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylim(-1.05, 1.15)
    ax.set_xticks(x)
    ax.set_xticklabels(visible_coords, fontsize=10)
    ax.set_ylabel('Pearson r  ( meglio)', fontsize=11)
    ax.set_title(
        f'Pearson Correlation  Wrist Cartesian Coordinates{title_suffix}\n'
        '(struttura temporale, DTW-aligned; invariante alla ridondanza cinematica)',
        fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    p = output_dir / 'plot_cartesian_pearson.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {p.name}')


def plot_cartesian_trajectories(trajs, output_dir, title_suffix='', active_side='both'):
    """Wrist Cartesian trajectories per il braccio attivo."""
    from evaluation_and_comparison._metrics import _to_cartesian

    AXIS_LABELS = ['X (forward, m)', 'Y (lateral, m)', 'Z (up, m)']
    sides = (['right'] if active_side == 'right'
             else ['left'] if active_side == 'left'
             else ['right', 'left'])
    side_titles = {'right': 'Right wrist', 'left': 'Left wrist'}

    cart = {}
    for label, traj in trajs.items():
        if traj is None:
            continue
        rw, _, lw, _ = _to_cartesian(traj)
        cart[label] = {'right': rw, 'left': lw}
    if not cart:
        return

    n_cols = len(sides)
    fig, axes = plt.subplots(3, n_cols, figsize=(7 * n_cols, 10), sharex=False)
    if n_cols == 1:
        axes = axes.reshape(3, 1)
    fig.suptitle(f'Wrist Cartesian Trajectories{title_suffix}', fontsize=13, fontweight='bold')
    for col_i, side_key in enumerate(sides):
        axes[0, col_i].set_title(side_titles[side_key], fontsize=11, fontweight='bold')

    ls_map = {'Baseline': '-', 'Canonical': '--'}
    for ax_i in range(3):
        for col_i, side_key in enumerate(sides):
            ax = axes[ax_i, col_i]
            for label, data in cart.items():
                t = data[side_key]
                ax.plot(t[:, ax_i],
                        color=PALETTE.get(label, '#7f8c8d'),
                        linewidth=2.5 if label == 'Baseline' else 1.5,
                        linestyle=ls_map.get(label, '-'),
                        label=label, alpha=1.0 if label == 'Baseline' else 0.8)
            if col_i == 0:
                ax.set_ylabel(AXIS_LABELS[ax_i], fontsize=9)
            if ax_i == 2:
                ax.set_xlabel('Frame', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(alpha=0.3)
            if ax_i == 0 and col_i == n_cols - 1:
                ax.legend(fontsize=8, loc='best')

    fig.tight_layout()
    p = output_dir / 'plot_cartesian_trajectories.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {p.name}')


def plot_cartesian_rmse(results, output_dir, title_suffix='', active_side='both'):
    """Bar chart RMSE cartesiano per endpoint del braccio attivo (in metri)."""
    methods = {k: v for k, v in results.items()
               if 'cart_rmse_r_wrist' in v and k != 'Human demos'}
    if not methods:
        return

    if active_side == 'right':
        ep_keys   = ['cart_rmse_r_wrist', 'cart_rmse_r_elbow']
        ep_labels = ['R wrist', 'R elbow']
    elif active_side == 'left':
        ep_keys   = ['cart_rmse_l_wrist', 'cart_rmse_l_elbow']
        ep_labels = ['L wrist', 'L elbow']
    else:
        ep_keys   = ['cart_rmse_r_wrist', 'cart_rmse_l_wrist',
                     'cart_rmse_r_elbow', 'cart_rmse_l_elbow']
        ep_labels = ['R wrist', 'L wrist', 'R elbow', 'L elbow']

    x     = np.arange(len(ep_labels))
    width = 0.75 / max(len(methods), 1)
    off0  = -(len(methods) - 1) / 2 * width

    fig, ax = plt.subplots(figsize=(max(5, len(ep_labels) * 2.2), 5))
    for i, (m, res) in enumerate(methods.items()):
        vals = [res.get(k, np.nan) for k in ep_keys]
        ax.bar(x + off0 + i * width, vals, width,
               label=m, color=PALETTE.get(m, f'C{i}'),
               edgecolor='white', linewidth=0.8, alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(ep_labels, fontsize=11)
    ax.set_ylabel('RMSE (metri)  (lower=better)', fontsize=11)
    ax.set_title(
        f'RMSE Cartesiano per Endpoint{title_suffix}\n'
        '(invariante alla ridondanza cinematica)',
        fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    p = output_dir / 'plot_cartesian_rmse.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {p.name}')



def plot_cartesian_velocity(trajs, output_dir, baseline=None):
    """
    Profilo di velocita cartesiana: wrist e elbow in m/frame.
    Solo il braccio attivo (da baseline). Wrist continua, Elbow tratteggiata.
    """
    from evaluation_and_comparison._metrics import _to_cartesian

    items = [(lbl, t) for lbl, t in trajs.items() if t is not None and len(t) > 2]
    if not items:
        return

    if baseline is not None and len(baseline) > 1:
        stds        = np.std(baseline, axis=0)
        joint_idx   = int(np.argmax(stds))
        active_side = 'right' if joint_idx < 8 else 'left'
        sides       = [(active_side, 'R' if active_side == 'right' else 'L')]
    else:
        sides = [('right', 'R'), ('left', 'L')]

    for side_key, side_tag in sides:
        n    = len(items)
        fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=False)
        if n == 1:
            axes = [axes]

        for i, (ax, (lbl, traj)) in enumerate(zip(axes, items)):
            rw, re, lw, le = _to_cartesian(traj)
            wrist   = rw if side_key == 'right' else lw
            elbow   = re if side_key == 'right' else le
            speed_w = np.linalg.norm(np.diff(wrist, axis=0), axis=1)
            speed_e = np.linalg.norm(np.diff(elbow, axis=0), axis=1)
            color   = PALETTE.get(lbl, '#7f8c8d')
            ax.plot(speed_w, color=color, linewidth=1.8, label='wrist')
            ax.plot(speed_e, color=color, linewidth=1.2,
                    linestyle='--', alpha=0.7, label='elbow')
            ax.axhline(0, color='grey', linewidth=0.8, linestyle=':')
            ax.set_title(lbl, fontsize=11, fontweight='bold')
            ax.set_xlabel('Frame', fontsize=9)
            ax.set_ylabel('||vel|| (m/frame)', fontsize=8)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(alpha=0.3)
            if i == 0:
                ax.legend(fontsize=8)

        fig.suptitle(
            f'Cartesian Velocity Profile -- {side_tag} arm  [wrist / elbow --]\n'
            '(bell-shaped = moto umano naturale; misura in metri)',
            fontsize=11, fontweight='bold')
        fig.tight_layout()
        p = output_dir / f'plot_cartesian_velocity_{side_tag.lower()}_arm.png'
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved -> {p.name}')


def plot_summary_heatmap(results: Dict, output_dir: Path,
                          title_suffix: str = '',
                          active_side: str = 'both') -> None:
    '''
    Heatmap riassuntiva: righe = metodi, colonne = metriche scalari chiave.
    Le metriche RMSE cartesiane si adattano al braccio attivo.
    '''
    s = active_side
    wrist_key = 'cart_rmse_l_wrist' if s == 'left'  else ('cart_rmse_r_wrist' if s == 'right' else 'cart_rmse_r_wrist')
    elbow_key = 'cart_rmse_l_elbow' if s == 'left'  else ('cart_rmse_r_elbow' if s == 'right' else 'cart_rmse_r_elbow')
    wrist_lbl = 'RMSE\nLw (m)'     if s == 'left'  else ('RMSE\nRw (m)'      if s == 'right' else 'RMSE\nRw (m)')
    elbow_lbl = 'RMSE\nLe (m)'     if s == 'left'  else ('RMSE\nRe (m)'      if s == 'right' else 'RMSE\nRe (m)')

    METRICS = [
        ('cart_dtw_norm', 'DTW\n(m/frame)', True,  '{:.4f}'),
        (wrist_key,       wrist_lbl,        True,  '{:.4f}'),
        (elbow_key,       elbow_lbl,        True,  '{:.4f}'),
        ('cart_pearson_mean', 'Pearson\ncart', False, '{:.3f}'),
        ('rmse_mean',     'RMSE\n(joint)',   True,  '{:.2f}'),
        ('pearson_mean',  'Pearson\njoint',  False, '{:.3f}'),
    ]

    METHOD_ORDER = ['Human demos', 'Canonical', 'MLP', 'GRU', 'Transformer']
    methods = [m for m in METHOD_ORDER if m in results] + \
              [m for m in results if m not in METHOD_ORDER]

    # Costruisci matrice dati (n_methods  n_metrics)
    data     = np.full((len(methods), len(METRICS)), np.nan)
    for i, m in enumerate(methods):
        for j, (key, _, _, _) in enumerate(METRICS):
            val = results[m].get(key, np.nan)
            if not np.isnan(val):
                data[i, j] = val

    # Normalizza ogni colonna in [0,1] dove 1 = migliore
    scores = np.full_like(data, np.nan)
    for j, (_, _, lower_is_better, _) in enumerate(METRICS):
        col      = data[:, j]
        finite   = col[np.isfinite(col)]
        if len(finite) < 2 or np.ptp(finite) < 1e-12:
            scores[:, j] = 0.5
            continue
        norm = (col - np.nanmin(col)) / np.ptp(finite)
        scores[:, j] = (1 - norm) if lower_is_better else norm

    col_labels = [label for _, label, _, _ in METRICS]
    formats    = [fmt   for _, _, _, fmt   in METRICS]

    fig, ax = plt.subplots(figsize=(max(8, len(METRICS) * 1.6),
                                    max(3, len(methods) * 0.9)))
    im = ax.imshow(scores, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

    # Etichette assi
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=11)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    # Valori numerici nelle celle
    for i in range(len(methods)):
        for j, fmt in enumerate(formats):
            val = data[i, j]
            txt = fmt.format(val) if not np.isnan(val) else ''
            score = scores[i, j]
            color = 'white' if (score < 0.25 or score > 0.75) else 'black'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Performance  (verde = migliore)', fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['peggiore', 'medio', 'migliore'], fontsize=8)

    ax.set_title(
        f'Riepilogo Metriche{title_suffix}',
        fontsize=13, fontweight='bold', pad=18)

    # Linee di separazione tra metodi
    for i in range(len(methods) - 1):
        ax.axhline(i + 0.5, color='white', linewidth=1.5)
    for j in range(len(col_labels) - 1):
        ax.axvline(j + 0.5, color='white', linewidth=1.5)

    fig.tight_layout()
    p = output_dir / 'plot_summary_heatmap.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {p.name}')