'''
_plots.py
=============================================================================
All visualization functions for the evaluation_and_comparison module.

Sections:
  A. Shared helpers
  B. Per-exercise plots  (called by evaluate_exercise.py)
  C. Modality analysis plots   (called by evaluate_modality.py)
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
# A. Shared helpers
# ============================================================================

# Active side helpers
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
# B. Plot for single exercise evaluation
# ============================================================================

def plot_degradation_chain(results: Dict, output_dir: Path, exercise_num: int, modality: str, arch_variance: Optional[Dict] = None) -> None:
    '''
    Degradation chain: bar chart of Cartesian DTW (m) vs baseline.
    If arch_variance is available, adds error bars (±std) to the BC
    architecture bars.
    '''
    CHAIN_ORDER = ['Canonical', 'CanonicalShape', 'MLP', 'GRU', 'Transformer']
    methods = [m for m in CHAIN_ORDER if m in results and 'cart_dtw' in results[m]]

    if not methods:
        return

    values = [results[m]['cart_dtw'] for m in methods]
    colors = [PALETTE.get(m, '#95a5a6') for m in methods]

    # Errorbar (std): only for architectures with available variance data
    yerr = []
    for m in methods:
        var = (arch_variance or {}).get(m, None)
        std = var.get('cart_dtw', np.nan) if var else np.nan
        yerr.append(std if not np.isnan(std) else 0.0)
    has_err = any(e > 0 for e in yerr)

    fig, ax = plt.subplots(figsize=(max(5, len(methods) * 1.6), 5))

    bars = ax.bar(methods, values,
                  yerr=yerr if has_err else None,
                  error_kw=dict(ecolor='#555', capsize=5, capthick=1.5, elinewidth=1.5),
                  color=colors, edgecolor='white',
                  linewidth=1.5, alpha=0.88, width=0.55)

    for bar, val, err in zip(bars, values, yerr):
        label = f'{val:.3f} m'
        if has_err and err > 0:
            label += f'\n±{err:.3f}'
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.015 + (err if has_err else 0),
                label,
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=bar.get_facecolor())

    ax.set_ylabel('Cartesian DTW vs baseline  (m)  ↓ best', fontsize=11)
    ax.set_title(
        f'Degradation Chain - Exercise {exercise_num:03d} [{modality}]\n'
        'cartesian DTW (m) vs baseline'
        + ('  |  errorbar = std between training run' if has_err else ''),
        fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(v + e for v, e in zip(values, yerr)) * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    p = output_dir / 'plot_degradation_chain.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {p.name}')


def plot_rmse_per_joint(results: Dict, output_dir: Path, title_suffix: str = '', active_side: str = 'both') -> None:
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
    ax.set_ylabel('RMSE (deg)', fontsize=11)
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

    # Filter by active arm, then by non-constant joints 
    a_idx, a_labels = _side_joint_filter(active_side)
    side_idx    = [ACTIVE_IDX[i] for i in a_idx]

    active_mask = np.zeros(len(side_idx), dtype=bool)
    for res in methods.values():
        pj = res.get('pearson_per_joint', np.zeros(16))
        active_mask |= (np.abs(pj[side_idx]) > 0.05)

    visible_idx    = [side_idx[i]    for i, keep in enumerate(active_mask) if keep]
    visible_labels = [a_labels[i] for i, keep in enumerate(active_mask) if keep]

    if not visible_idx:
        print('  [skip] plot_pearson_per_joint: all joints are constant in the baseline.')
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
        '(only non-constant joints in the baseline; DTW-aligned temporal structure)',
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



def plot_smoothness(results: Dict, output_dir: Path, title_suffix: str = '') -> None:
    methods = list(results.keys())
    smooth  = [results[m]['smoothness'] for m in methods]
    colors  = [PALETTE.get(m, f'C{i}') for i, m in enumerate(methods)]
    ymax    = max((s for s in smooth if not np.isnan(s)), default=1.0)

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 1.8), 5))
    bars = ax.bar(methods, smooth, color=colors, edgecolor='white', linewidth=1.5)
    for bar, val in zip(bars, smooth):
        if np.isnan(val):
            continue
        bar_h = bar.get_height()   # negativ
        bar_x = bar.get_x() + bar.get_width() / 2
        if abs(bar_h) > abs(ymax) * 0.08:
            ax.text(bar_x, bar_h / 2,
                    f'{val:.4f}', ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white')
        else:
            ax.text(bar_x, -abs(ymax) * 0.04,
                    f'{val:.4f}', ha='center', va='top',
                    fontsize=8, fontweight='bold')
    ax.set_ylabel('Smoothness: mean(jerk)  (more fluid)', fontsize=10)
    ax.set_title(
        f'Smoothness{title_suffix}\n(intrinsic measure, not vs baseline)',
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
            "(bell-shaped ~ human natural movement; intrinsic measure)",
            fontsize=11, fontweight="bold")
        fig.tight_layout()
        p = output_dir / f"plot_velocity_{joint_short}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved -> {p.name}")

# ============================================================================
# C. Plot per analisis modality
# ============================================================================

def plot_modality_grouped_bar(agg: Dict[str, Dict[str, Dict]], metric: str, ylabel: str, title: str, output_path: Path) -> None:
    '''
    Grouped bar: X axis = architectures, colors = modality  (Stereo / Mixed / Mono).
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


def plot_modality_rmse_heatmap(agg: Dict[str, Dict[str, Dict]], arch: str, output_path: Path) -> None:
    '''Heatmap RMSE per joint modality for a singol architecture.'''
    mods = list(MODALITY_GROUPS.keys())
    data = np.array([
        agg.get(mod, {}).get(arch, {}).get(
            'rmse_per_joint', np.full(len(ACTIVE_IDX), np.nan)
        )
        for mod in mods
    ])
    
    # Extract only active joints if the vectors are 16-long 
    if data.shape[1] == 16:
        data = data[:, ACTIVE_IDX]

    finite = data[np.isfinite(data)]
    vmax   = float(np.nanmax(finite)) if len(finite) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(10, 2.0 + len(mods) * 0.7))
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd', vmin=0, vmax=vmax)
    plt.colorbar(im, ax=ax, label='RMSE (deg)')
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


def plot_per_exercise_lines(ex_results: Dict[int, Dict[str, Dict]], metric: str, ylabel: str, title: str, output_path: Path) -> None:
    '''
    Line plot: X axis = exercise types (15), a line for each (architecture modality).
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
# D. Cartesian plots (for single exercise)
# ============================================================================

def plot_cartesian_pearson(results: Dict, output_dir: Path, title_suffix: str = '', active_side: str = 'both') -> None:
    '''
    Bar chart of Pearson correlation per Cartesian wrist coordinate.
    6 coordinates: r_wrist_X, r_wrist_Y, r_wrist_Z, l_wrist_X, l_wrist_Y, l_wrist_Z.

    Unlike the joint-space version, all coordinates carry meaningful values
    here (the wrist always moves in 3D).
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

    # Shows only coordinates where at least one method has |pearson| > 0.05
    visible_mask   = np.any(np.abs(all_vals) > 0.05, axis=0)
    visible_coords = [l for l, v in zip(COORD_LABELS, visible_mask) if v]
    visible_all    = all_vals[:, visible_mask]

    if not visible_coords:
        print('  [skip] plot_cartesian_pearson: all the coordinates are constant')
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
    ax.set_ylabel('Pearson r  (better)', fontsize=11)
    ax.set_title(
        f'Pearson Correlation  Wrist Cartesian Coordinates{title_suffix}\n'
        '(temporal structure, DTW-aligned; invariant to kinematic redundancy)',
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
    """Wrist e Elbow Cartesian trajectories for the active arm (3x2 grid)."""
    from evaluation_and_comparison._metrics import _to_cartesian

    AXIS_LABELS = ['X (forward, m)', 'Y (lateral, m)', 'Z (up, m)']

    # Select active arm
    side = ('right' if active_side == 'right'
            else 'left' if active_side == 'left'
            else 'left')   # default: left

    # Columns: wrist (0) e elbow (1)
    col_titles = [f'{side.capitalize()} wrist', f'{side.capitalize()} elbow']

    # Compute FK for each trajectory 
    cart = {}
    for label, traj in trajs.items():
        if traj is None:
            continue
        rw, re, lw, le = _to_cartesian(traj)
        wrist = rw if side == 'right' else lw
        elbow = re if side == 'right' else le
        cart[label] = {'wrist': wrist, 'elbow': elbow}
    if not cart:
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=False)
    fig.suptitle(f'Cartesian Trajectories - {side.capitalize()} arm{title_suffix}',
                 fontsize=13, fontweight='bold')

    for col_i, endpoint in enumerate(['wrist', 'elbow']):
        axes[0, col_i].set_title(col_titles[col_i], fontsize=11, fontweight='bold')
        for ax_i in range(3):
            ax = axes[ax_i, col_i]
            for label, data in cart.items():
                t = data[endpoint]
                ax.plot(t[:, ax_i],
                        color=PALETTE.get(label, '#7f8c8d'),
                        linewidth=2.5 if label == 'Baseline' else 1.5,
                        linestyle='--' if label == 'Canonical' else '-',
                        label=label,
                        alpha=1.0 if label == 'Baseline' else 0.8)
            if col_i == 0:
                ax.set_ylabel(AXIS_LABELS[ax_i], fontsize=9)
            if ax_i == 2:
                ax.set_xlabel('Frame', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(alpha=0.3)
            if ax_i == 0 and col_i == 1:
                ax.legend(fontsize=8, loc='best')

    fig.tight_layout()
    p = output_dir / 'plot_cartesian_trajectories.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {p.name}')


def plot_cartesian_rmse(results, output_dir, title_suffix='', active_side='both'):
    """Bar chart RMSE cartesian per endpoint of the active arm (meters)."""
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
    ax.set_ylabel('RMSE (meters)  (lower=better)', fontsize=11)
    ax.set_title(
        f'RMSE Cartesian per Endpoint{title_suffix}\n'
        '(invariant to kinematic redundancy)',
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
    Cartesian velocity profile: wrist and elbow in m/frame.
    Only the active arm (from baseline). Wrist continuous, Elbow dashed.
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
            '(bell-shaped = human-like motion; measurement in meters)',
            fontsize=11, fontweight='bold')
        fig.tight_layout()
        p = output_dir / f'plot_cartesian_velocity_{side_tag.lower()}_arm.png'
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved -> {p.name}')


def plot_3d_trajectories(trajs: Dict[str, Optional[np.ndarray]], output_dir: Path, active_side: str = 'left') -> None:
    '''
    Saves an interactive HTML file with 5 3D subplots (one per method).
    Each subplot shows the wrist (solid line) and elbow (dashed line)
    of the active arm. Open in a browser to rotate/zoom.
    Output: exercise_XXX/n_XX/evaluation/plot_3d_trajectories.html
    '''
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("  [skip] plot_3d_trajectories: plotly not installed. "
              "Install with: pip install plotly")
        return

    from evaluation_and_comparison._metrics import _to_cartesian

    ORDER = ['Baseline', 'Canonical', 'MLP', 'GRU', 'Transformer']
    items = [(lbl, trajs[lbl]) for lbl in ORDER
             if lbl in trajs and trajs[lbl] is not None]
    if not items:
        return

    # Compute FK for each method
    cart_data = {}
    for label, traj in items:
        rw, re, lw, le = _to_cartesian(traj)
        wrist = rw if active_side == 'right' else lw
        elbow = re if active_side == 'right' else le
        cart_data[label] = {'wrist': wrist, 'elbow': elbow}

    # Global axis consistent across subplots
    all_pts = np.vstack([v for d in cart_data.values() for v in d.values()])
    pad = 0.05
    axis_range = {
        'x': [all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad],
        'y': [all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad],
        'z': [all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad],
    }

    n    = len(items)
    specs = [[{'type': 'scene'}] * n]
    col_titles = [lbl for lbl, _ in items]

    fig = make_subplots(
        rows=1, cols=n,
        specs=specs,
        subplot_titles=col_titles,
        horizontal_spacing=0.02,
    )

    axis_style = dict(
        showgrid=True,
        gridcolor='lightgrey',
        gridwidth=1,
        showbackground=True,
        backgroundcolor='rgb(245,245,245)',
        showline=True,
        linecolor='grey',
        zeroline=True,
        zerolinecolor='grey',
    )

    scene_cfg = dict(
        xaxis=dict(title='X (m)', range=axis_range['x'], **axis_style),
        yaxis=dict(title='Y (m)', range=axis_range['y'], **axis_style),
        zaxis=dict(title='Z (m)', range=axis_range['z'], **axis_style),
        aspectmode='cube',
    )

    for col_i, (label, _) in enumerate(items):
        scene_n  = 'scene' if col_i == 0 else f'scene{col_i + 1}'
        color    = PALETTE.get(label, '#7f8c8d')
        data     = cart_data[label]
        show_leg = (col_i == 0)

        for endpoint, dash, opacity, name in [
            ('wrist', 'solid', 0.95, 'wrist'),
            ('elbow', 'dash',  0.65, 'elbow'),
        ]:
            pts = data[endpoint]
            # Frame indices per tooltip
            hover = [f'frame {i}<br>x={pts[i,0]:.3f} y={pts[i,1]:.3f} z={pts[i,2]:.3f}'
                     for i in range(len(pts))]

            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='lines',
                line=dict(color=color, width=4 if dash == 'solid' else 2,
                          dash=dash),
                opacity=opacity,
                name=name,
                legendgroup=name,
                showlegend=show_leg,
                hovertemplate='%{text}<extra>' + name + '</extra>',
                text=hover,
                scene=scene_n,
            ), row=1, col=col_i + 1)

            # Markers start and end points
            for pt, sym in [(pts[0], 'circle'), (pts[-1], 'square')]:
                fig.add_trace(go.Scatter3d(
                    x=[pt[0]], y=[pt[1]], z=[pt[2]],
                    mode='markers',
                    marker=dict(size=5, color=color, symbol=sym,
                                opacity=0.9 if dash == 'solid' else 0.5),
                    showlegend=False,
                    hoverinfo='skip',
                    scene=scene_n,
                ), row=1, col=col_i + 1)

        fig.update_layout(**{scene_n: scene_cfg})

    side_label = active_side.capitalize()
    fig.update_layout(
        title=dict(
            text=f'3D Trajectories -- {side_label} arm   '
                 f'[wrist (solid) / elbow (dashed)]',
            font=dict(size=14),
            x=0.5,
        ),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'),
        height=550,
        margin=dict(l=10, r=10, t=80, b=10),
        paper_bgcolor='white',
        font=dict(family='Arial', size=11),
    )

    p = output_dir / 'plot_3d_trajectories.html'
    fig.write_html(str(p), include_plotlyjs='cdn')
    print(f'  Saved -> {p.name}  (open in browser)')



def plot_summary_heatmap(results: Dict, output_dir: Path, title_suffix: str = '', active_side: str = 'both', human_bounds: Optional[Dict] = None, arch_variance: Optional[Dict] = None) -> None:
    '''
    Summary heatmap: rows = methods, columns = key scalar metrics.

    Per-column bounds (from human_bounds, computed on individual demos):
      - green = best human demo performance on that metric
      - red   = worst human demo performance
      - gray  = beyond the worst bound
      - Pearson: fixed [0, 1]
    Human demos do not appear as a row.
    '''
    if human_bounds is None:
        human_bounds = {}
    s = active_side
    wrist_key = 'cart_rmse_l_wrist' if s == 'left' else 'cart_rmse_r_wrist'
    elbow_key = 'cart_rmse_l_elbow' if s == 'left' else 'cart_rmse_r_elbow'
    peak_key  = 'cart_peak_l_wrist' if s == 'left' else 'cart_peak_r_wrist'
    wrist_lbl = 'RMSE\nLw (m)'     if s == 'left' else 'RMSE\nRw (m)'
    elbow_lbl = 'RMSE\nLe (m)'     if s == 'left' else 'RMSE\nRe (m)'
    peak_lbl  = 'Peak\nLw (m)'     if s == 'left' else 'Peak\nRw (m)'

    # (key, label, lower_is_better, format, bound_type)
    METRICS = [
        ('cart_dtw',          'DTW\n(m)',        True,  '{:.3f}', 'human'),
        (wrist_key,           wrist_lbl,         True,  '{:.4f}', 'human'),
        (elbow_key,           elbow_lbl,         True,  '{:.4f}', 'human'),
        ('rmse_mean',         'RMSE\njoint (°)', True,  '{:.2f}', 'human'),
        (peak_key,            peak_lbl,          True,  '{:.4f}', 'human'),
        ('smoothness',        'Smooth\n(↑)',     False, '{:.4f}', 'human'),
        ('cart_pearson_mean', 'Pearson\ncart',   False, '{:.3f}', 'fixed_01'),
        ('pearson_mean',      'Pearson\njoint',  False, '{:.3f}', 'fixed_01'),
    ]

    METHOD_ORDER = ['Canonical', 'CanonicalShape', 'MLP', 'GRU', 'Transformer']
    methods = [m for m in METHOD_ORDER if m in results] + \
              [m for m in results if m not in METHOD_ORDER and m != 'Human demos']

    if not methods:
        return

    human_r = results.get('Human demos', {})  # fallback
    n_m, n_c = len(methods), len(METRICS)

    # Matrix of data and scores
    data   = np.full((n_m, n_c), np.nan)
    scores = np.full((n_m, n_c), np.nan)   # [0,1]=valid, -1=out-of-bounds

    for i, m in enumerate(methods):
        for j, (key, _, lib, _, bt) in enumerate(METRICS):
            val = results[m].get(key, np.nan)
            if np.isnan(val):
                continue
            data[i, j] = val
            best, worst = _get_metric_bounds(key, lib, bt, human_bounds)
            scores[i, j] = _score_value(val, best, worst, lib)

    col_labels = [lbl for _, lbl, _, _, _ in METRICS]
    formats    = [fmt for _, _, _, fmt, _ in METRICS]

    fig, ax = plt.subplots(figsize=(max(10, n_c * 1.5), max(3.5, n_m * 1.1)))

    # Imshow only for valid cells [0,1]
    display = np.where(scores == -1, np.nan, scores)
    im = ax.imshow(display, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

    ax.set_xticks(range(n_c))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(range(n_m))
    ax.set_yticklabels(methods, fontsize=11)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    for i in range(n_m):
        for j, fmt in enumerate(formats):
            key   = METRICS[j][0]
            val   = data[i, j]
            score = scores[i, j]
            if np.isnan(val):
                continue

            # ±std from arch_variance (only for BC architectures, not canonical)
            method  = methods[i]
            var_d   = (arch_variance or {}).get(method, None) if arch_variance else None
            std_val = var_d.get(key, np.nan) if var_d else np.nan
            std_txt = f'±{fmt.format(std_val)}' if not np.isnan(std_val) else ''

            val_txt  = fmt.format(val)
            cell_txt = f'{val_txt}\n{std_txt}' if std_txt else val_txt

            if score == -1.0:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=True, facecolor='#888888',
                    edgecolor='white', linewidth=1.5, zorder=2))
                ax.text(j, i, cell_txt,
                        ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold', zorder=3,
                        linespacing=1.3)
            else:
                txt_color = 'white' if (score < 0.25 or score > 0.75) else 'black'
                ax.text(j, i, cell_txt,
                        ha='center', va='center', fontsize=8.5,
                        color=txt_color, fontweight='bold',
                        linespacing=1.3)

    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Performance  (green = best)', fontsize=9)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(['worst\ndemo', 'medium', 'best\ndemo'], fontsize=8)

    ax.set_title(
        f'Summary Metrics{title_suffix}\n'
        '(green/red = range human demos  |  gray = beyond worst limit)',
        fontsize=12, fontweight='bold', pad=18)

    for i in range(n_m - 1):
        ax.axhline(i + 0.5, color='white', linewidth=1.5)
    for j in range(n_c - 1):
        ax.axvline(j + 0.5, color='white', linewidth=1.5)

    fig.tight_layout()
    p = output_dir / 'plot_summary_heatmap.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {p.name}')

# ============================================================================
# Shared helpers per heatmap + spider (bounds based on Human demos)
# ============================================================================

def _get_metric_bounds(key: str, lower_is_better: bool, bound_type: str, human_bounds: Dict) -> tuple:
    '''
    Returns (best, worst) for a metric.
      bound_type='human'    → (best_human_demo, worst_human_demo) from human_bounds
      bound_type='fixed_01' → (1.0, 0.0) for Pearson
    If the key is not in human_bounds, returns (nan, nan) → gray cells.
    '''
    if bound_type == 'fixed_01':
        return (1.0, 0.0)
    best, worst = human_bounds.get(key, (np.nan, np.nan))
    return (best, worst)


def _score_value(val: float, best: float, worst: float, lower_is_better: bool) -> float:
    '''
    Normalizes val to [0, 1] where:
      1 = best human demo performance  (green)
      0 = worst human demo performance (red)
    Returns -1 if val exceeds the worst bound (gray cell).
    Returns np.nan if val, best, or worst are NaN.
    '''
    if np.isnan(val) or np.isnan(best) or np.isnan(worst):
        return np.nan
    span = abs(best - worst)
    if span < 1e-12:
        return 0.5
    if lower_is_better:
        # best=min, worst=max  →  score = (worst - val) / span
        if val > worst:
            return -1.0   # grey
        return float(np.clip((worst - val) / span, 0.0, 1.0))
    else:
        # best=max, worst=min  →  score = (val - worst) / span
        if val < worst:
            return -1.0   # grey
        return float(np.clip((val - worst) / span, 0.0, 1.0))


def plot_spider_chart(results: Dict, output_dir: Path, title_suffix: str = '', active_side: str = 'both', human_bounds: Optional[Dict] = None) -> None:
    '''
    Spider / radar chart - all Cartesian metrics (active side).
    Axes: DTW cart, RMSE wrist, RMSE elbow, Peak wrist, Pearson cart.

    Score in [0, 1]:
      1 = best human demo on that metric   (green outer)
      0 = worst human demo                 (red center)
    Out-of-bound values → clamped to 0. Human demos not shown.
    '''
    if human_bounds is None:
        human_bounds = {}

    s         = active_side
    wrist_key = 'cart_rmse_l_wrist' if s == 'left' else 'cart_rmse_r_wrist'
    elbow_key = 'cart_rmse_l_elbow' if s == 'left' else 'cart_rmse_r_elbow'
    peak_key  = 'cart_peak_l_wrist' if s == 'left' else 'cart_peak_r_wrist'
    wrist_lbl = 'RMSE\nLw (m)'     if s == 'left' else 'RMSE\nRw (m)'
    elbow_lbl = 'RMSE\nLe (m)'     if s == 'left' else 'RMSE\nRe (m)'
    peak_lbl  = 'Peak\nLw (m)'     if s == 'left' else 'Peak\nRw (m)'

    SPIDER_METRICS = [
        # (key, label, lower_is_better, bound_type)
        ('cart_dtw',          'DTW\ncart (m)',    True,  'human'),
        (wrist_key,           wrist_lbl,          True,  'human'),
        (elbow_key,           elbow_lbl,          True,  'human'),
        (peak_key,            peak_lbl,           True,  'human'),
        ('cart_pearson_mean', 'Pearson\ncart',    False, 'fixed_01'),
    ]

    METHOD_ORDER = ['Canonical', 'CanonicalShape', 'MLP', 'GRU', 'Transformer']
    methods = [m for m in METHOD_ORDER if m in results and m != 'Human demos']

    if not methods:
        return

    N      = len(SPIDER_METRICS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    labels  = [lbl for _, lbl, _, _ in SPIDER_METRICS]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for method in methods:
        scores = []
        for key, _, lib, bt in SPIDER_METRICS:
            val         = results[method].get(key, np.nan)
            best, worst = _get_metric_bounds(key, lib, bt, human_bounds)
            sv          = _score_value(val, best, worst, lib)
            # out-of-bounds or nan → 0
            if isinstance(sv, float) and not np.isnan(sv) and sv != -1.0:
                scores.append(float(np.clip(sv, 0.0, 1.0)))
            else:
                scores.append(0.0)
        scores += scores[:1]

        color = PALETTE.get(method, '#7f8c8d')
        ax.plot(angles, scores, 'o-', linewidth=2.0,
                color=color, label=method, alpha=0.85)
        ax.fill(angles, scores, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.00])
    ax.set_yticklabels(['0.25\n(worst)', '0.50', '0.75', '1.00\n(best)'],
                       fontsize=7.5, color='grey')
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.6)

    ax.legend(loc='upper right', bbox_to_anchor=(1.40, 1.18), fontsize=10)
    ax.set_title(
        f'Spider Chart - Cartesian Performance{title_suffix}\n'
        '(1 = best human demo,  0 = worst human demo)',
        fontsize=12, fontweight='bold', pad=22)

    fig.tight_layout()
    p = output_dir / 'plot_spider_chart.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved -> {p.name}')
