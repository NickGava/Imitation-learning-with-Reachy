'''
visualize_fk_dataset.py
=============================================================================
FK verification plots for the baseline and canonical trajectories of a given
exercise. Mirrors _plot_fk_verification() from create_baselines.py, but saves
the figures to disk instead of showing them interactively.

For each CSV found in data/dataset/exercise_XXX/ (baseline.csv, canonical.csv),
the script computes forward kinematics on every frame and plots wrist and elbow
trajectories in the Reachy torso frame (X=forward, Y=left, Z=up).

Input:
  _data/dataset/exercise_XXX/baseline.csv
  _data/dataset/exercise_XXX/canonical.csv

Output:
  _data/dataset/exercise_XXX/plots/baseline_fk.png
  _data/dataset/exercise_XXX/plots/canonical_fk.png

Usage:
  python visualize_dataset_plot.py --exercise 1
'''

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from config import DATA_ROOT, JOINT_COLS

JOINT_COLS = [
    'r_shoulder_pitch', 'r_shoulder_roll', 'r_arm_yaw', 'r_elbow_pitch',
    'l_shoulder_pitch', 'l_shoulder_roll', 'l_arm_yaw', 'l_elbow_pitch',
]
from data_acquisition.run_ik import fk


# ---------------------------------------------------------------------------
# FK figure builder  (same logic as _plot_fk_verification in create_baselines.py,
#                     but returns the Figure instead of calling plt.show())
# ---------------------------------------------------------------------------
def _fk_figure(trajectory: np.ndarray, exercise_num: int, label: str) -> plt.Figure:
    '''
    Computes FK for every frame and plots wrist / elbow trajectories in 3D
    (Reachy torso frame) for both arms.

    Parameters:
        trajectory   : (N, 16) array of joint angles in degrees (JOINT_COLS order)
        exercise_num : exercise number, used in the figure title
        label        : 'baseline' or 'canonical', shown in the title

    Returns:
        fig : matplotlib Figure (caller is responsible for closing it)
    '''
    r_wrist, r_elbow = [], []
    l_wrist, l_elbow = [], []

    for q_deg in trajectory:
        # Right arm: joints 0-3 (shoulder_pitch … elbow_pitch); pad to 7 for fk()
        q_r = np.deg2rad(np.pad(q_deg[:4], (0, 3)))
        e_r, w_r = fk(q_r, 'right')
        r_elbow.append(e_r)
        r_wrist.append(w_r)

        # Left arm: joints 4-7; pad to 7 for fk()
        q_l = np.deg2rad(np.pad(q_deg[4:8], (0, 3)))
        e_l, w_l = fk(q_l, 'left')
        l_elbow.append(e_l)
        l_wrist.append(w_l)

    r_wrist = np.array(r_wrist)
    r_elbow = np.array(r_elbow)
    l_wrist = np.array(l_wrist)
    l_elbow = np.array(l_elbow)

    frames = np.arange(len(trajectory))

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    fig.suptitle(
        f'FK Verification – Exercise {exercise_num:03d}  [{label}]\n'
        f'(Reachy torso frame: X=forward, Y=left, Z=up)',
        fontsize=11, fontweight='bold',
    )

    axis_labels = ['X (forward, m)', 'Y (lateral, m)', 'Z (up, m)']
    sides = [
        ('Right arm', r_wrist, r_elbow),
        ('Left arm',  l_wrist, l_elbow),
    ]

    for dim in range(3):
        for col_idx, (side_title, wrist, elbow) in enumerate(sides):
            ax = axes[dim, col_idx]
            ax.plot(frames, wrist[:, dim], label='wrist', color='#3498db', linewidth=1.5)
            ax.plot(frames, elbow[:, dim], label='elbow', color='#e67e22', linewidth=1.5, linestyle='--')
            ax.set_ylabel(axis_labels[dim], fontsize=9)
            if dim == 0:
                ax.set_title(side_title, fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    for ax in axes[2]:
        ax.set_xlabel('Frame', fontsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_trajectory(csv_path: Path) -> np.ndarray:
    '''
    Reads JOINT_COLS from a CSV file.

    Returns:
        (N, 16) float array in degrees
    Raises:
        ValueError if any expected column is missing
    '''
    df = pd.read_csv(csv_path)
    missing = [c for c in JOINT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns in {csv_path.name}: {missing}')
    return df[JOINT_COLS].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='FK verification plots for baseline and canonical trajectories.'
    )
    parser.add_argument('--exercise', type=int, required=True, help='Exercise number.')
    args = parser.parse_args()

    exercise_name = f'exercise_{args.exercise:03d}'
    exercise_dir  = DATA_ROOT / 'dataset' / exercise_name

    if not exercise_dir.exists():
        print(f'Error: folder not found → {exercise_dir}')
        return

    output_dir = exercise_dir / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        ('baseline',  exercise_dir / 'baseline.csv'),
        ('canonical', exercise_dir / 'canonical.csv'),
    ]

    any_saved = False
    for label, csv_path in sources:
        if not csv_path.exists():
            print(f'[SKIP] {csv_path.name} not found in {exercise_dir.name}/')
            continue

        print(f'Processing {label} ...')
        try:
            trajectory = _load_trajectory(csv_path)
        except ValueError as e:
            print(f'[ERROR] {e}')
            continue

        fig      = _fk_figure(trajectory, args.exercise, label)
        out_path = output_dir / f'{label}_fk.png'
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.show()
        plt.close(fig)
        print(f'  Saved → {out_path.relative_to(DATA_ROOT)}')
        any_saved = True

    if not any_saved:
        print('No CSVs found. Run create_baselines.py and/or compute_canonical.py first.')
    else:
        print('Done.')


if __name__ == '__main__':
    main()