'''
create_baseline.py
=============================================================================
Creates a baseline trajectory for one exercise by interpolating between
user-defined keyframes (in joint space) with a minimum jerk profile.

Keyframe format:
  Each keyframe is a dict:
    {
      'duration': float,        # seconds to reach this pose from previous
      'r_shoulder_pitch': float,
      'r_shoulder_roll':  float,
      ... (any joint can be omitted - defaults to previous keyframe value)
    }

Output:
  _data/dataset/exercise_XXX/baseline.csv

Usage:
  python create_baseline.py --exercise 1                # generate and save baseline for exercise 1
  python create_baseline.py --exercise 1 --fps 25       # custom frame rate (default: 30 fps)
'''

import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from typing import List, Dict

from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode

from data_acquisition.run_ik import fk
from utilities.config import DATA_ROOT, JOINT_LIMITS_DEG, DEFAULT_FPS, JOINT_COLS, HEAD_NEUTRAL, GRIPPER_RANGE, STARTING_POSE
from create_baselines.exercises import EXERCISES


# ---------------------------------------------------------------------------
# Joint angles plot for canonical trajectory
# ---------------------------------------------------------------------------
RIGHT_JOINTS = [
    ('r_shoulder_pitch', 0, 'right'),
    ('r_shoulder_roll',  1, 'right'),
    ('r_arm_yaw',        2, 'right'),
    ('r_elbow_pitch',    3, 'right'),
]
LEFT_JOINTS = [
    ('l_shoulder_pitch', 0, 'left'),
    ('l_shoulder_roll',  1, 'left'),
    ('l_arm_yaw',        2, 'left'),
    ('l_elbow_pitch',    3, 'left'),
]
JOINT_LABELS = {
    'r_shoulder_pitch': 'Shoulder Pitch', 'r_shoulder_roll': 'Shoulder Roll',
    'r_arm_yaw':        'Arm Yaw',        'r_elbow_pitch':   'Elbow Pitch',
    'l_shoulder_pitch': 'Shoulder Pitch', 'l_shoulder_roll': 'Shoulder Roll',
    'l_arm_yaw':        'Arm Yaw',        'l_elbow_pitch':   'Elbow Pitch',
}
LINE_COLOR_CANONICAL = '#2ca02c'
Y_PADDING = 5.0


# ---------------------------------------------------------------------------
# Minimum jerk interpolation
# ---------------------------------------------------------------------------
def _min_jerk(t_norm: np.ndarray) -> np.ndarray:
    '''
    Minimum jerk profile: s(t) = 10t³ - 15t⁴ + 6t⁵ (teoria del minimum jerk di Flash & Hogan (1985))
    Input  t_norm : array of values in [0, 1]
    Output s      : array of values in [0, 1]
    '''
    t = np.clip(t_norm, 0.0, 1.0)
    return 10*t**3 - 15*t**4 + 6*t**5


def _interpolate_segment(q_start: np.ndarray, q_end: np.ndarray, duration: float, fps: int) -> np.ndarray:
    '''
    Interpolates from q_start to q_end over 'duration' seconds at 'fps'.
    Returns array of shape (N_frames, 16).
    Last frame (q_end) is excluded to avoid duplication at segment joints.
    '''
    n_frames = max(1, int(round(duration * fps)))
    t_norm   = np.linspace(0.0, 1.0, n_frames + 1)[:-1]   # exclude last
    s        = _min_jerk(t_norm)                          # shape: (n_frames,)
    return q_start + s[:, None] * (q_end - q_start)       # shape: (n_frames, 16)


# ---------------------------------------------------------------------------
# Keyframe expansion
# ---------------------------------------------------------------------------
def _expand_keyframes(keyframes: List[Dict], fps: int) -> np.ndarray:
    '''
    Converts a list of keyframe dicts into a full trajectory array.

    Parameters:
        keyframes : list of dicts.
                    keyframes[0] has NO 'duration' and defines the starting pose
                    (the trajectory begins here, with no interpolation from STARTING_POSE).
                    All subsequent keyframes must have 'duration' and are interpolated
                    from the previous pose using a minimum-jerk profile.
                    Unspecified joints inherit from the previous keyframe.
        fps       : frames per second

    Returns:
        trajectory : (N_total_frames, 16) array of joint angles in degrees
    '''
    def _apply_kf(kf: Dict, base: np.ndarray) -> np.ndarray:
        '''Apply a keyframe dict on top of a base pose, return new pose.'''
        pose = base.copy()
        for i, col in enumerate(JOINT_COLS):
            if col in kf:
                pose[i] = float(kf[col])
        if 'gripper_open' in kf:
            pose[7]  = GRIPPER_RANGE['right_hand']['open'] if kf['gripper_open'] else GRIPPER_RANGE['right_hand']['closed']
            pose[15] = GRIPPER_RANGE['left_hand']['open']  if kf['gripper_open'] else GRIPPER_RANGE['left_hand']['closed']
        return pose

    default_pose = np.array([STARTING_POSE[c] for c in JOINT_COLS], dtype=float)

    # --- First keyframe: starting pose (no interpolation) ---
    q_start  = _apply_kf(keyframes[0], default_pose)
    segments = [q_start[None, :]]   # single frame
    q_prev   = q_start

    # --- Remaining keyframes: interpolate ---
    for kf in keyframes[1:]:
        q_end    = _apply_kf(kf, q_prev)
        duration = float(kf.get('duration', 1.0))
        seg      = _interpolate_segment(q_prev, q_end, duration, fps)
        segments.append(seg)
        q_prev   = q_end

    # Add the final pose as the last frame
    segments.append(q_prev[None, :])

    return np.vstack(segments)   # shape: (N, 16)


# ---------------------------------------------------------------------------
# Joint limits check
# ---------------------------------------------------------------------------
def _check_limits(trajectory: np.ndarray) -> bool:
    '''
    Warns if any frame violates Reachy joint limits.
    Returns True if all frames are within limits.
    '''
    ok = True
    for side, indices in [('right', range(7)), ('left', range(8, 15))]:
        limits = JOINT_LIMITS_DEG[side]
        for local_i, global_i in enumerate(indices):
            col_name = JOINT_COLS[global_i]
            vals     = trajectory[:, global_i]
            lo, hi   = limits[local_i]
            if np.any(vals < lo) or np.any(vals > hi):
                vmin, vmax = vals.min(), vals.max()
                print(f'[LIMIT WARNING] {col_name}: '
                      f'range [{vmin:.1f}°, {vmax:.1f}°] '
                      f'exceeds limits [{lo:.1f}°, {hi:.1f}°]')
                ok = False
    if ok:
        print('Joint limits: all OK')
    return ok


# ---------------------------------------------------------------------------
# FK verification plot
# ---------------------------------------------------------------------------
def _plot_fk_verification(trajectory: np.ndarray, exercise_num: int, output_dir: Path) -> None:
    '''
    Computes FK for every frame and plots wrist and elbow trajectories in 3D (torso frame) for both arms.
    All Y axes share the same scale. X-axis label is shown on every subplot.
    Saves the figure to output_dir/baseline.png automatically.
    '''
    r_wrist, r_elbow = [], []
    l_wrist, l_elbow = [], []

    for q_deg in trajectory:
        # Right arm: joints 0-6 (shoulder_pitch to wrist_roll)
        q_r = np.deg2rad(q_deg[:7])
        e_r, w_r = fk(q_r, 'right')
        r_elbow.append(e_r)
        r_wrist.append(w_r)

        # Left arm: joints 8-14
        q_l = np.deg2rad(q_deg[8:15])
        e_l, w_l = fk(q_l, 'left')
        l_elbow.append(e_l)
        l_wrist.append(w_l)

    r_wrist  = np.array(r_wrist)
    r_elbow  = np.array(r_elbow)
    l_wrist  = np.array(l_wrist)
    l_elbow  = np.array(l_elbow)

    frames = np.arange(len(trajectory))

    # Global Y range across all FK data so every subplot uses the same scale
    all_vals = np.concatenate([r_wrist, r_elbow, l_wrist, l_elbow], axis=0)
    y_min, y_max = all_vals.min(), all_vals.max()
    y_pad = (y_max - y_min) * 0.05 or 0.05   # 5 % padding, at least 5 cm
    y_lim = (y_min - y_pad, y_max + y_pad)

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=False)
    fig.suptitle(f'FK Verification - Exercise {exercise_num:03d}\n'
                 f'(Reachy torso frame: X=forward, Y=left, Z=up)', fontsize=11, fontweight='bold')

    labels = ['X (forward, m)', 'Y (lateral, m)', 'Z (up, m)']

    for dim in range(3):
        # Right arm
        ax = axes[dim, 0]
        ax.plot(frames, r_wrist[:, dim],  label='wrist',  color='#3498db', linewidth=1.5)
        ax.plot(frames, r_elbow[:, dim],  label='elbow',  color='#e67e22', linewidth=1.5, linestyle='--')
        ax.set_ylabel(labels[dim], fontsize=9)
        ax.set_xlabel('Frame', fontsize=9)
        ax.set_ylim(*y_lim)
        if dim == 0:
            ax.set_title('Right arm', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Left arm
        ax = axes[dim, 1]
        ax.plot(frames, l_wrist[:, dim],  label='wrist',  color='#3498db', linewidth=1.5)
        ax.plot(frames, l_elbow[:, dim],  label='elbow',  color='#e67e22', linewidth=1.5, linestyle='--')
        ax.set_xlabel('Frame', fontsize=9)
        ax.set_ylim(*y_lim)
        if dim == 0:
            ax.set_title('Left arm', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout()

    # Auto-save PNG
    plots_dir = output_dir 
    plots_dir.mkdir(parents=True, exist_ok=True)
    png_path  = plots_dir / 'baseline.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f'Plot saved  -> {png_path}')

    # plt.show()


def _plot_joints_baseline(trajectory: np.ndarray, output_dir: Path, exercise_num: int) -> None:
    '''
    Plots joint angles of the baseline trajectory using the same layout as
    plot_joints.py. Saves to output_dir/joints_baseline.png.
    '''
    df     = pd.DataFrame(trajectory, columns=JOINT_COLS)
    frames = np.arange(len(df))

    n_rows = max(len(RIGHT_JOINTS), len(LEFT_JOINTS))
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=2,
        figsize=(14, 2.8 * n_rows),
        sharex=False,
    )
    if n_rows == 1:
        axes = np.array([axes])

    fig.suptitle(
        f'Joint angles (baseline) - exercise_{exercise_num:03d}\n'
        f'(Y scale = robot joint limits)',
        fontsize=13, fontweight='bold', y=1.01,
    )
    axes[0, 0].set_title('Right arm', fontsize=12, fontweight='bold', pad=8)
    axes[0, 1].set_title('Left arm',  fontsize=12, fontweight='bold', pad=8)

    for row_idx in range(n_rows):
        ax_r = axes[row_idx, 0]
        if row_idx < len(RIGHT_JOINTS):
            col, lim_idx, side = RIGHT_JOINTS[row_idx]
            if col in df.columns:
                ax_r.plot(frames, df[col].values, color=LINE_COLOR_CANONICAL, linewidth=1.5)
            ax_r.set_ylabel(f'{JOINT_LABELS.get(col, col)}\n(deg)', fontsize=9)
            y_min, y_max = JOINT_LIMITS_DEG[side][lim_idx]
            ax_r.set_ylim(y_min - Y_PADDING, y_max + Y_PADDING)
        else:
            ax_r.set_visible(False)

        ax_l = axes[row_idx, 1]
        if row_idx < len(LEFT_JOINTS):
            col, lim_idx, side = LEFT_JOINTS[row_idx]
            if col in df.columns:
                ax_l.plot(frames, df[col].values, color=LINE_COLOR_CANONICAL, linewidth=1.5)
            y_min, y_max = JOINT_LIMITS_DEG[side][lim_idx]
            ax_l.set_ylim(y_min - Y_PADDING, y_max + Y_PADDING)
        else:
            ax_l.set_visible(False)

        for ax in (ax_r, ax_l):
            if ax.get_visible():
                ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
                ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
                ax.grid(True, which='minor', linestyle=':',  linewidth=0.4, alpha=0.4)
                ax.tick_params(axis='both', labelsize=8)
                if row_idx == n_rows - 1:
                    ax.set_xlabel('Frame', fontsize=9)

    plots_dir = output_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_dir / 'joints_baseline.png'
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Plot saved  -> {output_path}')


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def _save_baseline(trajectory: np.ndarray, fps: int, exercise_num: int, output_dir: Path) -> Path:
    '''Saves the trajectory as baseline.csv.'''
    N          = len(trajectory)
    timestamps = np.arange(N) / fps

    out = pd.DataFrame({'frame':     np.arange(N), 'timestamp': timestamps})
    for j, col in enumerate(JOINT_COLS):
        out[col] = trajectory[:, j]

    out['head_x'] = HEAD_NEUTRAL[0]
    out['head_y'] = HEAD_NEUTRAL[1]
    out['head_z'] = HEAD_NEUTRAL[2]

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'baseline.csv'
    out.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Create baseline trajectory from keyframes.')
    parser.add_argument('--exercise', type=int, required=True, help='Exercise number (must be defined in EXERCISES dict).')
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS, help=f'Frame rate (default: {DEFAULT_FPS}).')
    args = parser.parse_args()

    if args.exercise not in EXERCISES:
        print(f'Error: exercise {args.exercise} not defined in EXERCISES dict.')
        print(f'Available exercises: {sorted(EXERCISES.keys())}')
        return

    exercise_name = f'exercise_{args.exercise:03d}'
    output_dir    = DATA_ROOT / 'dataset' / exercise_name
    keyframes     = EXERCISES[args.exercise]

    print(f'\n{"="*60}')
    print(f'Create Baseline - Exercise {args.exercise:03d}, Keyframes: {len(keyframes)}, FPS: {args.fps}')
    print(f'{"="*60}')

    # __________ Interpolate __________
    trajectory = _expand_keyframes(keyframes, args.fps)
    duration   = len(trajectory) / args.fps
    print(f'Frames    : {len(trajectory)}')
    print(f'Duration  : {duration:.2f} s')

    # __________ Joint limits check __________
    print('\nChecking joint limits ...')
    if not _check_limits(trajectory):
        print('Trajectory violates joint limits - aborting.')
        return

    # __________ FK verification plot __________
    print('\nPlotting FK verification ...')
    _plot_fk_verification(trajectory, args.exercise, output_dir)

    # __________ Save __________
    path = _save_baseline(trajectory, args.fps, args.exercise, output_dir)
    print(f'\nSaved -> {path}')
    print(f'Frames: {len(trajectory)}  Duration: {duration:.2f}s')

    # __________ Joint angles plot for baseline __________
    print('\nPlotting baseline joint angles ...')
    _plot_joints_baseline(trajectory, output_dir, args.exercise)

    print('\nDone.')


if __name__ == '__main__':
    main()