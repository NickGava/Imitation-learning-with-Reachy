'''
visualize_joints.py
=============================================================================
Plots joint angles from joint_ik.csv for a selected recording.

Layout:
  Left  column : right arm joints  (r_*)
  Right column : left  arm joints  (l_*)
  One subplot per joint, Y axis fixed to the same scale across all subplots
  (reference: joint with the largest range of motion in this recording).

Input:
  _data/landmarks/subject_XXX/exercise_XXX/video_XXX/joint_ik.csv

Output:
  _data/landmarks/subject_XXX/exercise_XXX/video_XXX/plots/joints.png

Usage:
  py visualize_joints.py
  → prompts for subject / exercise / video
'''

#  Imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from config import DATA_ROOT, JOINT_LIMITS_DEG
from ask_inputs import ask_inputs

#  Constants 
# Joints to plot: (column_name, index_in_JOINT_LIMITS_DEG, side)
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

# Human-readable labels
JOINT_LABELS = {
    'r_shoulder_pitch': 'Shoulder Pitch',
    'r_shoulder_roll':  'Shoulder Roll',
    'r_arm_yaw':        'Arm Yaw',
    'r_elbow_pitch':    'Elbow Pitch',
    'l_shoulder_pitch': 'Shoulder Pitch',
    'l_shoulder_roll':  'Shoulder Roll',
    'l_arm_yaw':        'Arm Yaw',
    'l_elbow_pitch':    'Elbow Pitch',
}

# Plot style
LINE_COLOR_R = '#1f77b4'    # blue  – right arm
LINE_COLOR_L = '#d62728'    # red   – left arm
LINE_WIDTH   = 1.5
Y_PADDING = 5.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    #  Ask inputs 
    subject_name, exercise_name, video_name = ask_inputs()

    #  Build paths 
    video_dir  = DATA_ROOT / 'landmarks' / subject_name / exercise_name / video_name
    ik_path    = video_dir / 'joint_ik.csv'
    plot_dir   = video_dir / 'plots'

    if not ik_path.exists():
        print(f'ERROR: joint_ik.csv not found -> {ik_path}')
        return

    #  Load data 
    df     = pd.read_csv(ik_path)
    frames = df['frame'].values

    # Keep only joints present in this CSV
    r_joints = [(col, idx, side) for col, idx, side in RIGHT_JOINTS if col in df.columns]
    l_joints = [(col, idx, side) for col, idx, side in LEFT_JOINTS  if col in df.columns]

    if not r_joints and not l_joints:
        print('ERROR: no recognisable joint columns found in joint_ik.csv')
        return

    n_rows = max(len(r_joints), len(l_joints))

    #  Figure 
    fig, axes = plt.subplots(
        nrows   = n_rows,
        ncols   = 2,
        figsize = (14, 2.8 * n_rows),
        sharex  = False,
    )
    if n_rows == 1:
        axes = np.array([axes])

    fig.suptitle(
        f'Joint angles – {subject_name} / {exercise_name} / {video_name}\n'
        f'(Y scale = robot joint limits)',
        fontsize=13, fontweight='bold', y=1.01,
    )

    axes[0, 0].set_title('Right arm', fontsize=12, fontweight='bold', pad=8)
    axes[0, 1].set_title('Left arm',  fontsize=12, fontweight='bold', pad=8)

    #  Plot each joint 
    for row_idx in range(n_rows):

        # ----- Left column: right arm -----
        ax_r = axes[row_idx, 0]
        if row_idx < len(r_joints):
            col, lim_idx, side = r_joints[row_idx]
            ax_r.plot(frames, df[col].values, color=LINE_COLOR_R, linewidth=LINE_WIDTH)
            ax_r.set_ylabel(f'{JOINT_LABELS.get(col, col)}\n(deg)', fontsize=9)
            y_min, y_max = JOINT_LIMITS_DEG[side][lim_idx]
            ax_r.set_ylim(y_min - Y_PADDING, y_max + Y_PADDING)
        else:
            ax_r.set_visible(False)

        # ----- Right column: left arm -----
        ax_l = axes[row_idx, 1]
        if row_idx < len(l_joints):
            col, lim_idx, side = l_joints[row_idx]
            ax_l.plot(frames, df[col].values, color=LINE_COLOR_L, linewidth=LINE_WIDTH)
            y_min, y_max = JOINT_LIMITS_DEG[side][lim_idx]
            ax_l.set_ylim(y_min - Y_PADDING, y_max + Y_PADDING) 
        else:
            ax_l.set_visible(False)

        # ----- Grid and ticks -----
        for ax in (ax_r, ax_l):
            if ax.get_visible():
                ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(10))
                ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
                ax.grid(True, which='minor', linestyle=':',  linewidth=0.4, alpha=0.4)
                ax.tick_params(axis='y', labelsize=8)

    #  X axis: tick labels everywhere, 'Frame' label only on bottom 
    for row_idx in range(n_rows):
        for col_idx in range(2):
            ax = axes[row_idx, col_idx]
            if ax.get_visible():
                ax.tick_params(axis='x', labelsize=8)
                if row_idx == n_rows - 1:
                    ax.set_xlabel('Frame', fontsize=9)

    #  Save 
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_path = plot_dir / 'joints.png'
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved -> {output_path.relative_to(DATA_ROOT)}')


if __name__ == '__main__':
    main()