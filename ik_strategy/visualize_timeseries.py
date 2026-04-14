'''
visualize_timeseries.py  (v2)
=============================================================================
2D time-series visualization of landmark data in WORLD FRAME: X=left, Y=down, Z=backward.

Figures produced:
  pose.png       - shoulder, elbow, wrist + nose per side  
  right_hand.png - all 5 right hand landmarks              
  left_hand.png  - all 5 left hand landmarks               

Input:
  _data/landmarks/subject_XXX/exercise_XXX/video_XXX/pose_cleaned.csv
  _data/landmarks/subject_XXX/exercise_XXX/video_XXX/right_hand_cleaned.csv
  _data/landmarks/subject_XXX/exercise_XXX/video_XXX/left_hand_cleaned.csv

Output (same folder):
  _data/landmarks/subject_XXX/exercise_XXX/video_XXX/plots/pose.png
  _data/landmarks/subject_XXX/exercise_XXX/video_XXX/plots/right_hand.png
  _data/landmarks/subject_XXX/exercise_XXX/video_XXX/plots/left_hand.png
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import DATA_ROOT
from ask_inputs import ask_inputs

# Pose
POSE_GROUPS = {
    'Right side': ['nose', 'right_shoulder', 'right_elbow', 'right_wrist'],
    'Left side':  ['nose', 'left_shoulder',  'left_elbow',  'left_wrist'],
}

# Hand
HAND_JOINTS = ['wrist', 'index_mcp', 'pinky_mcp', 'thumb_tip', 'index_tip']

# Spatial dimensions plotted
DIMS = [('x', 'X (m)'), ('y', 'Y (m)'), ('z', 'Z (m)')]

# Visual style
_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
           '#8c564b', '#e377c2']
_STYLES = ['-', '--', '-.', ':', '-', '--', '-.']

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_csv(csv_path) -> pd.DataFrame:
    """Loads a CSV file; returns an empty DataFrame if not found."""
    if not csv_path.exists():
        print(f"[WARNING] File not found: {csv_path}")
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"[WARNING] No data in {csv_path}")
    return df

def _plot_group(axes_col: list, df: pd.DataFrame, joints: list, has_visibility: bool = False) -> None:
    """
    Draws one line per joint in each Axes of axes_col.

    Parameters:
        axes_col       : list of 3 Axes objects (one per dimension: X, Y, Z)
        df             : landmark DataFrame
        joints         : ordered list of joint name strings
        has_visibility : if True, uses the joint's visibility score to set alpha
    """
    for j_idx, joint in enumerate(joints):
        color  = _COLORS[j_idx % len(_COLORS)]
        lstyle = _STYLES[j_idx % len(_STYLES)]

        required = [f'{joint}_{d}' for d, _ in DIMS]
        if not all(c in df.columns for c in required):
            continue                    # column missing - skip silently

        sub = df.dropna(subset=required)
        if sub.empty:
            continue

        frames = sub['frame'].values

        # Alpha from mean visibility (pose only)
        alpha = 1.0
        if has_visibility:
            vis_col = f'{joint}_vis'
            if vis_col in sub.columns:
                alpha = float(np.clip(sub[vis_col].mean(), 0.25, 1.0))

        label = joint.replace('_', ' ')

        for ax_idx, (dim, _) in enumerate(DIMS):
            axes_col[ax_idx].plot(
                frames,
                sub[f'{joint}_{dim}'].values,
                color     = color,
                linestyle = lstyle,
                linewidth = 1.5,
                alpha     = alpha,
                label     = label,
            )

def _style_grid(axes, col_titles, dim_labels, n_rows, n_cols) -> None:
    """
    Applies consistent styling to a grid of Axes.
    """
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c] if n_cols > 1 else axes[r]
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
            ax.legend(fontsize=7, loc='upper right', framealpha=0.7)

            # Y-axis label - leftmost column only
            if c == 0:
                ax.set_ylabel(dim_labels[r], fontsize=9)

            # Column title - top row only
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10, fontweight='bold')

            # X-axis label - bottom row only
            if r == n_rows - 1:
                ax.set_xlabel('Frame', fontsize=9)

# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
def build_pose_figure(df_pose: pd.DataFrame, title: str) -> plt.Figure:
    """
    Pose figure - 3 rows (X, Y, Z) x 2 columns (Right side / Left side).
    Each subplot shows nose + shoulder + elbow + wrist for that body side.
    """
    groups    = list(POSE_GROUPS.items())       # [('Right side', [...]), ...]
    n_cols    = len(groups)
    n_rows    = len(DIMS)
    dim_lbls  = [lbl for _, lbl in DIMS]
    col_names = [name for name, _ in groups]

    fig, axes = plt.subplots(n_rows, n_cols, figsize  = (14, 9), sharex = True, sharey = "row")
    fig.suptitle(f"Pose - {title}", fontsize=12, fontweight='bold')

    for col_idx, (_, joints) in enumerate(groups):
        col_axes = [axes[r, col_idx] for r in range(n_rows)]
        _plot_group(col_axes, df_pose, joints, has_visibility=True)

    _style_grid(axes, col_names, dim_lbls, n_rows, n_cols)
    fig.tight_layout()
    return fig

def build_hand_figure(df_hand: pd.DataFrame, hand_side: str, title: str) -> plt.Figure:
    """
    Hand figure - 3 rows (X, Y, Z) x 1 column.
    All 5 landmarks are drawn as separate lines.
    """
    n_rows   = len(DIMS)
    
    fig, axes = plt.subplots(n_rows, 1, figsize = (10, 8), sharex = True, sharey = False)
    fig.suptitle(f"{hand_side} - {title}", fontsize=12, fontweight='bold')

    _plot_group(axes, df_hand, HAND_JOINTS, has_visibility=False)

    # Style manually (single column - axes is a 1-D array)
    for r_idx, (_, ylabel) in enumerate(DIMS):
        ax = axes[r_idx]
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
        ax.legend(fontsize=7, loc='upper right', framealpha=0.7)
        if r_idx == 0:
            ax.set_title(hand_side, fontsize=10, fontweight='bold')
        if r_idx == n_rows - 1:
            ax.set_xlabel('Frame', fontsize=9)

    fig.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Time Series Visualizer (2D) ===")
    subject_name, exercise_name, video_name = ask_inputs()
    landmarks_folder = DATA_ROOT / "landmarks" / subject_name / exercise_name / video_name
    title = f"{subject_name} / {exercise_name} / {video_name}"

    # Load data
    df_pose  = load_csv(landmarks_folder / 'pose_cleaned.csv')
    df_rhand = load_csv(landmarks_folder / 'right_hand_cleaned.csv')
    df_lhand = load_csv(landmarks_folder / 'left_hand_cleaned.csv')

    if df_pose.empty and df_rhand.empty and df_lhand.empty:
        print("No data found. Check subject, exercise and video numbers.")
        return

    # Build and save figures
    output_dir = landmarks_folder / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    if not df_pose.empty:
        fig = build_pose_figure(df_pose, title)
        path = output_dir / 'pose.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {path}")

    if not df_rhand.empty:
        fig = build_hand_figure(df_rhand, 'Right Hand', title)
        path = output_dir / 'right_hand.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {path}")

    if not df_lhand.empty:
        fig = build_hand_figure(df_lhand, 'Left Hand', title)
        path = output_dir / 'left_hand.png'
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {path}")

    print("Done.")


if __name__ == "__main__":
    main()