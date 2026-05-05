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
import matplotlib.ticker as ticker

from config import DATA_ROOT
from ask_inputs import ask_inputs

Y_PADDING = 0.02    # metres of padding above/below the shared Y range

# Pose
POSE_GROUPS = {
    'Right side': ['right_shoulder', 'right_elbow', 'right_wrist'],
    'Left side':  ['left_shoulder',  'left_elbow',  'left_wrist'],
}

# Spatial dimensions plotted
DIMS = [('x', 'X - Left/Right (m)'), ('y', 'Y - Up/Down (m)'), ('z', 'Z - Forward/Backward (m)')]

# Visual style
_COLORS = [ '#2ca02c', '#e67e22', '#3498db']
_STYLES = ['-.', '--', '-', ':', '-', '--', '-.']

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

def _shared_ylim(df: pd.DataFrame, joints: list) -> tuple:
    """Computes shared Y limits across all joints and dimensions."""
    cols = []
    for joint in joints:
        for dim, _ in DIMS:
            col = f'{joint}_{dim}'
            if col in df.columns:
                cols.append(col)
    if not cols:
        return -1.0, 1.0
    values = df[cols].values.flatten()
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return -1.0, 1.0
    return float(values.min()) - Y_PADDING, float(values.max()) + Y_PADDING


def _style_grid(axes, col_titles, dim_labels, n_rows, n_cols, y_min, y_max) -> None:
    """
    Applies consistent styling to a grid of Axes:
    - shared Y scale across all subplots
    - tick labels on every subplot
    - Y label on left column only
    - column title on top row only
    - X label on bottom row only
    """
    for r in range(n_rows):
        for c in range(n_cols):
            ax = axes[r, c] if n_cols > 1 else axes[r]

            # __ Shared Y scale __
            ax.set_ylim(y_min, y_max)
            ax.yaxis.set_major_locator(ticker.AutoLocator())
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

            # __ Tick labels everywhere __
            ax.tick_params(axis='both', labelsize=8)

            ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.7)
            ax.grid(True, which='minor', linestyle=':',  linewidth=0.4, alpha=0.4)
            ax.legend(fontsize=7, loc='upper right', framealpha=0.7)

            # __ Y label: left column only __
            if c == 0:
                ax.set_ylabel(dim_labels[r], fontsize=9)

            # __ Column title: top row only __
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10, fontweight='bold')

            # __ X label: bottom row only __
            if r == n_rows - 1:
                ax.set_xlabel('Frame', fontsize=9)

# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
def build_pose_figure(df_pose: pd.DataFrame, title: str) -> plt.Figure:
    """
    Pose figure - 3 rows (X, Y, Z) x 2 columns (Right side / Left side).
    Each subplot shows shoulder + elbow + wrist for that body side.
    """
    groups    = list(POSE_GROUPS.items())
    n_cols    = len(groups)
    n_rows    = len(DIMS)
    dim_lbls  = [lbl for _, lbl in DIMS]
    col_names = [name for name, _ in groups]

    # __ Shared Y limits across all joints and dimensions __
    all_joints = [j for _, joints in groups for j in joints]
    y_min, y_max = _shared_ylim(df_pose, all_joints)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 9), sharex=False, sharey=False)
    fig.suptitle(f'Pose – {title}', fontsize=12, fontweight='bold')

    for col_idx, (_, joints) in enumerate(groups):
        col_axes = [axes[r, col_idx] for r in range(n_rows)]
        _plot_group(col_axes, df_pose, joints, has_visibility=True)

    _style_grid(axes, col_names, dim_lbls, n_rows, n_cols, y_min, y_max)
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

    if df_pose.empty:
        print("No data found. Check subject, exercise and video numbers.")
        return

    # Build and save figures
    output_dir = landmarks_folder / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    if not df_pose.empty:
        fig = build_pose_figure(df_pose, title)
        path = output_dir / 'pose.png'
        # plt.show()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {path}")


    print("Done.")


if __name__ == "__main__":
    main()