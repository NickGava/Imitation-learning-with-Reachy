'''
visualize_timeseries.py
=============================================================================
Visualizes the landmark time series recorded by pose_estimation.py.

For a given subject, exercise and video, produces three figures:
  - Pose        : 3D trajectory of 7 body joints (nose, shoulders, elbows, wrists)
  - Right Hand  : 3D trajectory of 5 right hand landmarks
  - Left Hand   : 3D trajectory of 5 left hand landmarks

Each joint is plotted as a 3D trajectory where:
  - line opacity is proportional to MediaPipe visibility score (pose only)
  - green dot = first frame, red dot = last frame

Figures are saved as PNG in:
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/plots/

Usage example:
  python visualize_timeseries.py
  >>> Subject number:  1
  >>> Exercise number: 1
  >>> Video number:    1
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Non viene mai chiamato direttamente, ma è necessario per il plotting 3D -- noqa: F401 serve per silenziare il warning
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.gridspec import GridSpec
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_ROOT

# Landmark names
POSE_JOINTS = [
    'nose',
    'left_shoulder', 'right_shoulder',
    'left_elbow',    'right_elbow',
    'left_wrist',    'right_wrist',
]

HAND_JOINTS = [
    'thumb_tip', 'index_tip',
    'wrist', 'index_mcp', 'pinky_mcp',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_csv(csv_path):
    """
    Loads all rows from a CSV file.
    Returns a DataFrame (empty if not found).
    """
    if not csv_path.exists():
        print(f"[WARNING] File not found: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    if df.empty:
        print(f"[WARNING] No data found in {csv_path}")

    return df


def plot_joint_3d(ax, df, joint_name, has_visibility=True):
    """
    Plots the 3D trajectory of a single joint.
    Line opacity is modulated by visibility (pose only).
    Incomplete rows are skipped.
    """
    x_col = f'{joint_name}_x'
    y_col = f'{joint_name}_y'
    z_col = f'{joint_name}_z'
    v_col = f'{joint_name}_vis' if has_visibility else None

    # Drop rows where coordinates are missing (dropna is a pandas method)
    required = [x_col, y_col, z_col]
    sub = df.dropna(subset=required)

    if sub.empty:
        ax.set_title(joint_name, fontsize=8)
        return

    x = sub[x_col].values
    y = sub[y_col].values
    z = sub[z_col].values

    if has_visibility and v_col in sub.columns:
        vis = sub[v_col].values
    else:
        vis = np.ones(len(x))

    # Plot segment by segment with alpha proportional to visibility
    for i in range(len(x) - 1):
        alpha = float(np.clip((vis[i] + vis[i + 1]) / 2, 0.05, 1.0))
        ax.plot(
            [x[i], x[i + 1]],
            [y[i], y[i + 1]],
            [z[i], z[i + 1]],
            color='steelblue',
            alpha=alpha,
            linewidth=1.2
        )

    # Start and end markers
    ax.scatter(*[x[0]], *[y[0]], *[z[0]], color='green', s=20, zorder=5, label='start')
    ax.scatter(*[x[-1]], *[y[-1]], *[z[-1]], color='red', s=20, zorder=5, label='end')

    ax.set_title(joint_name, fontsize=8)
    ax.set_xlabel('X', fontsize=6, labelpad=1)
    ax.set_ylabel('Y', fontsize=6, labelpad=1)
    ax.set_zlabel('Z', fontsize=6, labelpad=1)
    ax.tick_params(labelsize=5)


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
def build_pose_figure(df_pose, title):
    """
    Creates the pose figure with layout:
        [     nose     ]
        [ L_sh ][ R_sh ]
        [ L_el ][ R_el ]
        [ L_wr ][ R_wr ]
    """
    fig = plt.figure(figsize=(12, 14))
    fig.suptitle(f"Pose — {title}", fontsize=12, fontweight='bold')

    gs = GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

    layout = {
        'nose':           gs[0, :],          # spans both columns
        'left_shoulder':  gs[1, 0],
        'right_shoulder': gs[1, 1],
        'left_elbow':     gs[2, 0],
        'right_elbow':    gs[2, 1],
        'left_wrist':     gs[3, 0],
        'right_wrist':    gs[3, 1],
    }

    # Put graphs in the specified layout
    for joint, spec in layout.items():
        ax = fig.add_subplot(spec, projection='3d')
        plot_joint_3d(ax, df_pose, joint, has_visibility=True)

    return fig


def build_hand_figure(df_hand, hand_side, title):
    """
    Creates a hand figure with layout:
        [ thumb_tip ][ index_tip ]  (row 0, 2 plots centered)
        [ wrist ][ index_mcp ][ pinky_mcp ]  (row 1, 3 plots)
    """
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(f"{hand_side} — {title}", fontsize=12, fontweight='bold')

    # Use a 2x6 grid so we can center the 2 plots on row 0
    gs = GridSpec(2, 6, figure=fig, hspace=0.5, wspace=0.45)

    layout = {
        'thumb_tip':  gs[0, 0:3],
        'index_tip':  gs[0, 3:6],
        'wrist':      gs[1, 0:2],
        'index_mcp':  gs[1, 2:4],
        'pinky_mcp':  gs[1, 4:6],
    }

    # Put graphs in the specified layout
    for joint, spec in layout.items():
        ax = fig.add_subplot(spec, projection='3d')
        plot_joint_3d(ax, df_hand, joint, has_visibility=False)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # --- User input ---
    print("=== Time Series Visualizer ===")
    try:
        subject_num  = int(input("Subject number:  ").strip())
        exercise_num = int(input("Exercise number: ").strip())
        video_num    = int(input("Video number:    ").strip())
    except ValueError:
        print("Error: all values must be integers.")
        return

    subject_name  = f"subject_{subject_num:03d}"
    exercise_name = f"exercise_{exercise_num:03d}"
    video_name    = f"video_{video_num:03d}"

    landmarks_folder = DATA_ROOT / "landmarks" / subject_name / exercise_name / video_name

    title = f"{subject_name} / {exercise_name} / {video_name}"

    # --- Load data ---
    df_pose  = load_csv(landmarks_folder / 'pose.csv')
    df_rhand = load_csv(landmarks_folder / 'right_hand.csv')
    df_lhand = load_csv(landmarks_folder / 'left_hand.csv')

    if df_pose.empty and df_rhand.empty and df_lhand.empty:
        print("No data found. Check subject, exercise and video numbers.")
        return

    # --- Build and save figures ---
    output_dir = landmarks_folder / 'plots'
    output_dir.mkdir(parents=True, exist_ok=True)

    if not df_pose.empty:
        fig_pose = build_pose_figure(df_pose, title)
        path_pose = output_dir / 'pose.png'
        fig_pose.savefig(path_pose, dpi=150, bbox_inches='tight')
        print(f"Saved: {path_pose}")

    if not df_rhand.empty:
        fig_rhand = build_hand_figure(df_rhand, 'Right Hand', title)
        path_rhand = output_dir / 'right_hand.png'
        fig_rhand.savefig(path_rhand, dpi=150, bbox_inches='tight')
        print(f"Saved: {path_rhand}")

    if not df_lhand.empty:
        fig_lhand = build_hand_figure(df_lhand, 'Left Hand', title)
        path_lhand = output_dir / 'left_hand.png'
        fig_lhand.savefig(path_lhand, dpi=150, bbox_inches='tight')
        print(f"Saved: {path_lhand}")


if __name__ == "__main__":
    main()