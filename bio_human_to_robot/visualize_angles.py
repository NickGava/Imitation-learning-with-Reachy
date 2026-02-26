"""
visualize_angles.py
--------------------
Visualizes the biomechanical joint angle time series produced by
get_landmark_live_bio.py.

Two figure types:
    1. Angle trajectories — time series of all 10 angles for each arm,
       with anatomical range-of-motion limits shown as shaded bands.
    2. Phase plots — shoulder flexion vs. elbow flexion for each arm,
       useful for visualizing synergies between joints.

Usage:
    python visualize_angles.py                    # uses default CSV path
    python visualize_angles.py my_recording.csv   # custom path

Requirements:
    - pandas, matplotlib installed
    - A CSV produced by get_landmark_live_bio.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from biomechanics import JOINT_LIMITS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_CSV = Path(__file__).parent / "human_motion_bio.csv"
OUTPUT_PNG  = Path(__file__).parent / "angles_visualization.png"

ANGLE_COLORS = {
    "flexion":      "#e74c3c",   # red
    "abduction":    "#3498db",   # blue
    "internal_rot": "#2ecc71",   # green
    "elbow":        "#f39c12",   # orange
    "wrist":        "#9b59b6",   # purple
}

ANGLE_META = {
    # column_suffix        : (label,                    color key,      limit key)
    "shoulder_flexion"      : ("Shoulder Flexion",      "flexion",      "shoulder_flexion"),
    "shoulder_abduction"    : ("Shoulder Abduction",    "abduction",    "shoulder_abduction"),
    "shoulder_internal_rot" : ("Shoulder Int. Rot.",    "internal_rot", "shoulder_internal_rot"),
    "elbow_flexion"         : ("Elbow Flexion",         "elbow",        "elbow_flexion"),
    "wrist_flexion"         : ("Wrist Flexion",         "wrist",        "wrist_flexion"),
}


def plot_angle_trajectories(df: pd.DataFrame, out_path: Path) -> None:
    """
    Plots all 10 anatomical joint angles over time, one subplot per angle,
    split into left arm (left column) and right arm (right column).

    Anatomical ROM limits are shown as horizontal dashed lines.

    Args:
        df:       DataFrame with timestamp and angle columns.
        out_path: Path where the PNG is saved.
    """
    t = df["timestamp"] - df["timestamp"].iloc[0]   # time from 0

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        "Biomechanical Joint Angles — Upper Body (Reachy Imitation)",
        fontsize=14, fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(len(ANGLE_META), 2, figure=fig, hspace=0.6, wspace=0.3)

    for row_idx, (suffix, (label, color_key, limit_key)) in enumerate(ANGLE_META.items()):
        color = ANGLE_COLORS[color_key]
        lo, hi = JOINT_LIMITS[limit_key]

        for col_idx, side in enumerate(["l", "r"]):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            col = f"{side}_{suffix}"

            ax.plot(t, df[col], color=color, linewidth=1.6, alpha=0.9, label=label)

            # Anatomical ROM limits
            ax.axhline(lo, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.axhline(hi, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.axhspan(lo, hi, alpha=0.04, color=color)
            ax.axhline(0,  color="black", linewidth=0.5, linestyle=":", alpha=0.4)

            side_label = "Left" if side == "l" else "Right"
            ax.set_title(f"{side_label} — {label}", fontsize=9, fontweight="bold")
            ax.set_xlabel("Time (s)", fontsize=8)
            ax.set_ylabel("Angle (°)", fontsize=8)
            ax.set_ylim(lo - 10, hi + 10)
            ax.tick_params(labelsize=7)
            ax.grid(True, linestyle="--", alpha=0.3)

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Angle trajectories saved → {out_path}")


def plot_phase_portraits(df: pd.DataFrame, out_path: Path) -> None:
    """
    Plots shoulder flexion vs. elbow flexion phase portraits for each arm.

    Phase portraits reveal motor synergies — characteristic coupling patterns
    between joints that persist across demonstrations. These are useful for
    evaluating reproduction quality and for learning movement primitives.

    Args:
        df:       DataFrame with angle columns.
        out_path: Path where the PNG is saved (overwrites _trajectories suffix).
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Phase Portrait: Shoulder Flexion vs. Elbow Flexion",
        fontsize=13, fontweight="bold",
    )

    for ax, side, side_label, color in zip(
        axes,
        ["l", "r"],
        ["Left Arm", "Right Arm"],
        ["#3498db", "#e74c3c"],
    ):
        sh = df[f"{side}_shoulder_flexion"].values
        el = df[f"{side}_elbow_flexion"].values

        # Color-encode time progression
        n = len(sh)
        for i in range(n - 1):
            alpha = 0.3 + 0.7 * i / n
            ax.plot(sh[i:i+2], el[i:i+2], color=color, alpha=alpha, linewidth=1.2)

        ax.scatter(sh[0],  el[0],  color="green", zorder=5, s=60, label="Start")
        ax.scatter(sh[-1], el[-1], color="black", zorder=5, s=60, label="End")

        ax.set_title(side_label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Shoulder Flexion (°)", fontsize=10)
        ax.set_ylabel("Elbow Flexion (°)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.3)

    phase_path = out_path.parent / (out_path.stem + "_phase.png")
    plt.savefig(phase_path, dpi=150, bbox_inches="tight")
    print(f"Phase portraits saved   → {phase_path}")


def print_summary(df: pd.DataFrame) -> None:
    """Prints a quick statistical summary of the recorded angles."""
    angle_cols = [c for c in df.columns if c != "timestamp"]
    duration   = df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]
    fps        = len(df) / duration if duration > 0 else 0

    print(f"\n{'='*60}")
    print(f"  Recording summary")
    print(f"  Frames   : {len(df)}")
    print(f"  Duration : {duration:.2f} s  ({fps:.1f} fps)")
    print(f"{'='*60}")
    print(f"{'Angle':<30}  {'Mean':>7}  {'Std':>6}  {'Min':>7}  {'Max':>7}")
    print(f"{'-'*60}")
    for col in angle_cols:
        s = df[col]
        print(f"{col:<30}  {s.mean():>+7.1f}  {s.std():>6.1f}  {s.min():>+7.1f}  {s.max():>+7.1f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} frames from: {csv_path}")

    print_summary(df)
    plot_angle_trajectories(df, OUTPUT_PNG)
    plot_phase_portraits(df, OUTPUT_PNG)

    plt.show()
