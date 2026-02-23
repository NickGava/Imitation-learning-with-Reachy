import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ----------------------------
# CONFIG
# ----------------------------
CSV_PATH = Path(__file__).parent / "reachy_motion_dataset.csv"
OUTPUT_PATH = Path(__file__).parent / "motion_visualization.png"

JOINTS = ["shoulder", "elbow", "wrist"]
SIDES = ["l", "r"]
SIDE_LABELS = {"l": "Left", "r": "Right"}
COORDS = ["x", "y", "z"]
COORD_COLORS = {"x": "#e74c3c", "y": "#2ecc71", "z": "#3498db"}

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(CSV_PATH)
t = df["timestamp"]

# ----------------------------
# PLOT
# ----------------------------
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Reachy — Human Motion Dataset: Joint Trajectories Over Time",
             fontsize=15, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

for row_idx, joint in enumerate(JOINTS):
    for col_idx, side in enumerate(SIDES):
        ax = fig.add_subplot(gs[row_idx, col_idx])

        for coord in COORDS:
            col_name = f"{side}_{joint}_{coord}"
            ax.plot(t, df[col_name],
                    color=COORD_COLORS[coord],
                    label=coord.upper(),
                    linewidth=1.6,
                    alpha=0.85)

        ax.set_title(f"{SIDE_LABELS[side]} {joint.capitalize()}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Position (m)", fontsize=9)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=8)

# Legenda coordinata condivisa in basso
handles = [plt.Line2D([0], [0], color=COORD_COLORS[c], linewidth=2, label=c.upper())
           for c in COORDS]
fig.legend(handles=handles, loc="lower center", ncol=3,
           fontsize=10, framealpha=0.8, title="Coordinate", title_fontsize=10,
           bbox_to_anchor=(0.5, 0.01))

plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
print(f"Grafico salvato in: {OUTPUT_PATH}")