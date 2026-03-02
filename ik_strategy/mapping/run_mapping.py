'''
run_mapping.py
=============================================================================
Maps cleaned pose landmarks and hand features to Reachy arm space.

For each arm (right and left), reads:
  - pose_cleaned.csv         → shoulder, elbow, wrist positions (world space)
  - {side}_hand_mapped.csv → gripper_angle, hand orientation quaternion

Pipeline per arm:
  1. Build torso frame
       Origin  : midpoint between left and right shoulder
       X-axis  : right shoulder → left shoulder  (lateral)
       Y-axis  : midpoint shoulders → midpoint hips  (vertical, pointing up)
       Z-axis  : X × Y  (forward, out of chest)

  2. Express shoulder, elbow, wrist in torso frame

  3. Scale arm segments to Reachy proportions
       Upper arm (shoulder → elbow) : REACHY_UPPER_ARM = 0.280 m
       Forearm   (elbow   → wrist)  : REACHY_FOREARM   = 0.250 m
       Scale factors are computed per video as the median segment length
       over all frames, then applied to each vector independently.
       This preserves direction while matching Reachy's morphology.

  4. Merge with hand features on frame index

Output (same folder as input):
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/right_arm_mapped.csv
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/left_arm_mapped.csv

Each output row:
  frame, timestamp,
  sh_x,    sh_y,    sh_z,
  elbow_x, elbow_y, elbow_z,
  wrist_x, wrist_y, wrist_z,
  gripper_angle,
  q_w, q_x, q_y, q_z

Notes:
  - All positions are in meters, expressed in the torso frame.
  - The shoulder column is kept for debug / validation purposes.
  - Rows where pose or hand features are missing are dropped.
  - The quaternion from hand_features describes hand orientation in
    camera frame; it is NOT re-expressed in torso frame here — that
    conversion happens in run_ik.py when building the 4×4 target pose.
'''

import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_ROOT

# ---------------------------------------------------------------------------
# Reachy arm segment lengths (meters)
# ---------------------------------------------------------------------------
REACHY_UPPER_ARM = 0.280    # shoulder → elbow
REACHY_FOREARM   = 0.250    # elbow   → wrist

# Output columns
ARM_MAPPED_HEADER = [
    'frame', 'timestamp',
    'sh_x',    'sh_y',    'sh_z',
    'elbow_x', 'elbow_y', 'elbow_z',
    'wrist_x', 'wrist_y', 'wrist_z',
    'gripper_angle',
    'q_w', 'q_x', 'q_y', 'q_z',
]

# Pose landmark column prefixes needed per side
# Keys match column names in pose_cleaned.csv
POSE_COLS = {
    'right': {
        'shoulder': 'right_shoulder',
        'elbow':    'right_elbow',
        'wrist':    'right_wrist',
    },
    'left': {
        'shoulder': 'left_shoulder',
        'elbow':    'left_elbow',
        'wrist':    'left_wrist',
    },
}

# Shared landmarks needed to build the torso frame
TORSO_COLS = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)


def _xyz(row: pd.Series, prefix: str) -> np.ndarray:
    """Extract [x, y, z] from a DataFrame row given a column prefix."""
    return np.array([row[f'{prefix}_x'], row[f'{prefix}_y'], row[f'{prefix}_z']])


def _build_torso_rotation(l_sh: np.ndarray, r_sh: np.ndarray,
                          l_hip: np.ndarray, r_hip: np.ndarray) -> np.ndarray:
    """
    Builds a 3×3 rotation matrix R_torso that transforms world-space vectors
    into the torso frame.

    Torso frame axes:
      X : normalize(r_sh - l_sh)          lateral (left → right)
      Y : normalize(mid_sh - mid_hip)     vertical (up)
      Z : X × Y                           forward (out of chest)

    Returns R (3×3) such that v_torso = R.T @ v_world
    (columns of R are the torso axes expressed in world frame)
    """
    mid_sh  = (l_sh + r_sh)   * 0.5
    mid_hip = (l_hip + r_hip) * 0.5

    x = _normalize(r_sh - l_sh)
    y = _normalize(mid_sh - mid_hip)
    # Re-orthogonalize: y might not be perfectly perpendicular to x
    y = _normalize(y - np.dot(y, x) * x)
    z = np.cross(x, y)

    # R columns = torso axes in world frame → R.T maps world → torso
    R = np.column_stack([x, y, z])
    return R


# ---------------------------------------------------------------------------
# Per-arm mapping
# ---------------------------------------------------------------------------
def _compute_scale_factors(df_pose: pd.DataFrame, side: str) -> tuple[float, float]:
    """
    Estimates per-video scale factors by computing the median segment length
    (shoulder→elbow and elbow→wrist) over all valid frames.

    Returns:
        (scale_upper, scale_lower) where
          scale_upper = REACHY_UPPER_ARM / median_human_upper_arm
          scale_lower = REACHY_FOREARM   / median_human_forearm
    """
    cols = POSE_COLS[side]

    upper_lengths = []
    lower_lengths = []

    for _, row in df_pose.iterrows():
        sh    = _xyz(row, cols['shoulder'])
        elbow = _xyz(row, cols['elbow'])
        wrist = _xyz(row, cols['wrist'])
        upper_lengths.append(np.linalg.norm(elbow - sh))
        lower_lengths.append(np.linalg.norm(wrist - elbow))

    median_upper = np.median(upper_lengths)
    median_lower = np.median(lower_lengths)

    if median_upper < 1e-9 or median_lower < 1e-9:
        raise ValueError(f"[{side}] Degenerate segment length — check pose data.")

    scale_upper = REACHY_UPPER_ARM / median_upper
    scale_lower = REACHY_FOREARM   / median_lower

    print(f"  human upper arm  (median): {median_upper*100:.1f} cm  →  scale {scale_upper:.3f}")
    print(f"  human forearm    (median): {median_lower*100:.1f} cm  →  scale {scale_lower:.3f}")

    return scale_upper, scale_lower


def _map_arm(df_pose: pd.DataFrame, df_hand: pd.DataFrame,
             side: str, scale_upper: float, scale_lower: float) -> pd.DataFrame:
    """
    For each frame:
      1. Build torso frame from shoulder/hip landmarks
      2. Express sh, elbow, wrist in torso frame (relative to shoulder origin)
      3. Scale upper arm and forearm vectors independently
      4. Recompose absolute positions (sh at origin, then elbow, then wrist)

    Returns a DataFrame with ARM_MAPPED_HEADER columns.
    """
    cols = POSE_COLS[side]
    rows = []

    for _, row in df_pose.iterrows():
        # --- Torso frame ---
        l_sh  = _xyz(row, 'left_shoulder')
        r_sh  = _xyz(row, 'right_shoulder')
        l_hip = _xyz(row, 'left_hip')
        r_hip = _xyz(row, 'right_hip')

        R = _build_torso_rotation(l_sh, r_sh, l_hip, r_hip)
        origin = (l_sh + r_sh) * 0.5       # torso frame origin (mid-shoulder)

        # --- Landmark positions in world frame ---
        sh_w    = _xyz(row, cols['shoulder'])
        elbow_w = _xyz(row, cols['elbow'])
        wrist_w = _xyz(row, cols['wrist'])

        # --- Express in torso frame (relative to mid-shoulder origin) ---
        sh_t    = R.T @ (sh_w    - origin)
        elbow_t = R.T @ (elbow_w - origin)
        wrist_t = R.T @ (wrist_w - origin)

        # --- Scale segments (direction preserved, length matched to Reachy) ---
        v_upper = elbow_t - sh_t                            # shoulder → elbow vector
        v_lower = wrist_t - elbow_t                         # elbow → wrist vector

        v_upper_scaled = _normalize(v_upper) * scale_upper * np.linalg.norm(v_upper)
        v_lower_scaled = _normalize(v_lower) * scale_lower * np.linalg.norm(v_lower)

        sh_out    = sh_t                                    # shoulder stays as-is
        elbow_out = sh_out + v_upper_scaled
        wrist_out = elbow_out + v_lower_scaled

        rows.append({
            'frame':     int(row['frame']),
            'timestamp': row['timestamp'],
            'sh_x': sh_out[0],    'sh_y': sh_out[1],    'sh_z': sh_out[2],
            'elbow_x': elbow_out[0], 'elbow_y': elbow_out[1], 'elbow_z': elbow_out[2],
            'wrist_x': wrist_out[0], 'wrist_y': wrist_out[1], 'wrist_z': wrist_out[2],
        })

    df_mapped = pd.DataFrame(rows)

    # --- Merge hand features ---
    df_merged = pd.merge(df_mapped, df_hand[['frame', 'gripper_angle',
                                              'q_w', 'q_x', 'q_y', 'q_z']],
                         on='frame', how='inner')

    return df_merged[ARM_MAPPED_HEADER]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
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

    folder = DATA_ROOT / "landmarks" / subject_name / exercise_name / video_name

    if not folder.is_dir():
        print(f"Error: folder not found → {folder}")
        return

    # --- Load pose (shared by both arms) ---
    pose_path = folder / "pose_cleaned.csv"
    if not pose_path.exists():
        print(f"Error: pose_cleaned.csv not found → {pose_path}")
        return

    pose_required = []
    for s in ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
              'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'):
        pose_required += [f'{s}_x', f'{s}_y', f'{s}_z']

    df_pose = pd.read_csv(pose_path)
    n_before = len(df_pose)

    df_pose = df_pose.dropna(subset=pose_required).reset_index(drop=True)
    print(f"Pose: {len(df_pose)} valid frames (dropped {n_before - len(df_pose)})\n")

    # --- Process each arm ---
    for side, hand_side in [('right', 'right_hand'), ('left', 'left_hand')]:
        print(f"=== {side} arm ===")

        hand_path = folder / f"{hand_side}_mapped.csv"
        if not hand_path.exists():
            print(f"[SKIP] {hand_side}_mapped.csv not found.\n")
            continue

        df_hand = pd.read_csv(hand_path)

        # Scale factors from median segment lengths
        scale_upper, scale_lower = _compute_scale_factors(df_pose, side)

        # Map
        df_out = _map_arm(df_pose, df_hand, side, scale_upper, scale_lower)

        n_dropped = len(df_pose) - len(df_out)
        if n_dropped:
            print(f"  frames dropped after hand merge: {n_dropped}")

        output_path = folder / f"{side}_arm_mapped.csv"
        df_out.to_csv(output_path, index=False)
        print(f"  rows saved: {len(df_out)}")
        print(f"  → {output_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()