'''
mapping.py
=============================================================================
Maps cleaned pose landmarks and hand features to Reachy arm space.

For each arm (right and left), reads:
  - pose_cleaned.csv         → shoulder, elbow, wrist positions (world space)
  - {side}_hand_mapped.csv   → gripper_angle, hand orientation quaternion

Pipeline per arm:
  1. Build Reachy base frame directly from pose landmarks
       Origin  : midpoint between left and right shoulder
       X-axis  : left shoulder → right shoulder  (lateral, matches Reachy)
       Y-axis  : midpoint hips → midpoint shoulders  (vertical, up)
       Z-axis  : X × Y  (forward, right-hand rule)

  2. Express shoulder, elbow, wrist in torso frame

  3. Scale arm segments to Reachy proportions
       Upper arm (shoulder → elbow) : REACHY_UPPER_ARM = 0.280 m
       Forearm   (elbow   → wrist)  : REACHY_FOREARM   = 0.250 m

  4. Merge with hand features on frame index

  5. Merge right and left arm DataFrames into a single output file

Output (same folder as input):
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/arms_mapped.csv

Each output row:
  frame, timestamp,
  r_sh_x,    r_sh_y,    r_sh_z,
  r_elbow_x, r_elbow_y, r_elbow_z,
  r_wrist_x, r_wrist_y, r_wrist_z,
  r_gripper_angle,
  r_q_w, r_q_x, r_q_y, r_q_z,
  l_sh_x,    l_sh_y,    l_sh_z,
  l_elbow_x, l_elbow_y, l_elbow_z,
  l_wrist_x, l_wrist_y, l_wrist_z,
  l_gripper_angle,
  l_q_w, l_q_x, l_q_y, l_q_z

Notes:
  - All positions are in meters, expressed in the torso frame.
  - The shoulder columns are kept for debug / validation purposes.
  - Pose frames with no corresponding hand detection are kept and filled
    with neutral values: gripper fully open, identity quaternion.
  - The quaternion from hand_features describes hand orientation in
    camera frame; it is NOT re-expressed in torso frame here — that
    conversion happens in run_ik.py when building the 4×4 target pose.
  - Frames present in only one arm (e.g. one hand_mapped.csv is missing)
    are kept; the missing arm columns are filled with NaN.
'''

import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_ROOT

# Reachy arm lengths (meters)
REACHY_UPPER_ARM = 0.280    # shoulder → elbow
REACHY_FOREARM   = 0.250    # elbow   → wrist

# Reachy gripper motor ranges (degrees)
GRIPPER_RANGE = {
    'right_hand': {'open': -69.0, 'closed':  20.0},
    'left_hand':  {'open':  69.0, 'closed': -20.0},
}

# Per-arm column names (without side prefix)
_ARM_COLS = [
    'sh_x',    'sh_y',    'sh_z',
    'elbow_x', 'elbow_y', 'elbow_z',
    'wrist_x', 'wrist_y', 'wrist_z',
    'gripper_angle',
    'q_w', 'q_x', 'q_y', 'q_z',
]

# Final combined output header
ARMS_MAPPED_HEADER = (
    ['frame', 'timestamp']
    + [f'r_{c}' for c in _ARM_COLS]
    + [f'l_{c}' for c in _ARM_COLS]
)

# Pose landmark column prefixes needed per side
POSE_COLS = {
    'right': {'shoulder': 'right_shoulder', 'elbow': 'right_elbow', 'wrist': 'right_wrist'},
    'left':  {'shoulder': 'left_shoulder',  'elbow': 'left_elbow',  'wrist': 'left_wrist'},
}

# Shared landmarks needed to build the torso frame
TORSO_COLS = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)

def _quat_to_R(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])

def _R_to_quat(R: np.ndarray) -> np.ndarray:
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w, x = 0.25/s, (R[2,1]-R[1,2])*s
        y, z = (R[0,2]-R[2,0])*s, (R[1,0]-R[0,1])*s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w, x = (R[2,1]-R[1,2])/s, 0.25*s
        y, z = (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w, x = (R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s
        y, z = 0.25*s, (R[1,2]+R[2,1])/s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w, x = (R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s
        y, z = (R[1,2]+R[2,1])/s, 0.25*s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)

def _xyz(row: pd.Series, prefix: str) -> np.ndarray:
    """Extract [x, y, z] from a DataFrame row given a column prefix."""
    return np.array([row[f'{prefix}_x'], row[f'{prefix}_y'], row[f'{prefix}_z']])

def _build_torso_rotation_matrix(
    l_sh: np.ndarray, r_sh: np.ndarray,
    l_hip: np.ndarray, r_hip: np.ndarray
) -> np.ndarray:
    """
    Builds a 3×3 rotation matrix R such that v_reachy = R.T @ v_world.

    Reachy base frame axes (expressed in world frame):
      Y : normalize(l_sh - r_sh)           lateral (right → left shoulder)
      Z : normalize(mid_sh - mid_hip)      vertical (up)
      X : Y × Z                            toward camera (right-hand rule)
    """
    mid_sh  = (l_sh + r_sh)   * 0.5
    mid_hip = (l_hip + r_hip) * 0.5

    y = _normalize(l_sh - r_sh)
    z = _normalize(mid_sh - mid_hip)
    z = _normalize(z - np.dot(z, y) * y)   # re-orthogonalize Z vs Y
    x = np.cross(y, z)                     # X = Y × Z → toward camera

    return np.column_stack([x, y, z])      # columns = Reachy axes in world frame


# ---------------------------------------------------------------------------
# Per-arm mapping  (returns DataFrame with un-prefixed columns)
# ---------------------------------------------------------------------------
def _map_arm(df_pose: pd.DataFrame, df_hand: pd.DataFrame, side: str) -> pd.DataFrame:
    """
    Processes one arm and returns a DataFrame with columns:
      frame, timestamp, sh_x/y/z, elbow_x/y/z, wrist_x/y/z,
      gripper_angle, q_w, q_x, q_y, q_z
    """
    cols = POSE_COLS[side]
    rows      = []
    R_frames  = []

    for _, row in df_pose.iterrows():
        # --- Torso frame ---
        l_sh  = _xyz(row, 'left_shoulder')
        r_sh  = _xyz(row, 'right_shoulder')
        l_hip = _xyz(row, 'left_hip')
        r_hip = _xyz(row, 'right_hip')

        R      = _build_torso_rotation_matrix(l_sh, r_sh, l_hip, r_hip)
        origin = (l_sh + r_sh) * 0.5           # torso frame origin (mid-shoulder)

        # --- Landmark positions in world frame → torso frame ---
        sh_t    = R.T @ (_xyz(row, cols['shoulder']) - origin)
        elbow_t = R.T @ (_xyz(row, cols['elbow'])    - origin)
        wrist_t = R.T @ (_xyz(row, cols['wrist'])    - origin)

        # --- Scale segments (preserve direction, match Reachy lengths) ---
        v_upper_scaled = _normalize(elbow_t - sh_t)    * REACHY_UPPER_ARM
        v_lower_scaled = _normalize(wrist_t - elbow_t) * REACHY_FOREARM

        sh_out    = sh_t
        elbow_out = sh_out    + v_upper_scaled
        wrist_out = elbow_out + v_lower_scaled

        rows.append({
            'frame':     int(row['frame']),
            'timestamp': row['timestamp'],
            'sh_x':    sh_out[0],    'sh_y':    sh_out[1],    'sh_z':    sh_out[2],
            'elbow_x': elbow_out[0], 'elbow_y': elbow_out[1], 'elbow_z': elbow_out[2],
            'wrist_x': wrist_out[0], 'wrist_y': wrist_out[1], 'wrist_z': wrist_out[2],
        })
        R_frames.append(R)

    df_mapped = pd.DataFrame(rows)

    # --- Merge hand features ---
    df_merged = pd.merge(
        df_mapped,
        df_hand[['frame', 'gripper_angle', 'q_w', 'q_x', 'q_y', 'q_z']],
        on='frame', how='left'
    )

    neutral_gripper = GRIPPER_RANGE[f'{side}_hand']['open']
    df_merged['gripper_angle'] = df_merged['gripper_angle'].fillna(neutral_gripper)
    df_merged['q_w'] = df_merged['q_w'].fillna(1.0)
    df_merged['q_x'] = df_merged['q_x'].fillna(0.0)
    df_merged['q_y'] = df_merged['q_y'].fillna(0.0)
    df_merged['q_z'] = df_merged['q_z'].fillna(0.0)

    # --- Rotate quaternion: MediaPipe world frame → Reachy base frame ---
    q_out = np.zeros((len(df_merged), 4))
    for i, (_, row) in enumerate(df_merged.iterrows()):
        q_world  = np.array([row['q_w'], row['q_x'], row['q_y'], row['q_z']])
        R_reachy = R_frames[i].T @ _quat_to_R(q_world)
        q_out[i] = _R_to_quat(R_reachy)

    df_merged[['q_w', 'q_x', 'q_y', 'q_z']] = q_out

    return df_merged[['frame', 'timestamp'] + _ARM_COLS]


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

    # --- Load and validate pose ---
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
    df_pose  = df_pose.dropna(subset=pose_required).reset_index(drop=True)
    print(f"Pose: {len(df_pose)} valid frames (dropped {n_before - len(df_pose)})\n")

    # --- Process both arms ---
    arm_dfs = {}
    for side, hand_side in [('right', 'right_hand'), ('left', 'left_hand')]:
        print(f"=== {side} arm ===")

        hand_path = folder / f"{hand_side}_mapped.csv"
        if not hand_path.exists():
            print(f"  [SKIP] {hand_side}_mapped.csv not found — arm will be NaN.\n")
            arm_dfs[side] = None
            continue

        df_hand = pd.read_csv(hand_path)
        df_arm  = _map_arm(df_pose, df_hand, side)
        print(f"  rows mapped: {len(df_arm)}")
        arm_dfs[side] = df_arm

    # --- Merge into a single DataFrame ---
    # Base: frame + timestamp from pose
    df_base = df_pose[['frame', 'timestamp']].copy()

    for side, prefix in [('right', 'r'), ('left', 'l')]:
        if arm_dfs[side] is not None:
            df_side = arm_dfs[side].drop(columns=['timestamp'])
            df_side = df_side.rename(columns={c: f'{prefix}_{c}' for c in _ARM_COLS})
            df_base = pd.merge(df_base, df_side, on='frame', how='left')
        else:
            # Fill with NaN columns so the header is always complete
            for c in _ARM_COLS:
                df_base[f'{prefix}_{c}'] = np.nan

    df_out = df_base[ARMS_MAPPED_HEADER]

    output_path = folder / "arms_mapped.csv"
    df_out.to_csv(output_path, index=False)
    print(f"\nCombined output: {len(df_out)} rows → {output_path}")
    print("Done.")


if __name__ == "__main__":
    main()