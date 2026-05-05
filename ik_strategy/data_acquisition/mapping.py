'''
mapping.py
=============================================================================
Maps cleaned pose landmarks to Reachy arm space.

For each arm (right and left), reads pose_cleaned.csv and extracts
shoulder, elbow, wrist positions (world space).

Pipeline per arm:
  1. Build Reachy base frame from pose landmarks
       Origin: midpoint between left and right shoulder
       Y-axis: right shoulder -> left shoulder  (lateral)
       Z-axis: midpoint hips -> midpoint shoulders  (vertical, up)
       X-axis: Y x Z  (forward, toward camera, right-hand rule)

  2. Express shoulder, elbow, wrist in torso frame

  3. Scale arm segments to Reachy proportions
       Upper arm (shoulder -> elbow): REACHY_UPPER_ARM = 0.280 m
       Forearm   (elbow   -> wrist) : REACHY_FOREARM   = 0.250 m

  4. Gripper fixed at 0 (hand landmarks not tracked)

Head: fixed neutral gaze point from HEAD_NEUTRAL config (no face tracking).

Output (same folder as input):
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/poses_mapped.csv
'''

import numpy as np
import pandas as pd

from config import DATA_ROOT, HEAD_NEUTRAL
from ask_inputs import ask_inputs

# Reachy arm lengths (meters)
REACHY_UPPER_ARM = 0.280    # shoulder -> elbow
REACHY_FOREARM   = 0.250    # elbow   -> wrist

# Per-arm column names
_ARM_COLS = [
    'sh_x',    'sh_y',    'sh_z',
    'elbow_x', 'elbow_y', 'elbow_z',
    'wrist_x', 'wrist_y', 'wrist_z',
    'gripper_angle',
    # hand orientation quaternion removed (gripper fixed at 0, orientation unused)
]

# Final combined output header
MAPPED_HEADER = (
    ['frame', 'timestamp']
    + [f'r_{c}' for c in _ARM_COLS]
    + [f'l_{c}' for c in _ARM_COLS]
    + ['head_x', 'head_y', 'head_z']
)

# Pose landmark column prefixes needed per side
POSE_COLS = {
    'right': {'shoulder': 'right_shoulder', 'elbow': 'right_elbow', 'wrist': 'right_wrist'},
    'left':  {'shoulder': 'left_shoulder',  'elbow': 'left_elbow',  'wrist': 'left_wrist'},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)

def _xyz(row: pd.Series, prefix: str) -> np.ndarray:
    """Extract [x, y, z] from a DataFrame row given a column prefix."""
    return np.array([row[f'{prefix}_x'], row[f'{prefix}_y'], row[f'{prefix}_z']])

def _build_torso_rotation_matrix(l_sh: np.ndarray, r_sh: np.ndarray, l_hip: np.ndarray, r_hip: np.ndarray) -> np.ndarray:
    """
    Builds a 3x3 rotation matrix R such that v_reachy = R.T @ v_world.

    Reachy base frame axes (expressed in world frame):
      Y : normalize(l_sh - r_sh)           lateral (right -> left shoulder)
      Z : normalize(mid_sh - mid_hip)      vertical (up)
      X : Y x Z                            toward camera (right-hand rule)
    """
    mid_sh  = (l_sh + r_sh)   * 0.5
    mid_hip = (l_hip + r_hip) * 0.5

    y = _normalize(l_sh - r_sh)
    z = _normalize(mid_sh - mid_hip)
    z = _normalize(z - np.dot(z, y) * y)   # re-orthogonalize Z vs Y
    x = np.cross(y, z)                     # X = Y x Z -> toward camera

    return np.column_stack([x, y, z])      # columns = Reachy axes in world frame

# ---------------------------------------------------------------------------
# Torso frame pre-computation
# ---------------------------------------------------------------------------
def _build_all_torso_frames(df_pose: pd.DataFrame) -> list:
    """
    Pre-computes the torso rotation matrix R and shoulder-midpoint origin for every row in df_pose.

    Returns a list of (R, origin) tuples, one per row, where:
      - R     : (3,3) rotation matrix  -> v_reachy = R.T @ (v_world - origin)
      - origin: (3,)  midpoint of left and right shoulder in world space
    """
    frames = []
    for _, row in df_pose.iterrows():
        l_sh  = _xyz(row, 'left_shoulder')
        r_sh  = _xyz(row, 'right_shoulder')
        l_hip = _xyz(row, 'left_hip')
        r_hip = _xyz(row, 'right_hip')
        R      = _build_torso_rotation_matrix(l_sh, r_sh, l_hip, r_hip)
        origin = (l_sh + r_sh) * 0.5
        frames.append((R, origin))
    return frames

# ---------------------------------------------------------------------------
# Per-arm mapping
# ---------------------------------------------------------------------------
def _map_arm(df_pose: pd.DataFrame, side: str, torso_frames: list) -> pd.DataFrame:
    """
    Processes one arm and returns a DataFrame with columns:
      frame, timestamp, sh_x/y/z, elbow_x/y/z, wrist_x/y/z, gripper_angle

    Gripper is fixed at 0 (hand landmarks not available with MediaPipe Pose).

    Parameters:
        df_pose      : cleaned pose DataFrame
        side         : 'right' or 'left'
        torso_frames : list of (R, origin) from _build_all_torso_frames()
    """
    cols = POSE_COLS[side]
    rows = []

    for i, (_, row) in enumerate(df_pose.iterrows()):
        R, origin = torso_frames[i]

        # Landmark positions from world frame to torso frame
        sh_t    = R.T @ (_xyz(row, cols['shoulder']) - origin)
        elbow_t = R.T @ (_xyz(row, cols['elbow'])    - origin)
        wrist_t = R.T @ (_xyz(row, cols['wrist'])    - origin)

        # Scale segments (preserve direction, match Reachy lengths)
        v_upper_scaled = _normalize(elbow_t - sh_t)    * REACHY_UPPER_ARM
        v_lower_scaled = _normalize(wrist_t - elbow_t) * REACHY_FOREARM

        sh_out    = sh_t
        elbow_out = sh_out    + v_upper_scaled
        wrist_out = elbow_out + v_lower_scaled

        rows.append({
            'frame':         int(row['frame']),
            'timestamp':     row['timestamp'],
            'sh_x':          sh_out[0],    'sh_y':    sh_out[1],    'sh_z':    sh_out[2],
            'elbow_x':       elbow_out[0], 'elbow_y': elbow_out[1], 'elbow_z': elbow_out[2],
            'wrist_x':       wrist_out[0], 'wrist_y': wrist_out[1], 'wrist_z': wrist_out[2],
            'gripper_angle': 0.0,          # fixed: hand landmarks not available
        })

    return pd.DataFrame(rows)[['frame', 'timestamp'] + _ARM_COLS]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    subject_name, exercise_name, video_name = ask_inputs()
    folder = DATA_ROOT / "landmarks" / subject_name / exercise_name / video_name
    if not folder.is_dir():
        print(f"Error: folder not found -> {folder}")
        return

    # --- Load and validate pose ---
    pose_path = folder / "pose_cleaned.csv"
    if not pose_path.exists():
        print(f"Error: pose_cleaned.csv not found -> {pose_path}")
        return

    pose_required = []
    for s in ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
              'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'):
        pose_required += [f'{s}_x', f'{s}_y', f'{s}_z']
    df_pose = pd.read_csv(pose_path)
    n_before = len(df_pose)
    df_pose  = df_pose.dropna(subset=pose_required).reset_index(drop=True)
    print(f"\nPose: {len(df_pose)} valid frames (dropped {n_before - len(df_pose)})\n")

    # --- Pre-compute torso frames ---
    torso_frames = _build_all_torso_frames(df_pose)

    # --- Process both arms ---
    arm_dfs = {}
    for side in ('right', 'left'):
        print(f"--- {side} arm ---")
        df_arm       = _map_arm(df_pose, side, torso_frames)
        arm_dfs[side] = df_arm
        print(f"rows mapped: {len(df_arm)}")

    # --- Merge arms and fixed head into a single DataFrame ---
    df_base = df_pose[['frame', 'timestamp']].copy()

    for side, prefix in [('right', 'r'), ('left', 'l')]:
        df_side = arm_dfs[side].drop(columns=['timestamp'])
        df_side = df_side.rename(columns={c: f'{prefix}_{c}' for c in _ARM_COLS})
        df_base = pd.merge(df_base, df_side, on='frame', how='left')

    # Head: fixed neutral gaze point (no face tracking)
    df_base['head_x'] = HEAD_NEUTRAL[0]
    df_base['head_y'] = HEAD_NEUTRAL[1]
    df_base['head_z'] = HEAD_NEUTRAL[2]

    df_out = df_base[MAPPED_HEADER]

    output_path = folder / "poses_mapped.csv"
    df_out.to_csv(output_path, index=False)
    print(f"\nCombined output: {len(df_out)} rows -> {output_path.relative_to(DATA_ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()