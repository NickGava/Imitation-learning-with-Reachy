'''
mapping.py
=============================================================================
Maps cleaned pose landmarks and hand features to Reachy arm space, and
transforms the head gaze point from face features.

For each arm (right and left), reads:
  - pose_cleaned.csv         -> shoulder, elbow, wrist positions (world space)
  - {right,left}_hand_mapped.csv   -> gripper_angle, hand orientation quaternion

For the head, reads:
  - face_features.csv        -> head forward unit vector (world space)

Pipeline per arm:
  1. Build Reachy base frame directly from pose landmarks
       Origin: midpoint between left and right shoulder
       Y-axis: right shoulder -> left shoulder  (lateral)
       Z-axis: midpoint hips -> midpoint shoulders  (vertical, up)
       X-axis: Y x Z  (forward, toward camera, right-hand rule)

  2. Express shoulder, elbow, wrist in torso frame

  3. Scale arm segments to Reachy proportions
       Upper arm (shoulder -> elbow): REACHY_UPPER_ARM = 0.280 m
       Forearm   (elbow   -> wrist) : REACHY_FOREARM   = 0.250 m

  4. Merge with hand features on frame index; rotate the hand orientation
     quaternion from the MediaPipe camera frame to the Reachy torso frame

Pipeline for the head:
  1. Rotate the head forward unit vector (world space) to Reachy torso frame
     using the same per-frame rotation matrix R computed for the arms

  2. Get nose position in torso frame (used as approximate head origin)

  3. Compute gaze point = nose_torso + 1.0 * forward_torso
     This is the 3D point the head is looking at, directly usable by
     ReachySDK's lookAt(x, y, z).

Merge right and left arm DataFrames and the head gaze into a single output file.

Output (same folder as input):
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/poses_mapped.csv
'''

import numpy as np
import pandas as pd

from config import DATA_ROOT, GRIPPER_RANGE
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
    'q_w', 'q_x', 'q_y', 'q_z',
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

    lateral = l_sh - r_sh
    lateral[2] = 0.0                        # ← ignora Z: spalle alla stessa profondità
    y = _normalize(lateral)
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
def _map_arm(df_pose: pd.DataFrame, df_hand: pd.DataFrame, side: str, torso_frames: list) -> pd.DataFrame:
    """
    Processes one arm and returns a DataFrame with columns:
      frame, timestamp, sh_x/y/z, elbow_x/y/z, wrist_x/y/z,
      gripper_angle, q_w, q_x, q_y, q_z

    Parameters:
        df_pose      : cleaned pose DataFrame
        df_hand      : hand features DataFrame ({side}_hand_mapped.csv)
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
            'frame':     int(row['frame']),
            'timestamp': row['timestamp'],
            'sh_x':    sh_out[0],    'sh_y':    sh_out[1],    'sh_z':    sh_out[2],
            'elbow_x': elbow_out[0], 'elbow_y': elbow_out[1], 'elbow_z': elbow_out[2],
            'wrist_x': wrist_out[0], 'wrist_y': wrist_out[1], 'wrist_z': wrist_out[2],
        })

    df_mapped = pd.DataFrame(rows)

    # Merge hand features
    df_merged = pd.merge(
        df_mapped,
        df_hand[['frame', 'gripper_angle', 'q_w', 'q_x', 'q_y', 'q_z']],
        on='frame', how='left'
    )
    # Handle missing hand features by interpolation
    for col in ['gripper_angle', 'q_w', 'q_x', 'q_y', 'q_z']:
        df_merged[col] = df_merged[col].interpolate(method='linear', limit_direction='both')
    # Fallback for any remaining NaNs
    neutral_gripper = GRIPPER_RANGE[f'{side}_hand']['open']
    df_merged['gripper_angle'] = df_merged['gripper_angle'].fillna(neutral_gripper)
    df_merged['q_w'] = df_merged['q_w'].fillna(1.0)
    df_merged['q_x'] = df_merged['q_x'].fillna(0.0)
    df_merged['q_y'] = df_merged['q_y'].fillna(0.0)
    df_merged['q_z'] = df_merged['q_z'].fillna(0.0)

    # Rotate quaternion: MediaPipe world frame -> Reachy torso frame
    q_out = np.zeros((len(df_merged), 4))
    for i, (_, row) in enumerate(df_merged.iterrows()):
        q_world  = np.array([row['q_w'], row['q_x'], row['q_y'], row['q_z']])
        R        = torso_frames[i][0]
        R_reachy = R.T @ _quat_to_R(q_world)
        q_out[i] = _R_to_quat(R_reachy)

    df_merged[['q_w', 'q_x', 'q_y', 'q_z']] = q_out

    return df_merged[['frame', 'timestamp'] + _ARM_COLS]

# ---------------------------------------------------------------------------
# Head gaze mapping
# ---------------------------------------------------------------------------
def _map_head(df_pose: pd.DataFrame, df_face: pd.DataFrame, torso_frames: list) -> pd.DataFrame:
    """
    Computes the head gaze point in Reachy's torso frame for each pose frame.

    For each frame:
      1. Rotate the head forward unit vector (world space, from face_features.csv)
         into the Reachy torso frame using the pre-computed rotation matrix R.
      2. Express the nose position in the torso frame.
      3. Compute: gaze_point = nose_torso + 1.0 * forward_torso

    Parameters:
        df_pose      : cleaned pose DataFrame (must contain nose_x/y/z)
        df_face      : face features DataFrame (face_features.csv)
        torso_frames : list of (R, origin) from _build_all_torso_frames()

    Returns:
        DataFrame with columns: frame, timestamp, head_x, head_y, head_z
    """
    # --- Pre-interpolation of forward vector on each pose frame to fill NaNs ---
    all_frames = df_pose['frame'].astype(int).tolist()
    df_fwd = pd.DataFrame({'frame': all_frames})
    df_fwd = pd.merge(df_fwd, df_face[['frame', 'head_dx', 'head_dy', 'head_dz']], on='frame', how='left')  # merge
    for col in ['head_dx', 'head_dy', 'head_dz']:
        df_fwd[col] = df_fwd[col].interpolate(method='linear', limit_direction='both')  # interpolate     

    # Fallback for any remaining NaNs
    df_fwd['head_dx'] = df_fwd['head_dx'].fillna(1.0)
    df_fwd['head_dy'] = df_fwd['head_dy'].fillna(0.0)
    df_fwd['head_dz'] = df_fwd['head_dz'].fillna(0.0)

    # Re-normalize after linear interpolation
    norms = np.linalg.norm(df_fwd[['head_dx', 'head_dy', 'head_dz']].values, axis=1, keepdims=True)
    norms = np.where(norms < 1e-9, 1.0, norms)
    df_fwd[['head_dx', 'head_dy', 'head_dz']] = df_fwd[['head_dx', 'head_dy', 'head_dz']].values / norms

    # Quick lookup: frame -> interpolated forward vector
    face_lookup = {int(row['frame']): np.array([row['head_dx'], row['head_dy'], row['head_dz']])
                   for _, row in df_fwd.iterrows()}

    # --- Main loop ---
    rows = []
    for i, (_, row) in enumerate(df_pose.iterrows()):
        R, origin = torso_frames[i]
        frame_id  = int(row['frame'])

        nose_world = _xyz(row, 'nose')
        nose_torso = R.T @ (nose_world - origin)

        fwd_world = face_lookup[frame_id]
        fwd_torso = R.T @ fwd_world

        gaze_point = nose_torso + 1.0 * fwd_torso

        rows.append({
            'frame':     frame_id,
            'timestamp': row['timestamp'],
            'head_x':    gaze_point[0],
            'head_y':    gaze_point[1],
            'head_z':    gaze_point[2],
        })

    return pd.DataFrame(rows)

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
              'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
              'nose'):
        pose_required += [f'{s}_x', f'{s}_y', f'{s}_z']
    df_pose = pd.read_csv(pose_path)
    n_before = len(df_pose)
    df_pose  = df_pose.dropna(subset=pose_required).reset_index(drop=True)
    print(f"\nPose: {len(df_pose)} valid frames (dropped {n_before - len(df_pose)})\n")

    # --- Pre-compute torso frames ---
    torso_frames = _build_all_torso_frames(df_pose)

    # --- Load face features ---
    face_path = folder / "face_features.csv"
    if face_path.exists():
        df_face = pd.read_csv(face_path)
        print(f"Face: {len(df_face)} frames loaded from face_features.csv\n")
    else:
        print("[WARNING] face_features.csv not found - head gaze will use neutral default.\n")
        df_face = pd.DataFrame(columns=['frame', 'timestamp', 'head_dx', 'head_dy', 'head_dz'])

    # --- Process both arms ---
    arm_dfs = {}
    for side, hand_side in [('right', 'right_hand'), ('left', 'left_hand')]:
        print(f"--- {side} arm ---")

        hand_path = folder / f"{hand_side}_mapped.csv"
        if not hand_path.exists():
            print(f"[SKIP] {hand_side}_mapped.csv not found - arm will be NaN.\n")
            arm_dfs[side] = None
            continue

        df_hand = pd.read_csv(hand_path)
        df_arm  = _map_arm(df_pose, df_hand, side, torso_frames)
        print(f"rows mapped: {len(df_arm)}")
        arm_dfs[side] = df_arm

    # --- Process head ---
    print("--- head ---")
    df_head = _map_head(df_pose, df_face, torso_frames)
    print(f"  rows mapped: {len(df_head)}")

    # --- Merge arms and head into a single DataFrame ---
    df_base = df_pose[['frame', 'timestamp']].copy()

    for side, prefix in [('right', 'r'), ('left', 'l')]:
        if arm_dfs[side] is not None:
            df_side = arm_dfs[side].drop(columns=['timestamp'])
            df_side = df_side.rename(columns={c: f'{prefix}_{c}' for c in _ARM_COLS})
            df_base = pd.merge(df_base, df_side, on='frame', how='left')
        else:
            for c in _ARM_COLS:
                df_base[f'{prefix}_{c}'] = np.nan

    df_base = pd.merge(df_base, df_head[['frame', 'head_x', 'head_y', 'head_z']], on='frame', how='left')

    df_out = df_base[MAPPED_HEADER]

    output_path = folder / "poses_mapped.csv"
    df_out.to_csv(output_path, index=False)
    print(f"\nCombined output: {len(df_out)} rows -> {output_path.relative_to(DATA_ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
