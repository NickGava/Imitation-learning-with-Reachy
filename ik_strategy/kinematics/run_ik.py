'''
run_ik.py
=============================================================================
Computes Reachy motor angles from mapped arm trajectories.

For each arm reads:
  - {side}_arm_mapped.csv    → joint positions in torso frame + gripper + quaternion
  - pose_cleaned.csv         → re-used to compute per-frame R_torso, needed to
                               rotate the hand quaternion from camera frame to
                               torso frame (the quaternion in arm_mapped.csv is
                               still in camera/world frame from MediaPipe)

Pipeline per frame:
  1. Convert wrist quaternion: camera frame → torso frame → Reachy base frame
  2. Build 4×4 target pose for inverse_kinematics()
       - position : wrist_{x,y,z} + REACHY_SHOULDER_OFFSET  (torso → Reachy base)
       - rotation : R_reachy_base = R_TORSO_TO_REACHY @ R_torso.T @ R_from_q
  3. Call reachy.{side}_arm.inverse_kinematics(target_pose, q0=prev_angles)
       seed = previous frame angles  (option 1: smooth, physically consistent)
  4. Save motor angles + gripper_angle to arm_ik.csv

Output:
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/right_arm_ik.csv
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/left_arm_ik.csv

Each output row:
  frame, timestamp,
  shoulder_pitch, shoulder_roll, arm_yaw,
  elbow_pitch, forearm_yaw,
  wrist_pitch, wrist_roll,
  gripper_angle

CALIBRATION NOTES (⚠ verify before first run):
  REACHY_R_SHOULDER_OFFSET  : position of right shoulder joint in Reachy base frame
  REACHY_L_SHOULDER_OFFSET  : position of left  shoulder joint in Reachy base frame
  R_TORSO_TO_REACHY         : rotation aligning our torso frame with Reachy base frame
                               Set to identity for first debug — adjust if needed.
  JOINT_LIMITS              : used to warn (not clamp) if IK goes out of range.
'''

import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_ROOT

from reachy_sdk import ReachySDK

# ---------------------------------------------------------------------------
# Reachy morphology constants  ⚠ verify against SDK / URDF
# ---------------------------------------------------------------------------

# Shoulder joint positions in Reachy's base frame (meters)
# From the technical drawing: shoulders are ~190mm apart laterally,
# symmetric. Adjust Y/Z offsets based on actual URDF.
REACHY_R_SHOULDER_OFFSET = np.array([ 0.095,  0.0,  0.0])   # right shoulder
REACHY_L_SHOULDER_OFFSET = np.array([-0.095,  0.0,  0.0])   # left  shoulder

# Rotation from our torso frame to Reachy base frame.
# Our torso frame: X=right, Y=up, Z=forward
# Reachy base frame: verify with SDK — set to identity for first debug run.
R_TORSO_TO_REACHY = np.eye(3)   # ⚠ adjust after first visual check

# Default (rest) joint angles used as seed for the first frame (degrees)
REACHY_R_ARM_REST = np.array([0., -5., 0., -90., 0., 0., 0.])
REACHY_L_ARM_REST = np.array([0.,  5., 0., -90., 0., 0., 0.])

# Joint order returned by inverse_kinematics()
JOINT_NAMES = [
    'shoulder_pitch', 'shoulder_roll', 'arm_yaw',
    'elbow_pitch', 'forearm_yaw',
    'wrist_pitch', 'wrist_roll',
]

# Approximate joint limits for warning only (degrees)  ⚠ verify against SDK
JOINT_LIMITS = {
    'shoulder_pitch': (-180,  60),
    'shoulder_roll':  ( -15, 180),
    'arm_yaw':        ( -90,  90),
    'elbow_pitch':    (-125,   0),
    'forearm_yaw':    ( -90,  90),
    'wrist_pitch':    ( -45,  45),
    'wrist_roll':     ( -55,  55),
}

# Output header
IK_HEADER = ['frame', 'timestamp'] + JOINT_NAMES + ['gripper_angle']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)


def _xyz(row: pd.Series, prefix: str) -> np.ndarray:
    return np.array([row[f'{prefix}_x'], row[f'{prefix}_y'], row[f'{prefix}_z']])


def _build_torso_rotation(l_sh, r_sh, l_hip, r_hip) -> np.ndarray:
    """Same logic as run_mapping.py — rebuild R_torso per frame."""
    mid_sh  = (l_sh + r_sh)   * 0.5
    mid_hip = (l_hip + r_hip) * 0.5

    x = _normalize(r_sh - l_sh)
    y = _normalize(mid_sh - mid_hip)
    y = _normalize(y - np.dot(y, x) * x)
    z = np.cross(x, y)

    return np.column_stack([x, y, z])   # R.T maps world → torso


def _quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [w, x, y, z] to 3×3 rotation matrix."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def _build_target_pose(wrist_torso: np.ndarray, R_wrist_reachy: np.ndarray,
                       shoulder_offset: np.ndarray) -> np.ndarray:
    """
    Assembles the 4×4 homogeneous target pose for inverse_kinematics().

    Position: wrist in Reachy base frame
      p_reachy = R_TORSO_TO_REACHY @ wrist_torso + shoulder_offset

    Rotation: wrist orientation already rotated to Reachy base frame.
    """
    p = R_TORSO_TO_REACHY @ wrist_torso + shoulder_offset

    T = np.eye(4)
    T[:3, :3] = R_wrist_reachy
    T[:3,  3] = p
    return T


def _check_limits(angles: np.ndarray, frame_idx: int) -> None:
    """Warns (does not clamp) if any joint is near its limit."""
    for name, angle in zip(JOINT_NAMES, angles):
        lo, hi = JOINT_LIMITS[name]
        if not (lo <= angle <= hi):
            print(f"  ⚠ frame {frame_idx}: {name} = {angle:.1f}° out of [{lo}, {hi}]")


# ---------------------------------------------------------------------------
# Per-arm IK
# ---------------------------------------------------------------------------
def _run_ik(df_mapped: pd.DataFrame, df_pose: pd.DataFrame,
            reachy_arm, shoulder_offset: np.ndarray,
            rest_angles: np.ndarray) -> pd.DataFrame:
    """
    Runs IK for all frames of one arm.

    Parameters:
        df_mapped        : loaded {side}_arm_mapped.csv
        df_pose          : loaded pose_cleaned.csv (for per-frame R_torso)
        reachy_arm       : reachy.r_arm or reachy.l_arm
        shoulder_offset  : REACHY_{R,L}_SHOULDER_OFFSET
        rest_angles      : seed for the first frame

    Returns:
        DataFrame with IK_HEADER columns.
    """
    # Merge on frame index so rows are aligned
    df = pd.merge(df_mapped, df_pose, on='frame', suffixes=('', '_pose'))
    df = df.sort_values('frame').reset_index(drop=True)

    prev_angles = rest_angles.copy()
    rows = []
    n_failed = 0

    for _, row in df.iterrows():
        frame_idx = int(row['frame'])

        # --- Per-frame torso rotation ---
        l_sh  = _xyz(row, 'left_shoulder')
        r_sh  = _xyz(row, 'right_shoulder')
        l_hip = _xyz(row, 'left_hip')
        r_hip = _xyz(row, 'right_hip')
        R_torso = _build_torso_rotation(l_sh, r_sh, l_hip, r_hip)
        # R_torso.T maps world → torso  (already applied to positions in run_mapping)
        # R_torso   maps torso → world

        # --- Wrist orientation ---
        q = np.array([row['q_w'], row['q_x'], row['q_y'], row['q_z']])
        R_cam  = _quat_to_rotation_matrix(q)          # orientation in camera/world frame
        R_torso_frame = R_torso.T @ R_cam             # rotate to torso frame
        R_reachy = R_TORSO_TO_REACHY @ R_torso_frame  # rotate to Reachy base frame

        # --- Wrist position ---
        wrist_torso = np.array([row['wrist_x'], row['wrist_y'], row['wrist_z']])

        # --- Build 4×4 target pose ---
        target_pose = _build_target_pose(wrist_torso, R_reachy, shoulder_offset)

        # --- IK (option 1: seed = previous frame) ---
        try:
            angles = reachy_arm.inverse_kinematics(target_pose, q0=prev_angles)
            prev_angles = np.array(angles)
            _check_limits(prev_angles, frame_idx)
        except Exception as e:
            n_failed += 1
            print(f"  ✗ frame {frame_idx}: IK failed ({e}) — using previous angles")
            angles = prev_angles.tolist()

        rows.append([frame_idx, row['timestamp']] + list(angles) + [row['gripper_angle']])

    if n_failed:
        print(f"  Total IK failures: {n_failed} / {len(df)}")

    return pd.DataFrame(rows, columns=IK_HEADER)


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

    # --- Load pose (needed for per-frame R_torso) ---
    pose_path = folder / "pose_cleaned.csv"
    if not pose_path.exists():
        print(f"Error: pose_cleaned.csv not found → {pose_path}")
        return

    torso_cols = []
    for lm in ('left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'):
        torso_cols += [f'{lm}_x', f'{lm}_y', f'{lm}_z']

    df_pose = pd.read_csv(pose_path).dropna(subset=torso_cols).reset_index(drop=True)
    print(f"Pose loaded: {len(df_pose)} frames\n")

    reachy = ReachySDK(host="localhost")

    config = {
        'right': {
            'mapped_file':  'right_arm_mapped.csv',
            'output_file':  'right_arm_ik.csv',
            'arm':          reachy.r_arm,
            'sh_offset':    REACHY_R_SHOULDER_OFFSET,
            'rest_angles':  REACHY_R_ARM_REST,
        },
        'left': {
            'mapped_file':  'left_arm_mapped.csv',
            'output_file':  'left_arm_ik.csv',
            'arm':          reachy.l_arm,
            'sh_offset':    REACHY_L_SHOULDER_OFFSET,
            'rest_angles':  REACHY_L_ARM_REST,
        },
    }

    for side, cfg in config.items():
        print(f"=== {side} arm ===")

        mapped_path = folder / cfg['mapped_file']
        if not mapped_path.exists():
            print(f"[SKIP] {cfg['mapped_file']} not found.\n")
            continue

        df_mapped = pd.read_csv(mapped_path)

        df_ik = _run_ik(
            df_mapped     = df_mapped,
            df_pose       = df_pose,
            reachy_arm    = cfg['arm'],
            shoulder_offset = cfg['sh_offset'],
            rest_angles   = cfg['rest_angles'],
        )

        output_path = folder / cfg['output_file']
        df_ik.to_csv(output_path, index=False)
        print(f"  rows saved: {len(df_ik)}")
        print(f"  → {output_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()