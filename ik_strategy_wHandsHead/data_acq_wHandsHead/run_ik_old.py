'''
run_ik.py
=============================================================================
Offline IK solver with dual constraint: wrist position + elbow position.

The FK is re-implemented in pure numpy from the Reachy URDF; the IK is
solved frame-by-frame via numerical optimisation (scipy L-BFGS-B).

--- Forward Kinematics ---
The FK chain propagates a 4x4 homogeneous transform from the torso origin
through the 7 arm joints using rotation matrices R(u,θ).

--- Optimisation ---
For each frame, L-BFGS-B minimises a weighted cost over the 7 joint angles:
  cost = W_WRIST_POS · ‖wrist_fk - wrist_target‖²
       + W_ELBOW_POS · ‖elbow_fk - elbow_target‖²
       + W_WRIST_ORI · ‖R_fk - R_target‖²        
       + W_SMOOTH    · ‖q - q_prev‖²             

  Joint limits are enforced as hard bounds in the optimiser.
  The solution of the previous frame is used as warm start (q0 = q_prev).
  If a frame does not converge (success=False and cost ≥ 1e-4), the previous angles are kept.

--- Head ---
The head gaze point (head_x, head_y, head_z) from poses_mapped.csv is passed through to the output, no modification. 

--- Input / Output ---
Input:
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/poses_mapped.csv

Output:
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/joint_ik.csv
'''

import numpy as np
import pandas as pd
import scipy.optimize as opt

from config import DATA_ROOT, HEAD_COLS, JOINT_COLS, JOINT_LIMITS_DEG, JOINT_LIMIT_PADDING_DEG, REST_DEG
from ask_inputs import ask_inputs

# FK geometry: read from reachy.URDF
_FK_JOINTS = [
    # name            translation                      axis
    ('shoulder_pitch', None,                            np.array([0., 1., 0.])),
    ('shoulder_roll',  np.array([0., 0., 0.]),          np.array([1., 0., 0.])),
    ('arm_yaw',        np.array([0., 0., 0.]),          np.array([0., 0., 1.])),
    ('elbow_pitch',    np.array([0., 0., -0.28]),       np.array([0., 1., 0.])),  
    ('forearm_yaw',    np.array([0., 0., 0.]),          np.array([0., 0., 1.])),
    ('wrist_pitch',    np.array([0., 0., -0.25]),       np.array([0., 1., 0.])),  
    ('wrist_roll',     np.array([0., 0., -0.0325]),     np.array([1., 0., 0.])),  
]

_ELBOW_JOINT_IDX = 3
_WRIST_POS_IDX   = 5
_WRIST_ORI_IDX   = 6

SHOULDER_Y_OFFSET = {
    'right': -0.19,
    'left':   0.19,
}

# Reachy joint constants
#JOINT_NAMES = ['shoulder_pitch', 'shoulder_roll', 'arm_yaw', 'elbow_pitch', 'forearm_yaw', 'wrist_pitch', 'wrist_roll']

# Loss weights
W_WRIST_POS = 1.0
W_ELBOW_POS = 0.8
W_WRIST_ORI = 0
W_SMOOTH    = 0.05

# Output format
JOINTS_IK_HEADER = (['frame', 'timestamp'] + JOINT_COLS + HEAD_COLS)

# ---------------------------------------------------------------------------
# Pure numpy forward kinematics
# ---------------------------------------------------------------------------
def _rot(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """
    Rotation matrix R(u,θ): rotate by angle_rad around unit vector axis.
    Returns a 3x3 matrix.
    """
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    x, y, z = axis
    return np.array([
        [c + x*x*(1-c),     x*y*(1-c) - z*s,  x*z*(1-c) + y*s],
        [y*x*(1-c) + z*s,   c + y*y*(1-c),    y*z*(1-c) - x*s],
        [z*x*(1-c) - y*s,   z*y*(1-c) + x*s,  c + z*z*(1-c)  ],
    ])

def fk(q_rad: np.ndarray, side: str):
    """
    Forward kinematics for one arm.

    Parameters:
        q_rad : (7,) joint angles in radians, order = JOINT_NAMES
        side  : 'right' or 'left'

    Returns:
        elbow_pos (3,)  — position of elbow in torso frame
        wrist_pos (3,)  — position of wrist in torso frame
        wrist_R   (3,3) — rotation matrix of end-effector in torso frame
    """
    T = np.eye(4)
    elbow_pos = None
    wrist_pos = None
    wrist_R   = None

    for i, (_, trans, axis) in enumerate(_FK_JOINTS):
        if i == 0:
            trans = np.array([0., SHOULDER_Y_OFFSET[side], 0.])

        T[:3, 3] += T[:3, :3] @ trans
        R = _rot(axis, q_rad[i])
        T[:3, :3] = T[:3, :3] @ R

        if i == _ELBOW_JOINT_IDX:
            elbow_pos = T[:3, 3].copy()
        if i == _WRIST_POS_IDX:
            wrist_pos = T[:3, 3].copy()
        if i == _WRIST_ORI_IDX:
            wrist_R = T[:3, :3].copy()

    return elbow_pos, wrist_pos, wrist_R

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _quat_to_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])

def _cost(q, side, target_elbow, target_wrist, target_R, prev_q):
    elbow_fk, wrist_fk, R_fk = fk(q, side)
    e_wrist_pos = np.sum((wrist_fk - target_wrist) ** 2)
    e_elbow_pos = np.sum((elbow_fk - target_elbow) ** 2)
    e_ori       = np.sum((R_fk     - target_R)     ** 2)
    e_smooth    = np.sum((q        - prev_q)       ** 2)
    return (
        W_WRIST_POS * e_wrist_pos
        + W_ELBOW_POS * e_elbow_pos
        + W_WRIST_ORI * e_ori
        + W_SMOOTH    * e_smooth
    )

# ---------------------------------------------------------------------------
# Per-arm IK loop
# ---------------------------------------------------------------------------
def _run_ik_arm(df: pd.DataFrame, p: str, side: str) -> np.ndarray:
    """
    Optimises joint angles for every frame of one arm.
    Returns (N, 8): 7 joint angles [deg] + gripper angle [deg].
    """
    limits_rad = np.deg2rad(JOINT_LIMITS_DEG[side] + np.array([JOINT_LIMIT_PADDING_DEG, -JOINT_LIMIT_PADDING_DEG]))
    prev_q     = np.deg2rad(REST_DEG[side])

    n_frames = len(df)
    out      = np.zeros((n_frames, 8))
    n_warn   = 0

    for i, (_, row) in enumerate(df.iterrows()):

        target_elbow = np.array([row[f'{p}_elbow_x'], row[f'{p}_elbow_y'], row[f'{p}_elbow_z']])
        target_wrist = np.array([row[f'{p}_wrist_x'], row[f'{p}_wrist_y'], row[f'{p}_wrist_z']])
        q_quat       = np.array([row[f'{p}_q_w'], row[f'{p}_q_x'], row[f'{p}_q_y'], row[f'{p}_q_z']])
        target_R     = _quat_to_R(q_quat)

        result = opt.minimize(
            _cost,
            x0      = prev_q,
            args    = (side, target_elbow, target_wrist, target_R, prev_q),
            method  = 'L-BFGS-B',
            bounds  = limits_rad,
            options = {'maxiter': 300, 'ftol': 1e-9},
        )

        if result.success or result.fun < 1e-4:
            q_opt = result.x
        else:
            n_warn += 1
            print(f"  ⚠  frame {int(row['frame'])}: did not converge "
                  f"(cost={result.fun:.5f}) -- keeping previous angles")
            q_opt = prev_q

        prev_q     = q_opt
        out[i, :7] = np.rad2deg(q_opt)
        out[i,  7] = row[f'{p}_gripper_angle']

        if (i + 1) % 100 == 0:
            print(f"Frame {i+1:4d} / {n_frames}")

    if n_warn:
        print(f"Convergence warnings: {n_warn} / {n_frames}")

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    subject_name, exercise_name, video_name = ask_inputs()
    folder = DATA_ROOT / "landmarks" / subject_name / exercise_name / video_name
    if not folder.is_dir():
        print(f"Error: folder not found → {folder}")
        return

    mapped_path = folder / "poses_mapped.csv"
    if not mapped_path.exists():
        print(f"Error: poses_mapped.csv not found → {mapped_path}")
        return

    df = pd.read_csv(mapped_path)
    print(f"Loaded {len(df)} frames from {mapped_path.relative_to(DATA_ROOT)}\n")

    # Run IK per arm
    results = {}
    for side, prefix in [('right', 'r'), ('left', 'l')]:
        print(f"--- {side} arm ---")
        wrist_col = f"{prefix}_wrist_x"

        # Defensive check: if no wrist data, skip IK for this arm and fill with NaNs
        if wrist_col not in df.columns or df[wrist_col].isna().all():
            print(f"[SKIP] No data for {side} arm.\n")
            results[side] = None
            continue

        results[side] = _run_ik_arm(df, prefix, side)
        print(f"IK complete.\n")

    # Assemble output
    out_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        r = [int(row['frame']), row['timestamp']]

        # Arms: 8 values each (7 joints + gripper)
        for side in ('right', 'left'):
            r += list(results[side][i]) if results[side] is not None else [np.nan] * 8

        # Head: pass through gaze point
        r += [row.get('head_x', np.nan),
              row.get('head_y', np.nan),
              row.get('head_z', np.nan)]

        out_rows.append(r)

    df_out = pd.DataFrame(out_rows, columns=JOINTS_IK_HEADER)
    output_path = folder / "joint_ik.csv"
    df_out.to_csv(output_path, index=False)

    print(f"Saved {len(df_out)} rows -> {output_path.relative_to(DATA_ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
