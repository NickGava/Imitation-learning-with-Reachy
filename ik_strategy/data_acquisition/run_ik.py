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
       + W_SMOOTH    · ‖q - q_prev‖²             

  Joint limits are enforced as hard bounds in the optimiser.
  The solution of the previous frame is used as warm start (q0 = q_prev).
  If a frame does not converge (success=False and cost ≥ 1e-4), the previous angles are kept.

--- Output joints ---
Only the 4 joints relevant to arm movement are saved:
  shoulder_pitch, shoulder_roll, arm_yaw, elbow_pitch  (per side)
forearm_yaw, wrist_pitch, wrist_roll and gripper are excluded.
Head gaze is not included.

--- Input / Output ---
Input:
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/poses_mapped.csv

Output:
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/joint_ik.csv
'''

import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.signal import medfilt, savgol_filter

from utilities.config import DATA_ROOT, JOINT_LIMITS_DEG, JOINT_LIMIT_PADDING_DEG, REST_DEG
from utilities.ask_inputs import ask_inputs

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

SHOULDER_Y_OFFSET = {
    'right': -0.19,
    'left':   0.19,
}

# Loss weights
W_WRIST_POS = 1
W_ELBOW_POS = 0.80
W_SMOOTH    = 0.05

# The 4 joints we keep in the output (per side)
_SAVED_JOINT_NAMES = ['shoulder_pitch', 'shoulder_roll', 'arm_yaw', 'elbow_pitch']
_N_SAVED = len(_SAVED_JOINT_NAMES)  # 4

# Output CSV header
JOINTS_IK_HEADER = (
    ['frame', 'timestamp']
    + [f'r_{j}' for j in _SAVED_JOINT_NAMES]
    + [f'l_{j}' for j in _SAVED_JOINT_NAMES]
)

# __________ Tunable __________
SMOOTH_KERNEL  = 5    # medfilt kernel (dispari) — rimuove spike singoli
SMOOTH_POLY    = 3    # savgol polyorder

def _smooth_joints(arr: np.ndarray) -> np.ndarray:
    """
    Post-process IK joint angle timeseries (N, 8).
    Applies median filter then Savitzky-Golay on each joint independently.
    Gripper column (index 7) is left untouched (always 0).
    """
    smoothed = arr.copy()
    n = len(arr)
    SMOOTH_WINDOW  = max(5, int(n*0.2) | 1)   # savgol window  (dispari) — smoothing generale

    # Adatta le window se la sequenza è corta
    kernel = SMOOTH_KERNEL if SMOOTH_KERNEL <= n else (n if n % 2 == 1 else n - 1)
    window = SMOOTH_WINDOW if SMOOTH_WINDOW <= n else kernel

    for j in range(4):   # escludi gripper (col 7)
        col = medfilt(arr[:, j], kernel_size=kernel)
        col = savgol_filter(col, window_length=window, polyorder=SMOOTH_POLY)
        smoothed[:, j] = col

    return smoothed

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
        q_rad : (7,) joint angles in radians, order = shoulder_pitch .. wrist_roll
        side  : 'right' or 'left'

    Returns:
        elbow_pos (3,)  — position of elbow in torso frame
        wrist_pos (3,)  — position of wrist in torso frame
    """
    T = np.eye(4)
    elbow_pos = None
    wrist_pos = None

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

    return elbow_pos, wrist_pos

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cost(q, side, target_elbow, target_wrist, prev_q):
    elbow_fk, wrist_fk = fk(q, side)
    e_wrist_pos = np.sum((wrist_fk - target_wrist) ** 2)
    e_elbow_pos = np.sum((elbow_fk - target_elbow) ** 2)
    e_smooth    = np.sum((q        - prev_q)       ** 2)
    return (
        W_WRIST_POS * e_wrist_pos
        + W_ELBOW_POS * e_elbow_pos
        + W_SMOOTH    * e_smooth
    )

# ---------------------------------------------------------------------------
# Per-arm IK loop
# ---------------------------------------------------------------------------
def _run_ik_arm(df: pd.DataFrame, p: str, side: str) -> np.ndarray:
    """
    Optimises joint angles for every frame of one arm.

    The optimisation still runs over all 7 joints (needed for correct FK),
    but only the first 4 (shoulder_pitch, shoulder_roll, arm_yaw, elbow_pitch)
    are returned.

    Returns (N, 4): the 4 saved joint angles in degrees.
    """
    limits_rad = np.deg2rad(JOINT_LIMITS_DEG[side] + np.array([JOINT_LIMIT_PADDING_DEG, -JOINT_LIMIT_PADDING_DEG]))
    prev_q     = np.deg2rad(REST_DEG[side])

    n_frames = len(df)
    out      = np.zeros((n_frames, _N_SAVED))
    n_warn   = 0

    for i, (_, row) in enumerate(df.iterrows()):

        target_elbow = np.array([row[f'{p}_elbow_x'], row[f'{p}_elbow_y'], row[f'{p}_elbow_z']])
        target_wrist = np.array([row[f'{p}_wrist_x'], row[f'{p}_wrist_y'], row[f'{p}_wrist_z']])

        result = opt.minimize(
            _cost,
            x0      = prev_q,
            args    = (side, target_elbow, target_wrist, prev_q),
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

        prev_q        = q_opt
        out[i, :_N_SAVED] = np.rad2deg(q_opt[:_N_SAVED])  # save only first 4

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
        if wrist_col not in df.columns or df[wrist_col].isna().all():
            print(f"[SKIP] No data for {side} arm.\n")
            results[side] = None
            continue

        results[side] = _smooth_joints(_run_ik_arm(df, prefix, side))
        print(f"IK complete.\n")

    # Assemble output: frame, timestamp, 4 right joints, 4 left joints
    out_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        r = [int(row['frame']), row['timestamp']]

        for side in ('right', 'left'):
            r += list(results[side][i]) if results[side] is not None else [np.nan] * _N_SAVED

        out_rows.append(r)

    df_out = pd.DataFrame(out_rows, columns=JOINTS_IK_HEADER)
    output_path = folder / "joint_ik.csv"
    df_out.to_csv(output_path, index=False)

    print(f"Saved {len(df_out)} rows → {output_path.relative_to(DATA_ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()