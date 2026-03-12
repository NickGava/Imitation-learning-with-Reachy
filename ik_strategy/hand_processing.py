'''
hand_features.py
=============================================================================
Computes hand features from cleaned hand landmark CSVs.

For each hand (right and left), two features are computed per frame:

  1. Gripper angle  — motor angle (degrees) ready to send to Reachy.
                      Computed from the normalized pinch distance between thumb
                      tip and index tip (scale-invariant, normalized by the
                      wrist->index_mcp distance), then mapped linearly to the
                      Reachy motor range:

                        Right hand:  open = -69°,  closed = 20°
                        Left  hand:  open =  69°,  closed = -20°

                      where abs(69) corresponds to fully open gripper.

  2. Hand orientation — quaternion [q_w, q_x, q_y, q_z] describing the
                        orientation of the hand reference frame relative to the
                        camera frame. Built from three landmarks:
                          - wrist      (origin)
                          - index_mcp  (defines finger / forward direction)
                          - pinky_mcp  (defines lateral direction)

                        Reference frame construction:
                          e1 = normalize(index_mcp − wrist)       forward
                          e3 = normalize(e1 × (pinky_mcp − wrist)) palm normal
                          e2 = e3 × e1                             lateral

                        The rotation matrix R = [e1 | e2 | e3] is then
                        converted to a unit quaternion (w, x, y, z).

Note on MediaPipe hand coordinates:
  Hand landmarks use normalized image coordinates — x and y are reliable,
  z is a relative depth estimate (less accurate). Orientation on the XY plane
  is high quality; roll (depth axis) should be interpreted with more caution.

Input:
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/{right,left}_hand_cleaned.csv

Output (same folder):
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/right_hand_features.csv
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/left_hand_features.csv

Each output row:
  frame, timestamp, gripper_angle, q_w, q_x, q_y, q_z
'''

import numpy as np
import pandas as pd
from config import DATA_ROOT

# ---------------------------------------------------------------------------
# Reachy gripper motor ranges (degrees)
#   open  = abs(69°) — fully open gripper
#   closed = abs(20°) — fully closed gripper
# ---------------------------------------------------------------------------
GRIPPER_RANGE = {
    'right_hand': {'open': -40.0, 'closed':  20.0},
    'left_hand':  {'open':  40.0, 'closed': -20.0},
}

# Output header
FEATURES_HEADER = ['frame', 'timestamp', 'gripper_angle', 'q_w', 'q_x', 'q_y', 'q_z']


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def _normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a 3D vector. Returns zero vector if norm is near zero."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)


def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a unit quaternion [w, x, y, z].
    Uses the Shepperd method for numerical stability.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)       # ensure unit quaternion


# ---------------------------------------------------------------------------
# Per-frame feature computation
# ---------------------------------------------------------------------------
def _compute_gripper_angle(
    thumb_tip : np.ndarray,
    index_tip : np.ndarray,
    wrist     : np.ndarray,
    index_mcp : np.ndarray,
    side      : str
) -> float:
    """
    Computes the Reachy gripper motor angle (degrees) for a single frame.

    Steps:
      1. Compute normalized pinch closure in [0.0, 1.0]
           0.0 = fully open, 1.0 = fully closed
      2. Map linearly to the Reachy motor range for the given side:
           angle = open + closure * (closed - open)

    Parameters:
        thumb_tip, index_tip, wrist, index_mcp : 3D landmark positions
        side : 'right_hand' or 'left_hand'

    Returns:
        float : motor angle in degrees
    """
    pinch_dist = np.linalg.norm(thumb_tip - index_tip)
    scale      = np.linalg.norm(index_mcp - wrist)

    if scale < 1e-9:
        closure = 0.0
    else:
        # Raw ratio: 0 = closed, large = open → invert and clip to [0, 1]
        closure = float(np.clip(1.0 - pinch_dist / scale, 0.0, 1.0))
    
    r = GRIPPER_RANGE[side]
    return r['open'] + closure * (r['closed'] - r['open'])


def _compute_orientation(wrist : np.ndarray, index_mcp : np.ndarray, pinky_mcp : np.ndarray, side = 'right_hand') -> np.ndarray:
    """
    Builds a hand reference frame from three landmarks and returns the
    corresponding unit quaternion [w, x, y, z].

     Frame construction:
      e3 = normalize(-(index_mcp - wrist))           forward (finger direction)
      e1 = normalize(-(pinky_mcp - wrist) x e3)    palm normal (out of palm)
      e2 = e3 x e1                                lateral (completes frame)
    """
    v_forward = index_mcp - wrist
    v_lateral = pinky_mcp - wrist

    e3 = _normalize(-v_forward)
    
    sign = -1.0 if side == 'right_hand' else +1.0
    e1 = _normalize(np.cross(sign * v_lateral, e3))
    e2 = np.cross(e3, e1)

    R = np.column_stack([e1, e2, e3])
    return _rotation_matrix_to_quaternion(R)


# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------
def _process_hand(input_path, output_path, side: str) -> None:
    """
    Reads the cleaned hand CSV, computes features for each frame, and saves the result.

    Parameters:
        input_path  : path to {side}_cleaned.csv
        output_path : path to {side}_features.csv
        side        : 'right_hand' or 'left_hand' — determines gripper motor range
    """
    df = pd.read_csv(input_path)
    n_loaded = len(df)

    required_cols = [
        'thumb_tip_x', 'thumb_tip_y', 'thumb_tip_z',
        'index_tip_x', 'index_tip_y', 'index_tip_z',
        'wrist_x',     'wrist_y',     'wrist_z',
        'index_mcp_x', 'index_mcp_y', 'index_mcp_z',
        'pinky_mcp_x', 'pinky_mcp_y', 'pinky_mcp_z',
    ]

    df = df.dropna(subset=required_cols).reset_index(drop=True)
    n_valid = len(df)
    dropped = n_loaded - n_valid
    if dropped:
        print(f"incomplete rows skipped: {dropped}")

    rows = []
    for _, row in df.iterrows():
        thumb_tip  = np.array([row['thumb_tip_x'],  row['thumb_tip_y'],  row['thumb_tip_z']])
        index_tip  = np.array([row['index_tip_x'],  row['index_tip_y'],  row['index_tip_z']])
        wrist      = np.array([row['wrist_x'],      row['wrist_y'],      row['wrist_z']])
        index_mcp  = np.array([row['index_mcp_x'],  row['index_mcp_y'],  row['index_mcp_z']])
        pinky_mcp  = np.array([row['pinky_mcp_x'],  row['pinky_mcp_y'],  row['pinky_mcp_z']])

        angle = _compute_gripper_angle(thumb_tip, index_tip, wrist, index_mcp, side)
        q     = _compute_orientation(wrist, index_mcp, pinky_mcp, side)

        rows.append([row['frame'], row['timestamp'], angle, q[0], q[1], q[2], q[3]])

    out = pd.DataFrame(rows, columns=FEATURES_HEADER)
    out.to_csv(output_path, index=False)
    print(f"rows saved: {len(out)} / {n_loaded}")
    print(f"→ {output_path}")


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

    landmarks_folder = DATA_ROOT / "landmarks" / subject_name / exercise_name / video_name

    if not landmarks_folder.is_dir():
        print(f"Error: folder not found → {landmarks_folder}")
        return

    for side in ('right_hand', 'left_hand'):
        input_path  = landmarks_folder / f"{side}_cleaned.csv"
        output_path = landmarks_folder / f"{side}_mapped.csv"

        if not input_path.exists():
            print(f"[SKIP] {side}_cleaned.csv not found.\n")
            continue

        print(f"--- {side} ---")
        _process_hand(input_path, output_path, side)
        print()

    print("Done.")


if __name__ == "__main__":
    main()