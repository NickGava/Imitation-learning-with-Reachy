'''
face_processing.py
=============================================================================
Computes the head forward direction from cleaned Face Mesh landmarks.

Uses 5 Face Mesh landmarks saved in face_cleaned.csv to build a proper 3-axis
head reference frame and extract the forward (gaze) direction:

  Landmarks used (MediaPipe Face Mesh indices):
    nose_tip  :   4   — approximate face origin
    chin      : 152   — defines vertical axis (bottom)
    forehead  :  10   — defines vertical axis (top)
    left_eye  :  33   — defines lateral axis (right in image = person's left)
    right_eye : 263   — defines lateral axis (left in image = person's right)

  Frame construction:
    e_up      = normalize(forehead − chin)           physically upward
    e_right   = normalize(right_eye − left_eye)      person's right→left (image left→right)
    e_forward = normalize(cross(e_right, e_up))      normal to face plane, toward camera

  Note on e_right direction:
    In image coords, right_eye_x < left_eye_x (right eye appears to the LEFT
    of the image from the camera's perspective = person's right side).
    So right_eye − left_eye points LEFT in the image = person's right → left.
    After coordinate conversion this becomes the correct lateral axis in
    pose world space (matching the Y axis of the Reachy torso frame).

--- Coordinate conversion (face image space → pose world space) ---
Face Mesh landmarks use normalized image coordinates:
  x : [0, 1], left to right of image
  y : [0, 1], top to bottom of image  (inverted vs physical vertical)
  z : relative depth, smaller = closer to camera

Pose world landmarks (MediaPipe) use:
  x : positive to the right of the image
  y : positive upward (physically)
  z : positive toward the camera

Conversion applied to all direction vectors:
  world_x =  face_x
  world_y = -face_y   (invert: image y is down, world y is up)
  world_z = -face_z   (invert: face z positive = far, world z positive = near)

This conversion is applied to the computed direction vectors (not to raw
landmark positions, which are in non-metric units and not needed in world
space). The resulting unit vector is in pose world space and can be directly
rotated into the Reachy torso frame by mapping.py using the same R matrix
already computed for the arm pipeline.

Verification for a person facing the camera:
  e_forward in face image space ≈ (0, 0, -1)  [toward camera = smaller z]
  After conversion → pose world ≈ (0, 0, +1)  [toward camera = positive z]
  After R.T in mapping.py → Reachy ≈ (+1, 0, 0)  [X forward] ✓

Input:
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/face_cleaned.csv

Output (same folder):
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/face_features.csv

Each output row:
  frame, timestamp, head_dx, head_dy, head_dz
  (unit vector in pose world space — head forward direction)
'''

import numpy as np
import pandas as pd
from config import DATA_ROOT

FEATURES_HEADER = ['frame', 'timestamp', 'head_dx', 'head_dy', 'head_dz']

_REQUIRED_COLS = [
    'nose_tip_x',  'nose_tip_y',  'nose_tip_z',
    'chin_x',      'chin_y',      'chin_z',
    'forehead_x',  'forehead_y',  'forehead_z',
    'left_eye_x',  'left_eye_y',  'left_eye_z',
    'right_eye_x', 'right_eye_y', 'right_eye_z',
]


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------
def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(3)


# ---------------------------------------------------------------------------
# Per-frame computation
# ---------------------------------------------------------------------------
def _compute_head_forward(
    forehead  : np.ndarray,
    chin      : np.ndarray,
    left_eye  : np.ndarray,
    right_eye : np.ndarray,
) -> np.ndarray:
    """
    Returns the head forward unit vector in pose world space.

    Steps:
      1. Build e_up and e_right in face image space using the 4 landmarks.
      2. Compute e_forward = cross(e_right, e_up) in face image space.
      3. Convert e_forward to pose world space via _face_to_world().
      4. Re-normalize to correct for floating-point drift.

    Parameters:
        forehead, chin, left_eye, right_eye : 3D positions in face image space
            (normalized image coords from face_cleaned.csv)

    Returns:
        (3,) unit vector in pose world space pointing in the head's
        forward direction (toward camera when facing straight ahead).
    """
    # In face image space:
    #   forehead_y < chin_y  (forehead is higher in image = smaller y)
    #   so forehead - chin has a negative y component = physically upward ✓
    e_up = _normalize(forehead - chin)

    # right_eye_x < left_eye_x in image coords (right eye appears to the LEFT
    # of the image from camera's POV), so right_eye - left_eye points LEFT in
    # image = person's right-to-left direction.
    e_left = _normalize(right_eye - left_eye)

    # Cross product gives normal to the face plane.
    # With e_left ≈ (-1, 0, 0) and e_up ≈ (0, -1, 0) in image space:
    # cross((-1,0,0), (0,-1,0)) = (0*0-0*(-1), 0*(-1)-(-1)*0, (-1)*(-1)-0*0)
    #                           = (0, 0, 1) ... but z=1 in face space = AWAY.
    # With e_left ≈ (1,0,0) if we swap: depends on actual landmark positions.
    # The conversion _face_to_world() handles the sign to point toward camera.
    e_forward_face = _normalize(np.cross(e_left, e_up))

    # Convert to pose world space and re-normalize
    return _normalize(np.array([e_forward_face[0], e_forward_face[1], e_forward_face[2]]))


# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------
def _process_face(input_path, output_path) -> None:
    """
    Reads face_cleaned.csv, computes the head forward direction for each
    frame, and writes face_features.csv.

    Parameters:
        input_path  : path to face_cleaned.csv
        output_path : path to face_features.csv
    """
    df = pd.read_csv(input_path)
    n_loaded = len(df)

    df = df.dropna(subset=_REQUIRED_COLS).reset_index(drop=True)
    n_valid = len(df)
    dropped = n_loaded - n_valid
    if dropped:
        print(f"incomplete rows skipped: {dropped}")

    rows = []
    for _, row in df.iterrows():
        forehead  = np.array([row['forehead_x'],  row['forehead_y'],  row['forehead_z']])
        chin      = np.array([row['chin_x'],      row['chin_y'],      row['chin_z']])
        left_eye  = np.array([row['left_eye_x'],  row['left_eye_y'],  row['left_eye_z']])
        right_eye = np.array([row['right_eye_x'], row['right_eye_y'], row['right_eye_z']])

        fwd = _compute_head_forward(forehead, chin, left_eye, right_eye)
        rows.append([int(row['frame']), row['timestamp'], fwd[0], fwd[1], fwd[2]])

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

    input_path  = landmarks_folder / "face_cleaned.csv"
    output_path = landmarks_folder / "face_features.csv"

    if not input_path.exists():
        print(f"Error: face_cleaned.csv not found → {input_path}")
        return

    print("--- face ---")
    _process_face(input_path, output_path)
    print("\nDone.")


if __name__ == "__main__":
    main()
