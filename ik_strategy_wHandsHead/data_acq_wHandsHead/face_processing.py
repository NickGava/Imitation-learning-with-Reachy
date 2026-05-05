'''
face_processing.py
=============================================================================
Computes the head forward direction.

Uses 5 Face Mesh landmarks saved in face_cleaned.csv to build a proper 3-axis
head reference frame and extract the forward (gaze) direction:

  Frame construction:
    e_up           = normalize(forehead - chin)          physically upward
    e_left         = normalize(right_eye - left_eye)     person's right→left (image left→right)
    e_forward_face = normalize(cross(e_left, e_up))      normal to face plane, toward camera

Input:
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/face_cleaned.csv

Output (same folder):
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/face_features.csv

Each output row:
  frame, timestamp, head_dx, head_dy, head_dz
'''

import numpy as np
import pandas as pd

from config import DATA_ROOT
from ask_inputs import ask_inputs

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
def _compute_head_forward(forehead: np.ndarray, chin: np.ndarray, left_eye: np.ndarray, right_eye: np.ndarray) -> np.ndarray:
    """
    Returns the head forward unit vector.

    Parameters:
        forehead, chin, left_eye, right_eye: 3D positions in face image space
    """
    e_up = _normalize(forehead - chin)
    e_left = _normalize(right_eye - left_eye)
    e_forward_face = _normalize(np.cross(e_left, e_up))

    # Convert to pose world space and re-normalize
    return _normalize(np.array([e_forward_face[0], e_forward_face[1], e_forward_face[2]]))

# ---------------------------------------------------------------------------
# Per-file pipeline
# ---------------------------------------------------------------------------
def _process_face(input_path, output_path) -> None:
    """
    Reads face_cleaned.csv, computes the head forward direction for each frame, and writes face_features.csv.

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
    print(f"rows saved: {len(out)} / {n_loaded} → {output_path.relative_to(DATA_ROOT)}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    subject_name, exercise_name, video_name = ask_inputs()
    landmarks_folder = DATA_ROOT / "landmarks" / subject_name / exercise_name / video_name
    if not landmarks_folder.is_dir():
        print(f"Error: folder not found → {landmarks_folder}")
        return

    input_path  = landmarks_folder / "face_cleaned.csv"
    output_path = landmarks_folder / "face_features.csv"
    if not input_path.exists():
        print(f"Error: face_cleaned.csv not found → {input_path}")
        return

    print("\n--- face ---")
    _process_face(input_path, output_path)
    print("Done.")

if __name__ == "__main__":
    main()
