'''
save_landmarks.py
=============================================================================
Handles CSV initialization and per-frame landmark saving for a recording
session.

Four CSV files are created per video:
  - pose.csv       : world-space coordinates (meters, origin between hips)
                     of 9 body joints + visibility score
  - right_hand.csv : normalized coordinates of 5 right hand landmarks
  - left_hand.csv  : normalized coordinates of 5 left hand landmarks
  - face.csv       : normalized image coordinates of 5 Face Mesh landmarks
                     used to compute head orientation in face_processing.py

Each row represents one frame and includes:
  - frame index and timestamp (seconds since epoch)

Each video corresponds to a single movement — the context (subject, exercise,
video) is encoded in the folder path, not in the CSV columns.

Rows with missing landmarks (landmark not detected) are written with only
the metadata columns (frame, timestamp) and no coordinates — these incomplete
rows are discarded in the data cleaning phase.
'''

import csv
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Relevant landmark indices
# ---------------------------------------------------------------------------
POSE_INDICES = {
    'nose':            0,
    'left_shoulder':  11,
    'right_shoulder': 12,
    'left_elbow':     13,
    'right_elbow':    14,
    'left_wrist':     15,
    'right_wrist':    16,
    'left_hip':       23,
    'right_hip':      24,
}

HAND_INDICES = {
    'thumb_tip':  4,   # hand closure
    'index_tip':  8,   # hand closure
    'wrist':      0,   # hand orientation
    'index_mcp':  5,   # hand orientation
    'pinky_mcp': 17,   # hand orientation
}

# MediaPipe Face Mesh landmark indices.
# These landmarks come from results.face_landmarks (normalized image coords),
# NOT from pose_world_landmarks.  Coordinates:
#   x, y : normalized [0, 1] — x=0 left, y=0 top of image
#   z    : relative depth, smaller = closer to camera
FACE_INDICES = {
    'nose_tip':   4,    # approximate face origin
    'chin':     152,    # defines vertical axis (bottom)
    'forehead':  10,    # defines vertical axis (top)
    'left_eye':  33,    # defines lateral axis (right in image = person's left)
    'right_eye': 263,   # defines lateral axis (left in image = person's right)
}

# ---------------------------------------------------------------------------
# Header CSV
# ---------------------------------------------------------------------------
POSE_HEADER = ['frame', 'timestamp']
for name in POSE_INDICES:
    POSE_HEADER += [f'{name}_x', f'{name}_y', f'{name}_z', f'{name}_vis']

HAND_HEADER = ['frame', 'timestamp']
for name in HAND_INDICES:
    HAND_HEADER += [f'{name}_x', f'{name}_y', f'{name}_z']

# Face landmarks have no visibility score
FACE_HEADER = ['frame', 'timestamp']
for name in FACE_INDICES:
    FACE_HEADER += [f'{name}_x', f'{name}_y', f'{name}_z']


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------
def init_csv_files(video_folder):
    """
    Creates the video folder and initializes the four CSV files with headers.
    Returns a dict with the file paths.

    Parameters:
        video_folder (str): folder path where to save the CSV files,
                            e.g. "data/landmarks/subject_001/exercise_001/video_001"

    Returns:
        dict: {'pose': Path, 'right_hand': Path, 'left_hand': Path, 'face': Path}
    """
    video_folder = Path(video_folder)
    video_folder.mkdir(parents=True, exist_ok=True)

    paths = {
        'pose':       video_folder / 'pose.csv',
        'right_hand': video_folder / 'right_hand.csv',
        'left_hand':  video_folder / 'left_hand.csv',
        'face':       video_folder / 'face.csv',
    }

    with open(paths['pose'], 'w', newline='') as f:
        csv.writer(f).writerow(POSE_HEADER)

    with open(paths['right_hand'], 'w', newline='') as f:
        csv.writer(f).writerow(HAND_HEADER)

    with open(paths['left_hand'], 'w', newline='') as f:
        csv.writer(f).writerow(HAND_HEADER)

    with open(paths['face'], 'w', newline='') as f:
        csv.writer(f).writerow(FACE_HEADER)

    print(f"[save_landmarks] Initialized in: {video_folder}")
    return paths


def save_frame(results, frame_idx, csv_paths, fps: float = 30.0):
    """
    Saves the landmarks of a single frame into the respective CSV files.

    Pose: uses pose_world_landmarks (coordinates in meters, origin between hips).
    Face: uses face_landmarks (normalized image coordinates, no world space).

    If a landmark group is not detected, writes a row with only frame and
    timestamp (empty row = frame to discard in the data cleaning phase).

    Parameters:
        results:     object returned by holistic.process()
        frame_idx:   integer index of the current frame
        csv_paths:   returned dict from init_csv_files()
    """
    timestamp = frame_idx / fps

    # --- POSE (world space, meters) ---
    with open(csv_paths['pose'], 'a', newline='') as f:
        writer = csv.writer(f)
        if results.pose_world_landmarks:
            lms = results.pose_world_landmarks.landmark
            row = [frame_idx, timestamp]
            for idx in POSE_INDICES.values():
                lm = lms[idx]
                row += [lm.x, lm.y, lm.z, lm.visibility]
            writer.writerow(row)
        else:
            writer.writerow([frame_idx, timestamp])     # frame not detected

    # --- HANDS (normalized image coords) ---
    for side, key in [('right_hand_landmarks', 'right_hand'),
                      ('left_hand_landmarks',  'left_hand')]:
        with open(csv_paths[key], 'a', newline='') as f:
            writer = csv.writer(f)
            hand = getattr(results, side)
            if hand:
                lms = hand.landmark
                row = [frame_idx, timestamp]
                for idx in HAND_INDICES.values():
                    lm = lms[idx]
                    row += [lm.x, lm.y, lm.z]
                writer.writerow(row)
            else:
                writer.writerow([frame_idx, timestamp])     # hand not detected

    # --- FACE (normalized image coords, Face Mesh 468 landmarks) ---
    with open(csv_paths['face'], 'a', newline='') as f:
        writer = csv.writer(f)
        if results.face_landmarks:
            lms = results.face_landmarks.landmark
            row = [frame_idx, timestamp]
            for idx in FACE_INDICES.values():
                lm = lms[idx]
                row += [lm.x, lm.y, lm.z]
            writer.writerow(row)
        else:
            writer.writerow([frame_idx, timestamp])         # face not detected
