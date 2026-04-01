'''
save_landmarks.py
=============================================================================
Handles CSV initialization and per-frame landmark saving for a recording session.

Four CSV files are created per video:
  - pose.csv       : world-space coordinates (meters, origin between hips) of 9 body joints + visibility score
  - right_hand.csv : normalized coordinates of 5 right hand landmarks
  - left_hand.csv  : normalized coordinates of 5 left hand landmarks
  - face.csv       : normalized image coordinates of 5 Face Mesh landmarks

Each row represents one frame and includes: frame index and timestamp (seconds since epoch)

Each video corresponds to a single movement, the context (subject, exercise, video) is encoded in the folder path, not in the CSV columns.

Rows with missing landmarks (landmark not detected) are written with only the metadata columns (frame, timestamp) and no coordinates;
these incomplete rows are discarded in the data cleaning phase.
'''

import csv
from pathlib import Path

from config import POSE_INDICES, HAND_INDICES, FACE_INDICES

# ---------------------------------------------------------------------------
# Header CSV
# ---------------------------------------------------------------------------
POSE_HEADER = ['frame', 'timestamp']
for name in POSE_INDICES:
    POSE_HEADER += [f'{name}_x', f'{name}_y', f'{name}_z', f'{name}_vis']

HAND_HEADER = ['frame', 'timestamp']
for name in HAND_INDICES:
    HAND_HEADER += [f'{name}_x', f'{name}_y', f'{name}_z']

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
        video_folder (str): folder path where to save the CSV files, e.g. "data/landmarks/subject_001/exercise_001/video_001"

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

    return paths


def save_frame(results, frame_idx, csv_paths, fps: float = 30.0):
    """
    Saves the landmarks of a single frame into the respective CSV files.

    If a landmark group is not detected, writes a row with only frame and timestamp.

    Parameters:
        results:     object returned by holistic.process()
        frame_idx:   integer index of the current frame
        csv_paths:   returned dict from init_csv_files()
    """
    timestamp = frame_idx / fps

    # POSE (world space, meters) 
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

    # HANDS (normalized image coords)
    for side, key in [('right_hand_landmarks', 'right_hand'), ('left_hand_landmarks',  'left_hand')]:
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

    # FACE (normalized image coords)
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
