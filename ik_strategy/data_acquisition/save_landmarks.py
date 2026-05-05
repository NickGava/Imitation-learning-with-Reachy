'''
save_landmarks.py
=============================================================================
Handles CSV initialization and per-frame landmark saving for a recording session.

One CSV file is created per video:
  - pose.csv : world-space coordinates (metres, origin between hips) of body
               joints + visibility score, produced by MediaPipe Pose.

Each row represents one frame and includes: frame index and timestamp (seconds).

Each video corresponds to a single movement; context (subject, exercise, video)
is encoded in the folder path, not in the CSV columns.

Rows with missing landmarks (pose not detected) are written with only the
metadata columns (frame, timestamp); these incomplete rows are discarded in the
data cleaning phase.
'''

import csv
from pathlib import Path

from config import POSE_INDICES

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
POSE_HEADER = ['frame', 'timestamp']
for name in POSE_INDICES:
    POSE_HEADER += [f'{name}_x', f'{name}_y', f'{name}_z', f'{name}_vis']


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------
def init_csv_files(video_folder):
    """
    Creates the video folder and initializes pose.csv with its header.

    Parameters:
        video_folder : folder path where to save the CSV, e.g.
                       "data/landmarks/subject_001/exercise_001/video_001"

    Returns:
        dict: {'pose': Path}
    """
    video_folder = Path(video_folder)
    video_folder.mkdir(parents=True, exist_ok=True)

    paths = {'pose': video_folder / 'pose.csv'}

    with open(paths['pose'], 'w', newline='') as f:
        csv.writer(f).writerow(POSE_HEADER)

    return paths


def save_frame(results, frame_idx, csv_paths, fps: float = 30.0):
    """
    Saves the pose landmarks of a single frame into pose.csv.

    Uses pose_world_landmarks (metric, hip-centred) from MediaPipe Pose.
    If pose is not detected, writes a row with only frame and timestamp.

    Parameters:
        results    : object returned by mp.solutions.pose.Pose.process()
        frame_idx  : integer index of the current frame
        csv_paths  : dict returned by init_csv_files()
        fps        : frame rate of the source video
    """
    timestamp = frame_idx / fps

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
            writer.writerow([frame_idx, timestamp])     # pose not detected