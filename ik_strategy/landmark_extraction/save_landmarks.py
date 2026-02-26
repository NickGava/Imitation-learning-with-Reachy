'''
save_landmarks.py
=============================================================================
Handles CSV initialization and per-frame landmark saving for a recording
session.

Three CSV files are created per session:
  - pose.csv       : world-space coordinates (meters, origin between hips)
                     of 9 body joints + visibility score
  - right_hand.csv : normalized coordinates of 5 right hand landmarks
  - left_hand.csv  : normalized coordinates of 5 left hand landmarks

Each row represents one frame and includes:
  - frame index and timestamp (seconds since epoch)
  - gesture_id  : incremented at each new recording (0 = idle)
  - gesture     : binary flag (0 = idle, 1 = recording active)

Rows with missing landmarks (landmark not detected) are written with only
the metadata columns (frame, timestamp, gesture_id, gesture) and no
coordinates — these incomplete rows are discarded in the data cleaning phase.
'''

import csv
import os
import time

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

# ---------------------------------------------------------------------------
# Header CSV
# ---------------------------------------------------------------------------
POSE_HEADER = ['frame', 'timestamp', 'gesture_id', 'gesture']
for name in POSE_INDICES:
    POSE_HEADER += [f'{name}_x', f'{name}_y', f'{name}_z', f'{name}_vis']

HAND_HEADER = ['frame', 'timestamp', 'gesture_id', 'gesture']
for name in HAND_INDICES:
    HAND_HEADER += [f'{name}_x', f'{name}_y', f'{name}_z']


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------
def init_csv_files(session_folder):
    """
    Creates the session folder and initializes the three CSV files with headers.
    Returns a dict with the file paths.

    Parameters:
        session_folder (str): folder path where to save the CSV file - es. "data/session_001"

    Returns:
        dict: {'pose': str, 'right_hand': str, 'left_hand': str}
    """
    os.makedirs(session_folder, exist_ok=True)

    paths = {
        'pose': os.path.join(session_folder, 'pose.csv'),
        'right_hand': os.path.join(session_folder, 'right_hand.csv'),
        'left_hand': os.path.join(session_folder, 'left_hand.csv'),
    }

    with open(paths['pose'], 'w', newline='') as f:
        csv.writer(f).writerow(POSE_HEADER)

    with open(paths['right_hand'], 'w', newline='') as f:
        csv.writer(f).writerow(HAND_HEADER)

    with open(paths['left_hand'], 'w', newline='') as f:
        csv.writer(f).writerow(HAND_HEADER)

    print(f"[save_landmarks] Session initialized in: {session_folder}")
    return paths


def save_frame(results, frame_idx, csv_paths, gesture_id, gesture_active):
    """
    Saves the landmarks of a single frame into the respective CSV files.
    Uses pose_world_landmarks (coordinates in meters, origin between hips).
    If a landmark is not detected, writes a row with only frame_idx, gesture_id, gesture
    (empty row = frame to discard in the data cleaning phase).

    Parameters:
        results:         object returned by holistic.process()
        frame_idx:       integer index of the current frame
        csv_paths:       returned dict from init_csv_files()
        gesture_id:      integer counter incremented at each new gesture recording
        gesture_active:  bool, True if gesture recording is active (S pressed)
    """
    gesture_flag = 1 if gesture_active else 0       # binary flag: 0 = idle, 1 = recording gesture
    timestamp = time.time()                         # seconds since epoch, e.g. 1709123456.789

    # --- POSE ---
    with open(csv_paths['pose'], 'a', newline='') as f:
        writer = csv.writer(f)
        if results.pose_world_landmarks:
            lms = results.pose_world_landmarks.landmark
            row = [frame_idx, timestamp, gesture_id, gesture_flag]
            for idx in POSE_INDICES.values():
                lm = lms[idx]
                row += [lm.x, lm.y, lm.z, lm.visibility]
            writer.writerow(row)
        else:
            writer.writerow([frame_idx, timestamp, gesture_id, gesture_flag])  # frame not detected

    # --- HANDS ---
    for side, key in [('right_hand_landmarks', 'right_hand'),
                      ('left_hand_landmarks',  'left_hand')]:
        with open(csv_paths[key], 'a', newline='') as f:
            writer = csv.writer(f)
            hand = getattr(results, side)
            if hand:
                lms = hand.landmark
                row = [frame_idx, timestamp, gesture_id, gesture_flag]
                for idx in HAND_INDICES.values():
                    lm = lms[idx]
                    row += [lm.x, lm.y, lm.z]
                writer.writerow(row)
            else:
                writer.writerow([frame_idx, timestamp, gesture_id, gesture_flag])  # hand not detected