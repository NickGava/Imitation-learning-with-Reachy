'''
run_simulation.py
=============================================================================
Replays a recorded arm trajectory on the Reachy simulator (Unity via
ReachySDK) by reading arms_ik.csv and sending motor angles frame by frame.

Usage:
  python run_simulation.py
  → prompts for subject / exercise / video numbers

Controls:
  The script plays back the trajectory once at the original recording speed
  (derived from the timestamp column).  A minimum inter-frame delay of
  MIN_FRAME_DELAY seconds is enforced to avoid saturating the SDK.

Input:
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/arms_ik.csv

Columns expected in arms_ik.csv:
  frame, timestamp,
  r_shoulder_pitch, r_shoulder_roll, r_arm_yaw,
  r_elbow_pitch,    r_forearm_yaw,
  r_wrist_pitch,    r_wrist_roll,
  r_gripper_angle,
  l_shoulder_pitch, l_shoulder_roll, l_arm_yaw,
  l_elbow_pitch,    l_forearm_yaw,
  l_wrist_pitch,    l_wrist_roll,
  l_gripper_angle

Notes:
  - The script connects to ReachySDK at host "localhost" (Unity simulator).
  - Before playback Reachy is moved to its rest pose smoothly.
  - At the end of the trajectory Reachy returns to rest pose.
  - Frames where ALL joint columns for a given arm are NaN are skipped for
    that arm (the other arm is still updated).
  - Joint angles are sent in degrees, as expected by the Reachy SDK.
'''

import time
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DATA_ROOT

from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIMULATOR_HOST  = "localhost"

# Minimum time between consecutive frames (seconds).
# Prevents sending commands faster than the simulator can process them.
MIN_FRAME_DELAY = 0.02      # 50 Hz max

# Duration (seconds) for the initial / final rest-pose movement
GOTO_REST_DURATION = 2.0

# Rest pose angles (degrees) — used at start and end of playback
REST_ANGLES = {
    'right': {
        'r_shoulder_pitch':  0.,
        'r_shoulder_roll':  -5.,
        'r_arm_yaw':         0.,
        'r_elbow_pitch':   -90.,
        'r_forearm_yaw':     0.,
        'r_wrist_pitch':     0.,
        'r_wrist_roll':      0.,
    },
    'left': {
        'l_shoulder_pitch':  0.,
        'l_shoulder_roll':   5.,
        'l_arm_yaw':         0.,
        'l_elbow_pitch':   -90.,
        'l_forearm_yaw':     0.,
        'l_wrist_pitch':     0.,
        'l_wrist_roll':      0.,
    },
}

# Joint column names per arm (must match both CSV columns and SDK attribute names)
# e.g. 'r_shoulder_pitch' → getattr(reachy.r_arm, 'r_shoulder_pitch')
JOINT_COLS = {
    'right': ['r_shoulder_pitch', 'r_shoulder_roll', 'r_arm_yaw',
              'r_elbow_pitch', 'r_forearm_yaw', 'r_wrist_pitch', 'r_wrist_roll'],
    'left':  ['l_shoulder_pitch', 'l_shoulder_roll', 'l_arm_yaw',
              'l_elbow_pitch', 'l_forearm_yaw', 'l_wrist_pitch', 'l_wrist_roll'],
}

ARM_OBJ  = {'right': lambda r: r.r_arm, 'left': lambda r: r.l_arm}
GRIPPER_COL = {'right': 'r_gripper_angle', 'left': 'l_gripper_angle'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _goto_rest(reachy: ReachySDK) -> None:
    """Moves both arms smoothly to the rest pose."""
    goal = {}
    for side, cols in JOINT_COLS.items():
        arm = ARM_OBJ[side](reachy)
        for col in cols:
            goal[getattr(arm, col)] = REST_ANGLES[side][col]

    goto(goal_positions=goal,
         duration=GOTO_REST_DURATION,
         interpolation_mode=InterpolationMode.MINIMUM_JERK)
    time.sleep(GOTO_REST_DURATION + 0.1)


def _arm_columns_valid(row: pd.Series, side: str) -> bool:
    """Returns True if at least one joint column for this arm is not NaN."""
    return not all(pd.isna(row[c]) for c in JOINT_COLS[side])


def _send_frame(reachy: ReachySDK, row: pd.Series) -> None:
    """
    Sends joint goals for one frame to the simulator.
    Skips an arm if all its joint columns are NaN.
    """
    for side, cols in JOINT_COLS.items():
        if not _arm_columns_valid(row, side):
            continue

        arm = ARM_OBJ[side](reachy)
        for col in cols:
            val = row[col]
            if not pd.isna(val):
                getattr(arm, col).goal_position = float(val)

        # Gripper
        g_col = GRIPPER_COL[side]
        if g_col in row and not pd.isna(row[g_col]):
            arm.gripper.goal_position = float(row[g_col])


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
    ik_path = folder / "arms_ik.csv"

    if not ik_path.exists():
        print(f"Error: arms_ik.csv not found → {ik_path}")
        return

    df = pd.read_csv(ik_path)
    n_frames = len(df)
    print(f"Loaded {n_frames} frames from {ik_path}")

    # --- Compute per-frame delays from timestamps ---
    timestamps = df['timestamp'].to_numpy()
    # dt[i] = time to wait AFTER sending frame i before sending frame i+1
    dt = np.diff(timestamps)
    dt = np.clip(dt, MIN_FRAME_DELAY, None)   # enforce minimum delay
    dt = np.append(dt, MIN_FRAME_DELAY)        # last frame: use minimum

    # --- Connect to simulator ---
    print(f"Connecting to Reachy simulator at {SIMULATOR_HOST}…")
    reachy = ReachySDK(host=SIMULATOR_HOST)
    print("Connected.\n")

    # --- Turn on motors ---
    reachy.turn_on('r_arm')
    reachy.turn_on('l_arm')

    # --- Move to rest pose ---
    print("Moving to rest pose…")
    _goto_rest(reachy)
    print("Ready. Starting playback…\n")

    # --- Playback loop ---
    t_start = time.perf_counter()

    for i, (_, row) in enumerate(df.iterrows()):
        _send_frame(reachy, row)

        # Progress indicator every 50 frames
        if i % 50 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  Frame {i + 1:4d} / {n_frames}  ({elapsed:.1f}s elapsed)")

        time.sleep(float(dt[i]))

    elapsed_total = time.perf_counter() - t_start
    print(f"\nPlayback complete. Total time: {elapsed_total:.2f}s")

    # --- Return to rest pose ---
    print("Returning to rest pose…")
    _goto_rest(reachy)

    # --- Turn off motors ---
    reachy.turn_off_smoothly('r_arm')
    reachy.turn_off_smoothly('l_arm')
    print("Done.")


if __name__ == "__main__":
    main()