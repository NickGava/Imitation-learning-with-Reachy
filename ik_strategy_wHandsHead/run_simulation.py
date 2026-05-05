'''
run_simulation.py
=============================================================================
Replays a recorded arm trajectory on the Reachy simulator (Unity via
ReachySDK) by reading joint_ik.csv and sending motor angles frame by frame.
The head gaze point (head_x, head_y, head_z) is sent via lookAt() each frame.

Usage:
  python run_simulation.py
  → prompts for subject / exercise / video numbers

Controls:
  The script plays back the trajectory once at the original recording speed
  (derived from the timestamp column). A minimum inter-frame delay of
  MIN_FRAME_DELAY seconds is enforced to avoid saturating the SDK.

Input:
  data/landmarks/subject_XXX/exercise_XXX/video_XXX/joint_ik.csv

Columns expected in joint_ik.csv:
  frame, timestamp,
  r_shoulder_pitch, r_shoulder_roll, r_arm_yaw,
  r_elbow_pitch,    r_forearm_yaw,
  r_wrist_pitch,    r_wrist_roll,
  r_gripper,
  l_shoulder_pitch, l_shoulder_roll, l_arm_yaw,
  l_elbow_pitch,    l_forearm_yaw,
  l_wrist_pitch,    l_wrist_roll,
  l_gripper,
  head_x, head_y, head_z

Notes:
  - The script connects to ReachySDK at host "localhost" (Unity simulator).
  - Before playback, Reachy moves smoothly to the first frame's pose
    (arms + head).
  - At the end of the trajectory, Reachy returns to rest pose and neutral
    head orientation.
  - Frames where ALL joint columns for a given arm are NaN are skipped for
    that arm (the other arm and the head are still updated).
  - Frames where head_x/y/z are NaN are skipped for lookAt() (head holds
    its previous position).
  - Joint angles are sent in degrees, as expected by the Reachy SDK.
  - Head gaze coordinates are in Reachy's torso frame (X forward, Y lateral
    right→left, Z up), in meters.
'''

import time
import numpy as np
import pandas as pd
from config import DATA_ROOT, HEAD_COLS
import threading

from ask_inputs import ask_inputs

from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIMULATOR_HOST  = "localhost"   # IP address of the Reachy simulator (Unity)

MIN_FRAME_DELAY = 0.02      # 50 Hz max

GOTO_REST_DURATION = 2.0


# Neutral head gaze: 1 m ahead, at shoulder height (torso frame)
HEAD_NEUTRAL_GAZE = (1.0, 0.0, 0.15)

REST_ANGLES = {
    'right': {
        'r_shoulder_pitch': -20.,
        'r_shoulder_roll':   0.,
        'r_arm_yaw':         0.,
        'r_elbow_pitch':     0.,
        'r_forearm_yaw':     0.,
        'r_wrist_pitch':     0.,
        'r_wrist_roll':      0.,
        'r_gripper':         0.,
    },
    'left': {
        'l_shoulder_pitch': -20.,
        'l_shoulder_roll':   0.,
        'l_arm_yaw':         0.,
        'l_elbow_pitch':     0.,
        'l_forearm_yaw':     0.,
        'l_wrist_pitch':     0.,
        'l_wrist_roll':      0.,
        'l_gripper':         0.,
    },
}

# JOINT_COLS = {
#     'right': ['r_shoulder_pitch', 'r_shoulder_roll', 'r_arm_yaw',
#               'r_elbow_pitch', 'r_forearm_yaw', 'r_wrist_pitch', 'r_wrist_roll', 'r_gripper'],
#     'left':  ['l_shoulder_pitch', 'l_shoulder_roll', 'l_arm_yaw',
#               'l_elbow_pitch', 'l_forearm_yaw', 'l_wrist_pitch', 'l_wrist_roll', 'l_gripper'],
# }
JOINT_COLS = {
    'right': ['r_shoulder_pitch', 'r_shoulder_roll', 'r_arm_yaw', 'r_elbow_pitch',],
    'left':  ['l_shoulder_pitch', 'l_shoulder_roll', 'l_arm_yaw', 'l_elbow_pitch',],
}

ARM_OBJ     = {'right': lambda r: r.r_arm, 'left': lambda r: r.l_arm}
GRIPPER_COL = {'right': 'r_gripper', 'left': 'l_gripper'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _goto_rest(reachy: ReachySDK) -> None:
    """Moves both arms smoothly to the rest pose and the head to neutral."""
    goal = {}
    for side, cols in JOINT_COLS.items():
        arm = ARM_OBJ[side](reachy)
        for col in cols:
            goal[getattr(arm, col)] = REST_ANGLES[side][col]

    goto(goal_positions=goal,
         duration=GOTO_REST_DURATION,
         interpolation_mode=InterpolationMode.MINIMUM_JERK)
    time.sleep(0.5)

    # Return head to neutral after arms have settled
    x, y, z = HEAD_NEUTRAL_GAZE
    reachy.head.look_at(x=x, y=y, z=z,
                        duration=GOTO_REST_DURATION,
                        interpolation_mode=InterpolationMode.MINIMUM_JERK
                        )


def _goto_first_frame(reachy: ReachySDK, first_row: pd.Series) -> None:
    """Moves both arms smoothly to the first frame's pose, then head."""
    goal = {}
    for side, cols in JOINT_COLS.items():
        arm = ARM_OBJ[side](reachy)
        for col in cols:
            val = first_row.get(col, np.nan)
            if not pd.isna(val):
                goal[getattr(arm, col)] = float(val)

    if goal:
        goto(goal_positions=goal,
             duration=GOTO_REST_DURATION,
             interpolation_mode=InterpolationMode.MINIMUM_JERK)
        time.sleep(0.5)

    # Move head to first frame gaze
    hx = first_row.get('head_x', np.nan)
    hy = first_row.get('head_y', np.nan)
    hz = first_row.get('head_z', np.nan)
    if not any(pd.isna(v) for v in [hx, hy, hz]):
        reachy.head.look_at(x=float(hx), y=float(hy), z=float(hz),
                            duration=GOTO_REST_DURATION,
                            interpolation_mode=InterpolationMode.MINIMUM_JERK
                            )


def _arm_columns_valid(row: pd.Series, side: str) -> bool:
    """Returns True if at least one joint column for this arm is not NaN."""
    return not all(pd.isna(row[c]) for c in JOINT_COLS[side])


import threading

def _send_frame(reachy: ReachySDK, row: pd.Series, frame_delay: float) -> None:
    
    # --- Arms (instantaneous goal_position, non-blocking) ---
    def _send_arms():
        for side, cols in JOINT_COLS.items():
            if not _arm_columns_valid(row, side):
                continue
            arm = ARM_OBJ[side](reachy)
            for col in cols:
                val = row[col]
                if not pd.isna(val):
                    getattr(arm, col).goal_position = float(val)
            g_col = GRIPPER_COL[side]
            if g_col in row and not pd.isna(row[g_col]):
                getattr(arm, g_col).goal_position = float(row[g_col])

    # --- Head (blocking look_at) ---
    def _send_head():
        if all(c in row.index for c in HEAD_COLS):
            hx, hy, hz = row['head_x'], row['head_y'], row['head_z']
            if not any(pd.isna(v) for v in [hx, hy, hz]):
                reachy.head.look_at(
                    x=float(hx), y=float(hy), z=float(hz),
                    duration=frame_delay,
                    interpolation_mode=InterpolationMode.LINEAR
                )

    t_head = threading.Thread(target=_send_head, daemon=True)
    t_head.start()
    _send_arms()          # main thread: non-blocking, ritorna subito
    # non fare join() — lascia che il thread della testa finisca in background
    # il time.sleep(frame_delay) nel loop principale fa già da sincronizzatore

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    bl_or_else = input("From dataset or landmarks? (d/l): ").strip().lower()

    if bl_or_else == 'd':
        exercise_num = int(input("Exercise number: ").strip())
        exercise_name = f"exercise_{exercise_num:03d}"
        folder = DATA_ROOT / "dataset" / exercise_name
        ik_path = folder / "canonical.csv"
    else:
        subject_name, exercise_name, video_name = ask_inputs()
        folder  = DATA_ROOT / "landmarks" / subject_name / exercise_name / video_name
        ik_path = folder / "joint_ik.csv"
    # ik_path = DATA_ROOT / "dataset" / "exercise_005" / "canonical.csv"

    if not ik_path.exists():
        print(f"Error: {ik_path.name} not found → {ik_path}")
        return

    df = pd.read_csv(ik_path)
    n_frames = len(df)
    print(f"Loaded {n_frames} frames from {ik_path}")

    has_head = all(c in df.columns for c in HEAD_COLS)
    if not has_head:
        print("[WARNING] head_x/y/z columns not found — head will not move.")

    # --- Compute per-frame delays from timestamps ---
    timestamps = df['timestamp'].to_numpy()
    dt = np.diff(timestamps)
    dt = np.clip(dt, MIN_FRAME_DELAY, None)
    dt = np.append(dt, MIN_FRAME_DELAY)

    # --- Connect to simulator ---
    print(f"Connecting to Reachy simulator at {SIMULATOR_HOST}…")
    reachy = ReachySDK(host=SIMULATOR_HOST)
    print("Connected.\n")

    # --- Turn on motors ---
    reachy.turn_on('r_arm')
    reachy.turn_on('l_arm')
    reachy.turn_on('head')

    # --- Move to first frame pose ---
    print("Moving to first frame pose…")
    _goto_first_frame(reachy, df.iloc[0])
    time.sleep(0.2)
    print("Ready. Starting playback…\n")

    # --- Playback loop ---
    t_start = time.perf_counter()
    n_repetitions = 1
    for i in range(n_repetitions):
        for i, (_, row) in enumerate(df.iterrows()):
            _send_frame(reachy, row, float(dt[i]))

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
    reachy.turn_off_smoothly('head')
    print("Done.")


if __name__ == "__main__":
    main()
