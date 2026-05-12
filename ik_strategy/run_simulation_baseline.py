'''
run_simulation_baseline.py
=============================================================================
Runs baseline.csv for a given exercise on the Reachy simulator (Unity) or
the real robot, using ReachySDK directly (no ReachyController).

Usage:
  py run_simulation_baseline.py
  py run_simulation_baseline.py --exercise 1
  py run_simulation_baseline.py --exercise 1 --host 10.59.1.20

If --exercise is omitted, the script prompts interactively.
--host defaults to "localhost" (Unity simulator).

Input:
  _data/dataset/exercise_XXX/baseline.csv

Columns expected in baseline.csv:
  frame, timestamp,
  r_shoulder_pitch, r_shoulder_roll, r_arm_yaw, r_elbow_pitch,
  r_forearm_yaw,    r_wrist_pitch,   r_wrist_roll, r_gripper,
  l_shoulder_pitch, l_shoulder_roll, l_arm_yaw, l_elbow_pitch,
  l_forearm_yaw,    l_wrist_pitch,   l_wrist_roll, l_gripper
'''

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from utilities.config import DATA_ROOT, JOINT_COLS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIMULATOR_HOST  = 'localhost'
MIN_FRAME_DELAY = 0.033   # ~30 Hz
GOTO_DURATION   = 2.0     # seconds for smooth init / rest transitions


# ---------------------------------------------------------------------------
# ReachySDK helpers (same pattern as test_bc.py)
# ---------------------------------------------------------------------------
def _connect(host: str):
    from reachy_sdk import ReachySDK
    print(f'Connecting to Reachy at {host} ...')
    reachy = ReachySDK(host=host)
    reachy.turn_on('r_arm')
    reachy.turn_on('l_arm')
    print('Connected and motors on.\n')
    return reachy


def _get_joint(reachy, col: str):
    arm = reachy.r_arm if col.startswith('r_') else reachy.l_arm
    return getattr(arm, col)


def _goto_pose(reachy, pose: dict, duration: float) -> None:
    '''Moves both arms to {col: angle_deg} using minimum-jerk interpolation.'''
    from reachy_sdk.trajectory import goto
    from reachy_sdk.trajectory.interpolation import InterpolationMode
    goal = {_get_joint(reachy, col): float(val) for col, val in pose.items()}
    goto(goal, duration=duration, interpolation_mode=InterpolationMode.MINIMUM_JERK)
    time.sleep(duration + 0.1)


def _send_frame(reachy, row: dict) -> None:
    '''Writes goal_position for every joint present in JOINT_COLS.'''
    for col in JOINT_COLS:
        val = row.get(col)
        if val is not None and not np.isnan(val):
            _get_joint(reachy, col).goal_position = float(val)


def _shutdown(reachy) -> None:
    print('\nShutting down ...')
    reachy.turn_off_smoothly('r_arm')
    reachy.turn_off_smoothly('l_arm')
    print('Done.')


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------
def _run_baseline(reachy, csv_path: Path) -> None:
    if not csv_path.exists():
        print(f'[ERROR] File not found: {csv_path}')
        return

    df = pd.read_csv(csv_path)
    print(f'Loaded {len(df)} frames from {csv_path.name}')

    # ---- Move to the first frame pose before starting ----------------------
    first_row  = df.iloc[0]
    first_pose = {col: float(first_row[col])
                  for col in JOINT_COLS if col in first_row.index}
    print('Moving to start pose ...')
    _goto_pose(reachy, first_pose, GOTO_DURATION)

    # ---- Frame-by-frame playback at ~30 Hz ---------------------------------
    print('Starting playback ...')
    rows    = df.to_dict(orient='records')
    for _ in range(5):
        t_start = time.perf_counter()

        for step, row in enumerate(rows):
            t_target   = t_start + step * MIN_FRAME_DELAY
            _send_frame(reachy, row)
            sleep_time = t_target + MIN_FRAME_DELAY - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

    elapsed = time.perf_counter() - t_start
    print(f'Playback complete. Total time: {elapsed:.2f}s')

    # ---- Return to start pose ----------------------------------------------
    print('Returning to start pose ...')
    _goto_pose(reachy, first_pose, GOTO_DURATION)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Run baseline.csv on the Reachy simulator or real robot.')
    parser.add_argument('--exercise', type=int, default=None,
                        help='Exercise number.')
    parser.add_argument('--host', type=str, default=SIMULATOR_HOST,
                        help=f'ReachySDK host (default: {SIMULATOR_HOST}).')
    args = parser.parse_args()

    exercise = args.exercise
    if exercise is None:
        exercise = int(input('Exercise number: ').strip())

    csv_path = DATA_ROOT / 'dataset' / f'exercise_{exercise:03d}' / 'baseline.csv'
    print(f'Baseline : {csv_path}')
    print(f'Host     : {args.host}\n')

    reachy = _connect(args.host)
    try:
        _run_baseline(reachy, csv_path)
    finally:
        _shutdown(reachy)


if __name__ == '__main__':
    main()