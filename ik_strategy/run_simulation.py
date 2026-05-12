'''
run_simulation.py
=============================================================================
Single entry point to run any trajectory source on the Reachy simulator.

Modes:
  video       — replays joint_ik.csv from landmarks
                (requires --subject, --exercise, --video)
  baseline    — replays baseline.csv from dataset/exercise_XXX/
                (requires --exercise)
  canonical   — replays canonical.csv from dataset/exercise_XXX/
                (requires --exercise)
  mlp         — BC MLP  ensemble inference → send to robot
                (requires --exercise)
  gru         — BC GRU  ensemble inference → send to robot
                (requires --exercise)
  transformer — BC Transformer ensemble inference → send to robot
                (requires --exercise)

If no argument is provided, the script will prompt interactively.

Usage:
  py run_simulation.py
  py run_simulation.py --mode canonical  --exercise 1
  py run_simulation.py --mode video      --subject 1 --exercise 2 --video 3
  py run_simulation.py --mode mlp        --exercise 1
  py run_simulation.py --mode gru        --exercise 1 --runs 3
  py run_simulation.py --mode transformer --exercise 1 --host 10.59.1.20
  py run_simulation.py --mode mlp        --exercise 1 --steps 200
'''

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from reachy_sdk import ReachySDK
from reachyController.reachyController import ReachyController
from reachyController.timeSeries import TimeSeries
from reachyController import config as rc_config

from utilities.config import DATA_ROOT, JOINT_COLS
from utilities.ask_inputs import ask_inputs

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIMULATOR_HOST  = 'localhost'
MIN_FRAME_DELAY = 0.033    # ~30 Hz
GOTO_DURATION   = 2.0      # seconds for smooth init / rest transitions

VALID_MODES     = ['video', 'baseline', 'canonical', 'mlp', 'gru', 'transformer']
BC_MODES        = {'mlp', 'gru', 'transformer'}
CSV_MODES       = {'video', 'baseline', 'canonical'}

# All arm joint names expected by ReachyArm (8 per side, 16 total).
# This matches utilities/config.py JOINT_COLS exactly.
ALL_ARM_JOINTS = [
    'r_shoulder_pitch', 'r_shoulder_roll', 'r_arm_yaw', 'r_elbow_pitch',
    'r_forearm_yaw', 'r_wrist_pitch', 'r_wrist_roll', 'r_gripper',
    'l_shoulder_pitch', 'l_shoulder_roll', 'l_arm_yaw', 'l_elbow_pitch',
    'l_forearm_yaw', 'l_wrist_pitch', 'l_wrist_roll', 'l_gripper',
]


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------
def _connect(host: str) -> ReachyController:
    print(f'Connecting to Reachy at {host} ...')
    reachy    = ReachySDK(host=host)
    reachyC   = ReachyController(reachy)
    reachyC.turnOn()
    print('Connected and motors on.\n')
    return reachyC


def _shutdown(reachyC: ReachyController) -> None:
    print('\nShutting down ...')
    reachyC.turnOffSmooth()
    print('Done.')


# ---------------------------------------------------------------------------
# Motion helpers
# ---------------------------------------------------------------------------
def _build_arm_dicts(reachyC: ReachyController, pose: dict) -> tuple:
    '''
    Splits a {col: angle} pose dict into two {joint_obj: angle} dicts,
    one per arm. Missing joints default to 0.0.
    '''
    right_dict, left_dict = {}, {}
    for col, val in pose.items():
        if col.startswith('r_'):
            joint = reachyC.armRight._joints.get(col)
            if joint is not None:
                right_dict[joint] = float(val)
        elif col.startswith('l_'):
            joint = reachyC.armLeft._joints.get(col)
            if joint is not None:
                left_dict[joint] = float(val)
    return right_dict, left_dict


def _goto_pose(reachyC: ReachyController, pose: dict, duration: float) -> None:
    '''
    Moves both arms simultaneously to a given {col: angle} pose.
    Uses _debug_goto (no collision check) since trajectories are trusted.
    Blocks until both arms have settled.
    '''
    right_dict, left_dict = _build_arm_dicts(reachyC, pose)
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = []
        if right_dict:
            futures.append(ex.submit(reachyC.armRight._debug_goto, right_dict, duration))
        if left_dict:
            futures.append(ex.submit(reachyC.armLeft._debug_goto, left_dict, duration))
        for f in futures:
            f.result()
    time.sleep(0.2)


def _send_q(reachyC: ReachyController, q: np.ndarray) -> None:
    '''
    Non-blocking: writes goal_position for all joints in JOINT_COLS.
    Used for the BC frame-by-frame loop at 30 Hz.
    '''
    for i, col in enumerate(JOINT_COLS):
        arm   = reachyC.armRight if col.startswith('r_') else reachyC.armLeft
        joint = arm._joints.get(col)
        if joint is not None:
            joint.goal_position = float(q[i])


# ---------------------------------------------------------------------------
# CSV → TimeSeries
# ---------------------------------------------------------------------------
def _csv_to_timeseries(df: pd.DataFrame) -> TimeSeries:
    '''
    Converts a DataFrame with (frame, timestamp, *JOINT_COLS) columns into
    a TimeSeries that reachyC.playRecord() can consume.

    Joints not present in the CSV are set to 0.0 so that ReachyArm's FK
    (used by _safeGoto internally) always receives all required joints.
    '''
    joint_position = []
    for _, row in df.iterrows():
        frame = {col: (float(row[col]) if col in row.index else 0.0)
                 for col in ALL_ARM_JOINTS}
        frame['timestamp'] = float(row['timestamp'])
        joint_position.append(frame)

    timestamps = df['timestamp'].values
    sf         = 1.0 / float(np.median(np.diff(timestamps)))
    duration   = float(timestamps[-1] - timestamps[0])
    return TimeSeries(sf, duration, joint_position)


# ---------------------------------------------------------------------------
# Playback modes (video / baseline / canonical)
# ---------------------------------------------------------------------------
def _run_csv(reachyC: ReachyController, csv_path: Path) -> None:
    '''
    Loads a CSV trajectory and replays it on the robot via reachyC.playRecord().
    Both arms are driven simultaneously (ThreadPoolExecutor inside playRecord).
    '''
    if not csv_path.exists():
        print(f'[ERROR] File not found: {csv_path}')
        return

    df = pd.read_csv(csv_path)
    print(f'Loaded {len(df)} frames from {csv_path.name}')

    ts = _csv_to_timeseries(df)

    print('Starting playback ...')
    t_start = time.perf_counter()
    reachyC.playRecord(ts)
    elapsed = time.perf_counter() - t_start
    print(f'Playback complete. Total time: {elapsed:.2f}s')


# ---------------------------------------------------------------------------
# BC inference loop (common for all three architectures)
# ---------------------------------------------------------------------------
def _run_bc_loop_on_robot(
    reachyC  : ReachyController,
    q_traj   : np.ndarray,
    start_pose: dict,
) -> None:
    '''
    Sends a pre-computed q_traj (n_steps, N_JOINTS) to the robot at ~30 Hz.
    Moves to start pose before playback and returns to it afterwards.
    '''
    print(f'Moving to start pose ...')
    _goto_pose(reachyC, start_pose, GOTO_DURATION)

    n_steps = len(q_traj)
    print(f'Sending {n_steps} frames to robot at ~30 Hz ...')
    t_start = time.perf_counter()
    for step in range(n_steps):
        t_target = t_start + step * MIN_FRAME_DELAY
        _send_q(reachyC, q_traj[step])
        sleep_time = t_target + MIN_FRAME_DELAY - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)

    elapsed = time.perf_counter() - t_start
    print(f'Loop complete. Total time: {elapsed:.2f}s')

    print('Returning to start pose ...')
    _goto_pose(reachyC, start_pose, GOTO_DURATION)


def _run_bc(
    reachyC     : ReachyController,
    arch        : str,           # 'mlp', 'gru', 'transformer'
    exercise_dir: Path,
    n_steps,                     # int or None
    runs        : int,
) -> None:
    '''
    Loads the BC ensemble for the given architecture, runs the autoregressive
    inference loop (offline), then sends each run to the robot.
    '''
    # ---- Import the relevant test_bc module --------------------------------
    if arch == 'mlp':
        from bc_approach.MLP.test_bc import load_ensemble, run_bc_loop, _load_start_pose
    elif arch == 'gru':
        from bc_approach.GRU.test_bc import load_ensemble, run_bc_loop, _load_start_pose
    elif arch == 'transformer':
        from bc_approach.Transformer.test_bc import load_ensemble, run_bc_loop, _load_start_pose
    else:
        raise ValueError(f'Unknown BC architecture: {arch}')

    model_dir  = exercise_dir / arch.upper()
    ensemble   = load_ensemble(model_dir)
    start_pose = _load_start_pose(exercise_dir)

    # ---- Resolve n_steps ---------------------------------------------------
    if n_steps is not None:
        print(f'Steps: {n_steps}  (from --steps)')
    elif (exercise_dir / 'canonical.csv').exists():
        n_steps = len(pd.read_csv(exercise_dir / 'canonical.csv'))
        print(f'Steps: {n_steps}  (from canonical.csv)')
    elif (exercise_dir / 'baseline.csv').exists():
        n_steps = len(pd.read_csv(exercise_dir / 'baseline.csv'))
        print(f'Steps: {n_steps}  (from baseline.csv)')
    else:
        n_steps = 300
        print(f'Steps: {n_steps}  (default)')

    # ---- Run ---------------------------------------------------------------
    for run in range(1, runs + 1):
        print(f'\n--- Run {run} / {runs} ---')
        q_traj = run_bc_loop(ensemble, n_steps, start_pose)   # offline, fast
        _run_bc_loop_on_robot(reachyC, q_traj, start_pose)


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------
def _prompt_mode() -> str:
    print('Select mode:')
    for i, m in enumerate(VALID_MODES, 1):
        print(f'  {i}. {m}')
    raw = input('Mode (name or number): ').strip().lower()
    # Accept name or 1-based index
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(VALID_MODES):
            return VALID_MODES[idx]
    if raw in VALID_MODES:
        return raw
    print(f'[ERROR] Invalid mode: "{raw}"')
    sys.exit(1)


def _prompt_exercise() -> int:
    return int(input('Exercise number: ').strip())


def _prompt_subject() -> int:
    return int(input('Subject number: ').strip())


def _prompt_video() -> int:
    return int(input('Video number: ').strip())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Run any trajectory source on the Reachy simulator.')

    parser.add_argument('--mode',     choices=VALID_MODES, default=None,
                        help='Trajectory source (default: prompted interactively).')
    parser.add_argument('--exercise', type=int, default=None,
                        help='Exercise number.')
    parser.add_argument('--subject',  type=int, default=None,
                        help='Subject number (video mode only).')
    parser.add_argument('--video',    type=int, default=None,
                        help='Video number (video mode only).')
    parser.add_argument('--host',     type=str, default=SIMULATOR_HOST,
                        help=f'ReachySDK host (default: {SIMULATOR_HOST}).')
    parser.add_argument('--runs',     type=int, default=1,
                        help='Number of BC inference runs sent to the robot (default: 1).')
    parser.add_argument('--steps',    type=int, default=None,
                        help='Override number of BC inference steps.')
    args = parser.parse_args()

    # ---- Resolve mode ------------------------------------------------------
    mode = args.mode or _prompt_mode()

    # ---- Resolve paths based on mode ---------------------------------------
    if mode == 'video':
        subject  = args.subject  or _prompt_subject()
        exercise = args.exercise or _prompt_exercise()
        video    = args.video    or _prompt_video()
        csv_path = (DATA_ROOT / 'landmarks'
                    / f'subject_{subject:03d}'
                    / f'exercise_{exercise:03d}'
                    / f'video_{video:03d}'
                    / 'joint_ik.csv')
        exercise_dir = None

    elif mode in CSV_MODES:   # baseline or canonical
        exercise     = args.exercise or _prompt_exercise()
        exercise_dir = DATA_ROOT / 'dataset' / f'exercise_{exercise:03d}'
        csv_path     = exercise_dir / f'{mode}.csv'

    else:   # mlp / gru / transformer
        exercise     = args.exercise or _prompt_exercise()
        exercise_dir = DATA_ROOT / 'dataset' / f'exercise_{exercise:03d}'
        csv_path     = None

    # ---- Connect -----------------------------------------------------------
    reachyC = _connect(args.host)

    try:
        if mode in CSV_MODES or mode == 'video':
            _run_csv(reachyC, csv_path)
        else:
            _run_bc(reachyC, mode, exercise_dir, args.steps, args.runs)

    finally:
        _shutdown(reachyC)


if __name__ == '__main__':
    main()