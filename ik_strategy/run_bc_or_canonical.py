'''
run_bc.py
=============================================================================
Runs a trained BC model (or the canonical trajectory) on the Reachy simulator.

Two modes:
  --mode bc        : generates the trajectory in real time via the
                     autoregressive BC loop, starting from Reachy's rest pose.
  --mode canonical : replays canonical.csv frame by frame (same logic as
                     run_simulation.py but reads from the dataset folder).

Both modes send joint angles and head gaze to Reachy at each frame.

Usage:
  # BC mode: exercise 1
  python run_bc.py --exercise 1 --mode bc

  # Canonical mode: exercise 1
  python run_bc.py --exercise 1 --mode canonical

  # Custom host (default: localhost)
  python run_bc.py --exercise 1 --mode bc --host 10.59.1.20

  # Override number of steps for BC (default: length of canonical.csv)
  python run_bc.py --exercise 1 --mode bc --steps 300

Input:
  BC mode:
    data/dataset/exercise_XXX/bc_model.pth
    data/dataset/exercise_XXX/scaler.pkl

  Canonical mode:
    data/dataset/exercise_XXX/canonical.csv
'''

import argparse
import pickle
import time
import numpy as np
import pandas as pd
from pathlib import Path

from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode

import torch
import torch.nn as nn

from config import DATA_ROOT, REST_POSE, JOINT_COLS, HEAD_COLS

import reachyController.timeSeries

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SIMULATOR_HOST  = 'localhost'
MIN_FRAME_DELAY = 0.033    # ~30 Hz
GOTO_DURATION   = 2.0      # seconds to move to first pose / rest pose

HEAD_NEUTRAL = (1.0, 0.0, 0.15)


# ---------------------------------------------------------------------------
# BCModel
# ---------------------------------------------------------------------------
class BCModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=256, n_layers=2):
        super().__init__()
        layers = []
        in_features = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_features, hidden_size), nn.ReLU()]
            in_features = hidden_size
        layers.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Reachy helpers
# ---------------------------------------------------------------------------
def _connect(host: str):
    from reachy_sdk import ReachySDK
    from reachyController.reachyController import ReachyController
    print(f'Connecting to Reachy at {host} ...')
    reachy = ReachySDK(host=host)
    rController = ReachyController(reachy)
    return rController


def _goto_rest(reachy) -> None:
    print('Returning to rest pose ...')
    goal = {}
    for col, val in REST_POSE.items():
        arm   = reachy.r_arm if col.startswith('r_') else reachy.l_arm
        joint   = getattr(arm, col, None)
        if joint is not None:
            goal[joint] = val
    if goal:
        goto(goal, duration=GOTO_DURATION, interpolation_mode=InterpolationMode.MINIMUM_JERK)
        time.sleep(GOTO_DURATION + 0.1)
    reachy.head.look_at(*HEAD_NEUTRAL, duration=GOTO_DURATION, interpolation_mode=InterpolationMode.MINIMUM_JERK)


def _goto_pose(reachy, q: np.ndarray, head: np.ndarray) -> None:
    '''Moves smoothly to a given pose before starting playback.'''
    goal = {}
    for i, col in enumerate(JOINT_COLS):
        arm   = reachy.r_arm if col.startswith('r_') else reachy.l_arm
        joint = getattr(arm, col, None)
        if joint is not None:
            goal[joint] = float(q[i])
    if goal:
        goto(goal, duration=GOTO_DURATION, interpolation_mode=InterpolationMode.MINIMUM_JERK)
        time.sleep(GOTO_DURATION + 0.1)
    if not np.any(np.isnan(head)):
        reachy.head.look_at(*head.tolist(), duration=GOTO_DURATION, interpolation_mode=InterpolationMode.MINIMUM_JERK)


def _send_frame(reachy, q: np.ndarray, head: np.ndarray, delay: float) -> None:
    '''Sends joint angles and head gaze for one frame.'''
    for i, col in enumerate(JOINT_COLS):
        arm   = reachy.r_arm if col.startswith('r_') else reachy.l_arm
        joint = getattr(arm, col, None)
        if joint is not None and not np.isnan(q[i]):
            joint.goal_position = float(q[i])

    if not np.any(np.isnan(head)):
        reachy.head.look_at(*head.tolist(), duration=float(delay),
                            interpolation_mode=InterpolationMode.MINIMUM_JERK)


# ---------------------------------------------------------------------------
# BC mode
# ---------------------------------------------------------------------------
def _run_bc(exercise_num: int, dataset_root: Path, host: str, n_steps: int) -> None:
    exercise_name = f'exercise_{exercise_num:03d}'
    output_dir    = dataset_root / exercise_name
    model_path    = output_dir / 'bc_model.pth'
    scaler_path   = output_dir / 'scaler.pkl'

    # Load model
    if not model_path.exists():
        print(f'Error: bc_model.pth not found -> {model_path}')
        print('Run train_bc.py --exercise first.')
        return
    if not scaler_path.exists():
        print(f'Error: scaler.pkl not found -> {scaler_path}')
        return

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    model = BCModel(checkpoint['state_dim'], checkpoint['action_dim'], checkpoint['hidden_size'], checkpoint['n_layers'])
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    state_dim  = checkpoint['state_dim']
    action_dim = checkpoint['action_dim']
    head_in_action = action_dim > len(JOINT_COLS)   # True if --head-action was used

    print(f'Model loaded: state_dim={state_dim}  action_dim={action_dim}')
    print(f'Best val loss (training): {checkpoint.get("val_loss", "N/A"):.6f}')

    # Determine number of steps
    # Default: use length of baseline.csv (ground truth). Fallback to canonical.csv, then to 300 if neither is available.
    if n_steps is None:
        baseline_path  = output_dir / 'baseline.csv'
        canonical_path = output_dir / 'canonical.csv'
        if baseline_path.exists():
            n_steps = len(pd.read_csv(baseline_path))
            print(f'Steps: {n_steps} (from baseline.csv)')
        elif canonical_path.exists():
            n_steps = len(pd.read_csv(canonical_path))
            print(f'Steps: {n_steps} (from canonical.csv - baseline.csv not found)')
        else:
            n_steps = 300
            print(f'Steps: {n_steps} (default - no baseline.csv or canonical.csv found)')

    # Initial state: rest pose + zero velocity + neutral head
    state = np.zeros(state_dim, dtype=np.float32)
    rest_q = np.array([REST_POSE[c] for c in JOINT_COLS], dtype=np.float32)
    state[:16]  = rest_q
    state[16:32] = 0.0
    if state_dim > 32:
        state[32:35] = HEAD_NEUTRAL

    # Connect and move to rest pose
    reachy = _connect(host)
    _goto_rest(reachy)
    print(f'\nStarting BC loop ({n_steps} steps) ...\n')

    q_current   = rest_q.copy()
    head_current = np.array(HEAD_NEUTRAL, dtype=np.float32)
    t_start      = time.perf_counter()

    for step in range(n_steps):
        # Normalise state and predict delta
        state_norm = scaler.transform([state])
        x_t        = torch.tensor(state_norm, dtype=torch.float32)

        with torch.no_grad():
            delta = model(x_t).numpy()[0]   # shape: (action_dim,)

        # Apply joint delta
        q_new = q_current + delta[:16]

        # Apply head delta (if model was trained with --head-action)
        if head_in_action and len(delta) >= 19:
            head_current = head_current + delta[16:19]

        # Update state for next step
        state[:16]   = q_new
        state[16:32] = q_new - q_current
        if state_dim > 32:
            state[32:35] = head_current

        q_current = q_new

        # Send to simulator
        _send_frame(reachy, q_new, head_current, MIN_FRAME_DELAY)

        if step % 50 == 0:
            elapsed = time.perf_counter() - t_start
            print(f'  Step {step+1:4d} / {n_steps}  ({elapsed:.1f}s)')

        time.sleep(MIN_FRAME_DELAY)

    elapsed_total = time.perf_counter() - t_start
    print(f'\nBC loop complete. Total time: {elapsed_total:.2f}s')

    # Return to rest
    _goto_rest(reachy)
    reachy.turn_off_smoothly('r_arm')
    reachy.turn_off_smoothly('l_arm')
    reachy.turn_off_smoothly('head')


# ---------------------------------------------------------------------------
# Canonical mode
# ---------------------------------------------------------------------------
def _run_canonical(exercise_num: int, dataset_root: Path, host: str) -> None:

    exercise_name  = f'exercise_{exercise_num:03d}'
    canonical_path = dataset_root / exercise_name / 'canonical.csv'

    if not canonical_path.exists():
        print(f'Error: canonical.csv not found -> {canonical_path}')
        print('Run compute_canonical.py --exercise first.')
        return

    df = pd.read_csv(canonical_path)
    print(f'Canonical trajectory loaded: {len(df)} frames')

    # Compute inter-frame delays from timestamps
    timestamps = df['timestamp'].to_numpy()
    dt = np.diff(timestamps)
    dt = np.clip(dt, MIN_FRAME_DELAY, None)
    dt = np.append(dt, MIN_FRAME_DELAY)

    # Connect
    reachy = _connect(host)

    reachy.turnOn()

    timeS = reachyController.timeSeries.TimeSeries.loadFromCSV(canonical_path)
    print(timeS.jointPosition)
    reachy.armLeft.playArmRecord(timeS)

    reachy.turnOffSmooth()
    """
    # Move to first frame pose
    q0   = df[JOINT_COLS].iloc[0].values
    head0 = df[HEAD_COLS].iloc[0].values if all(c in df.columns for c in HEAD_COLS) \
            else np.array(HEAD_NEUTRAL)
    print('Moving to first frame pose ...')
    _goto_pose(reachy, q0, head0)
    print('Ready. Starting playback ...\n')

    t_start = time.perf_counter()

    for i, (_, row) in enumerate(df.iterrows()):
        q    = row[JOINT_COLS].values.astype(float)
        head = row[HEAD_COLS].values.astype(float) \
               if all(c in row.index for c in HEAD_COLS) \
               else np.array(HEAD_NEUTRAL)

        _send_frame(reachy, q, head, float(dt[i]))

        if i % 50 == 0:
            elapsed = time.perf_counter() - t_start
            print(f'  Frame {i+1:4d} / {len(df)}  ({elapsed:.1f}s)')

        time.sleep(float(dt[i]))

    elapsed_total = time.perf_counter() - t_start
    print(f'\nPlayback complete. Total time: {elapsed_total:.2f}s')

    _goto_rest(reachy)
    reachy.turn_off_smoothly('r_arm')
    reachy.turn_off_smoothly('l_arm')
    reachy.turn_off_smoothly('head')
"""

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Run BC model or canonical trajectory on Reachy simulator.')
    parser.add_argument('--exercise', type=int, required=True,
        help='Exercise number.')
    parser.add_argument('--mode', choices=['bc', 'canonical'], default='bc',
        help='bc: autoregressive BC loop. canonical: replay canonical.csv. '
             '(default: bc)')
    parser.add_argument('--host', type=str, default=SIMULATOR_HOST,
        help=f'ReachySDK host (default: {SIMULATOR_HOST}).')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps for BC mode. '
             'Default: length of canonical.csv, or 300 if not found.')
    args = parser.parse_args()

    dataset_root = DATA_ROOT / 'dataset'

    if args.mode == 'bc':
        _run_bc(args.exercise, dataset_root, args.host, args.steps)
    else:
        _run_canonical(args.exercise, dataset_root, args.host)


if __name__ == '__main__':
    main()